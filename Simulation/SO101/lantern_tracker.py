"""
SO101 Smart Lantern Tracker v10
=================================
- Camara cenital simulada (top_cam) detecta libro abierto
- IK completo: brazo se estira/encoge
- Voz con Gemini → intents ricos (action/weather/question)
- Comandos locales sin gastar cuota Gemini
- TTS con pyttsx3 (respuestas habladas)
- Clima con OpenWeatherMap
- Timeout 10s sin libro → apaga luz

Instalar:
  pip install opencv-python pyttsx3 requests

Controles:
  WASD  = mover libro    O = abrir/cerrar
  L     = luz manual     F = debug camara
  V     = voz Gemini     R = reset
  P     = pausa          Esc = salir
"""

import os, time, json, threading, msvcrt
import pathlib, requests

import mujoco, mujoco.viewer
import numpy as np, cv2

# ── Env ──────────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    _env = pathlib.Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=_env, override=True)
    if _env.exists(): print(f"[ENV] Cargado: {_env}")
except ImportError:
    pass

try:
    import speech_recognition as sr
    SR_OK = True
except ImportError:
    sr = None; SR_OK = False

try:
    from google import genai
    GENAI_OK = True
except ImportError:
    genai = None; GENAI_OK = False

try:
    import pyttsx3
    TTS_OK = True
except ImportError:
    pyttsx3 = None; TTS_OK = False

# ── Config ────────────────────────────────────────────────────────────────────
XML_PATH     = "my_smart_lantern_scene.xml"
DT           = 0.002
MOVE_SPEED   = 0.005
GEMINI_MODEL = "gemini-2.5-flash"

TABLE_SURFACE = 0.775
BOOK_HALF_H   = 0.014
BOOK_Z        = TABLE_SURFACE + BOOK_HALF_H
BOOK_START    = np.array([0.42, 0.0])

CAM_NAME  = "top_cam"
CAM_W, CAM_H = 320, 240

# Deteccion libro abierto (blanco/claro desde arriba)
WHITE_V    = 180
WHITE_S    = 60
MIN_AREA   = 1500
MIN_ASPECT = 0.4
MAX_ASPECT = 2.8
DEADBAND   = 0.12
TIMEOUT_S  = 10.0
YAW_GAIN   = 0.004
PITCH_GAIN = 0.003

# IK
BASE_HEIGHT      = 0.062
ARM1, ARM2, TOOL = 0.113, 0.135, 0.150
LIMITS = np.array([[-1.91986,1.91986],[-1.74533,1.74533],[-1.69,1.69],
                   [-1.65806,1.65806],[-2.74385,2.84121],[-0.17453,1.74533]])
REST = np.array([0.0, 1.0, -1.69, 0.8, 0.0, 0.4])
JOINT_NAMES = ["shoulder_pan","shoulder_lift","elbow_flex",
               "wrist_flex","wrist_roll","gripper"]

# ── Estado compartido ─────────────────────────────────────────────────────────
state = {
    "move":      np.zeros(2),
    "paused":    False, "quit":     False,
    "reset":     False, "book_open":False,
    "toggled":   False, "lamp_on":  False,
    "voice_req": False, "cam_debug":True,
}
lock = threading.Lock()
last_seen_time = 0.0
timed_out      = False

# ── TTS ───────────────────────────────────────────────────────────────────────
_tts_engine = None
_tts_lock   = threading.Lock()

def init_tts():
    global _tts_engine
    if not TTS_OK: print("[TTS] pyttsx3 no instalado. pip install pyttsx3"); return
    try:
        _tts_engine = pyttsx3.init()
        _tts_engine.setProperty("rate", 170)
        _tts_engine.setProperty("volume", 1.0)
        # Intentar voz en español
        for v in _tts_engine.getProperty("voices"):
            if "es" in v.id.lower() or "spanish" in v.name.lower():
                _tts_engine.setProperty("voice", v.id)
                break
        print("[TTS] Voz lista.")
    except Exception as e:
        print(f"[TTS] Error: {e}"); _tts_engine = None

def speak(text):
    if not text: return
    print(f"[ASISTENTE] {text}")
    if _tts_engine is None: return
    def _run():
        with _tts_lock:
            try: _tts_engine.say(text); _tts_engine.runAndWait()
            except: pass
    threading.Thread(target=_run, daemon=True).start()

# ── Joints ────────────────────────────────────────────────────────────────────
def get_joints(model, data):
    a = np.zeros(6)
    for i,n in enumerate(JOINT_NAMES):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        a[i] = data.qpos[model.jnt_qposadr[jid]]
    return a

def set_joints(model, data, angles):
    for i,n in enumerate(JOINT_NAMES):
        jid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        qadr = model.jnt_qposadr[jid]
        data.qpos[qadr] = float(np.clip(angles[i], LIMITS[i,0], LIMITS[i,1]))
    data.ctrl[:] = get_joints(model, data)

def blend(model, data, target, alpha=0.12):
    cur = get_joints(model, data)
    set_joints(model, data, np.clip(cur+alpha*(target-cur), LIMITS[:,0], LIMITS[:,1]))

# ── Freejoint ─────────────────────────────────────────────────────────────────
def _qadr(model, name):
    return int(model.jnt_qposadr[int(model.body(name).jntadr[0])])

def teleport(model, data, name, x, y, z):
    q = _qadr(model, name)
    data.qpos[q:q+7] = [x, y, z, 1, 0, 0, 0]

def get_xy(model, data, name):
    q = _qadr(model, name)
    return data.qpos[q:q+2].copy()

def move_book(model, data, dxy):
    xy = get_xy(model, data, "book_red")
    nx = float(np.clip(xy[0]+dxy[0], 0.10, 0.85))
    ny = float(np.clip(xy[1]+dxy[1], -0.42, 0.42))
    teleport(model, data, "book_red",  nx, ny, BOOK_Z)
    teleport(model, data, "book_open", nx, ny, TABLE_SURFACE+0.005)

# ── Libro visual ──────────────────────────────────────────────────────────────
CLOSED_G = {"br_cover_top":[0.72,0.08,0.08,1],"br_cover_bot":[0.72,0.08,0.08,1],
            "br_spine":[0.48,0.05,0.05,1],"br_pages":[0.95,0.93,0.87,1]}
OPEN_G   = {"bo_left":[0.99,0.98,0.95,1],"bo_right":[0.99,0.98,0.95,1],
            "bo_spine":[0.50,0.06,0.06,1],"bo_cvr_l":[0.72,0.08,0.08,1],
            "bo_cvr_r":[0.72,0.08,0.08,1]}

def show_book(model, is_open):
    for n,rgba in CLOSED_G.items():
        c=np.array(rgba,np.float32); c[3]=0.0 if is_open else 1.0
        model.geom_rgba[model.geom(n).id]=c
    for n,rgba in OPEN_G.items():
        c=np.array(rgba,np.float32); c[3]=1.0 if is_open else 0.0
        model.geom_rgba[model.geom(n).id]=c

# ── Luz ───────────────────────────────────────────────────────────────────────
def set_light(model, on):
    lid = model.light("flashlight").id
    if on:
        model.light_diffuse[lid] = np.array([3.0,2.6,2.0],np.float32)
        model.light_ambient[lid] = np.array([0.20,0.16,0.10],np.float32)
    else:
        model.light_diffuse[lid] = np.zeros(3,np.float32)
        model.light_ambient[lid] = np.zeros(3,np.float32)
    lens = model.geom("lantern_lens").id
    model.geom_rgba[lens] = (np.array([1.0,0.96,0.78,1.0]) if on
                             else np.array([0.14,0.14,0.11,1.0])).astype(np.float32)

# ── IK ────────────────────────────────────────────────────────────────────────
def compute_ik(model, data):
    base = data.xpos[model.body("base").id].copy()
    sh   = base.copy(); sh[2] += BASE_HEIGHT
    bpos = data.xpos[model.body("book_open").id].copy()
    bpos[2] = TABLE_SURFACE+0.012
    dx,dy = bpos[0]-base[0], bpos[1]-base[1]
    dz    = bpos[2]-sh[2]
    horiz = max(np.hypot(dx,dy), 0.01)
    pan   = float(np.clip(np.arctan2(-dy,dx), LIMITS[0,0], LIMITS[0,1]))
    dist  = max(np.hypot(horiz,dz), 1e-4)
    wh = horiz-TOOL*(horiz/dist); wv = dz-TOOL*(dz/dist)
    d  = np.clip(np.hypot(wh,wv), abs(ARM1-ARM2)+0.005, ARM1+ARM2-0.005)
    cos_e = (ARM1**2+ARM2**2-d**2)/(2*ARM1*ARM2)
    elbow = -(np.pi-np.arccos(np.clip(cos_e,-1,1)))
    atw   = np.arctan2(wv,wh)
    beta  = np.arccos(np.clip((ARM1**2+d**2-ARM2**2)/(2*ARM1*d),-1,1))
    lift  = atw+beta
    wrist = float(np.clip((np.arctan2(dz,horiz)+np.pi/2)-lift-elbow,
                          LIMITS[3,0],LIMITS[3,1]))
    return np.clip(np.array([pan,lift,elbow,wrist,0.0,0.35]),
                   LIMITS[:,0],LIMITS[:,1])

# ── Detección libro abierto — páginas blancas vistas desde arriba ───────────────
def detect_open_book(rgb):
    """
    Detecta las páginas blancas/claras del libro abierto desde la camara cenital.
    Filtra por brillo alto (V>175) y saturacion baja (S<70) = blanco puro.
    Valida aspect ratio rectangular de libro.
    Retorna (cx, cy, mask, area).
    """
    hsv  = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # Paginas blancas: muy brillante, poco saturado
    mask = cv2.inRange(hsv, np.array([0, 0, 175]), np.array([180, 70, 255]))
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None, mask, 0
    # Filtrar por area y aspect ratio (libro es rectangular)
    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 800:
            continue
        x, y, w, h = cv2.boundingRect(c)
        asp = w / max(h, 1)
        if 0.35 <= asp <= 3.0:
            candidates.append((area, c, x, y, w, h))
    if not candidates:
        return None, None, mask, 0
    candidates.sort(key=lambda t: t[0], reverse=True)
    area, best, x, y, w, h = candidates[0]
    # Centroide preciso con momentos
    M = cv2.moments(best)
    if M["m00"] == 0:
        return None, None, mask, area
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy, mask, area


# ── Vision tracking (logica del codigo de referencia, proporcional con deadband) ──
def vision_track(model, data, renderer, cam_id, debug=False):
    """
    Tracking proporcional basado en error de imagen.
    - Pan:   error horizontal → shoulder_pan
    - Pitch: error vertical   → wrist_flex
    Igual que el codigo de referencia pero para el marco verde del libro abierto.
    """
    global last_seen_time, timed_out

    renderer.update_scene(data, camera=cam_id)
    rgb = renderer.render()
    cx, cy, mask, area = detect_open_book(rgb)
    H, W = CAM_H, CAM_W

    if cx is None:
        elapsed = time.time() - last_seen_time
        if last_seen_time > 0 and elapsed > TIMEOUT_S:
            timed_out = True
        # Sin deteccion: volver suavemente a REST (igual que referencia)
        cur = get_joints(model, data)
        set_joints(model, data, cur + 0.01*(REST - cur))
        if debug:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if timed_out:
                cv2.putText(bgr,"TIMEOUT — luz OFF",(10,30),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
            elif last_seen_time > 0:
                remaining = max(0, TIMEOUT_S-(time.time()-last_seen_time))
                cv2.putText(bgr,f"Sin libro — {remaining:.1f}s",(10,30),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,140,255),2)
            else:
                cv2.putText(bgr,"Buscando marco verde...",(10,30),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(180,180,180),2)
            cv2.imshow("Vista Camara SO101",bgr)
            cv2.imshow("Mascara Verde",mask)
            cv2.waitKey(1)
        return False

    last_seen_time = time.time(); timed_out = False

    # Error normalizado -1..1
    ex = (cx - W/2) / (W/2)
    ey = (cy - H/2) / (H/2)

    DEADBAND_X   = 0.10
    DEADBAND_Y   = 0.10
    MIN_AREA_TRK = 500    # area minima para activar tracking
    YAW_SIGN     = -1.0
    YAW_RANGE    = 0.8
    PITCH_SIGN   = -1.0
    PITCH_RANGE  = 0.45
    BLEND        = 0.04   # suavizado proporcional (igual que referencia)

    if area > MIN_AREA_TRK:
        # Pan proporcional
        if abs(ex) > DEADBAND_X:
            desired_pan = data.ctrl[0] + YAW_SIGN * YAW_RANGE * ex
            data.ctrl[0] += BLEND * (desired_pan - data.ctrl[0])
        else:
            data.ctrl[0] += BLEND * (REST[0] - data.ctrl[0])

        # Pitch proporcional
        if abs(ey) > DEADBAND_Y:
            desired_pitch = data.ctrl[3] + PITCH_SIGN * PITCH_RANGE * ey
            data.ctrl[3] += BLEND * (desired_pitch - data.ctrl[3])
        else:
            data.ctrl[3] += BLEND * (REST[3] - data.ctrl[3])

    # Clamp joints
    data.ctrl[0] = float(np.clip(data.ctrl[0], LIMITS[0,0], LIMITS[0,1]))
    data.ctrl[3] = float(np.clip(data.ctrl[3], LIMITS[3,0], LIMITS[3,1]))

    # Sincronizar qpos con ctrl para que set_joints vea cambios
    for i,n in enumerate(JOINT_NAMES):
        jid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        qadr = model.jnt_qposadr[jid]
        data.qpos[qadr] = data.ctrl[i]

    if debug:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.circle(bgr,(cx,cy),10,(0,255,0),-1)
        cv2.line(bgr,(W//2,0),(W//2,H),(255,255,255),1)
        cv2.line(bgr,(0,H//2),(W,H//2),(255,255,255),1)
        cv2.putText(bgr,f"ex={ex:.2f} ey={ey:.2f} area={area:.0f}",(10,25),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        cv2.putText(bgr,"MARCO VERDE DETECTADO",(10,H-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
        cv2.imshow("Vista Camara SO101",bgr)
        cv2.imshow("Mascara Verde",mask)
        cv2.waitKey(1)
    return True

# ── Reset ─────────────────────────────────────────────────────────────────────
def do_reset(model, data):
    global timed_out, last_seen_time
    teleport(model,data,"book_red", BOOK_START[0],BOOK_START[1],BOOK_Z)
    teleport(model,data,"book_open",BOOK_START[0],BOOK_START[1],TABLE_SURFACE+0.005)
    show_book(model,False); set_light(model,False); set_joints(model,data,REST)
    timed_out=False; last_seen_time=0.0
    with lock: state["book_open"]=False; state["lamp_on"]=False
    mujoco.mj_forward(model,data)

# ── Clima ─────────────────────────────────────────────────────────────────────
def get_weather(city=None):
    city    = city or os.getenv("DEFAULT_CITY","Monterrey,MX")
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "No tengo configurada la API del clima."
    try:
        r = requests.get("https://api.openweathermap.org/data/2.5/weather",
                         params={"q":city,"appid":api_key,"units":"metric","lang":"es"},
                         timeout=8)
        r.raise_for_status(); d = r.json()
        desc = d["weather"][0]["description"]
        temp = round(d["main"]["temp"])
        feel = round(d["main"]["feels_like"])
        hum  = d["main"]["humidity"]
        return (f"En {city}, está {desc}, con {temp} grados, "
                f"sensación de {feel} y humedad de {hum} por ciento.")
    except requests.exceptions.HTTPError:
        return "No pude consultar el clima. Revisa tu API key."
    except Exception as e:
        return f"Error consultando clima: {e}"

# ── Gemini Voice Controller ───────────────────────────────────────────────────
class GeminiVoice:
    def __init__(self):
        self.enabled        = False
        self.gemini_enabled = False
        self.client         = None

        if not SR_OK:
            print("[VOZ] SpeechRecognition no instalado."); return
        try:
            self.rec = sr.Recognizer()
            self.mic = sr.Microphone()
            with self.mic as src:
                print("[VOZ] Calibrando mic...")
                self.rec.adjust_for_ambient_noise(src, duration=1)
            self.enabled = True
            print("[VOZ] Microfono listo. Presiona V para hablar.")
        except Exception as e:
            print(f"[VOZ] Error mic: {e}"); return

        if not GENAI_OK:
            print("[VOZ] google-genai no instalado. Solo comandos locales."); return
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("[VOZ] Sin GEMINI_API_KEY. Solo comandos locales."); return
        try:
            self.client         = genai.Client(api_key=api_key)
            self.gemini_enabled = True
            print("[VOZ] Gemini listo para preguntas generales.")
        except Exception as e:
            print(f"[VOZ] Gemini error: {e}")

    def listen(self):
        if not self.enabled: return None
        try:
            with self.mic as src:
                print("[VOZ] Escuchando...")
                audio = self.rec.listen(src, timeout=5, phrase_time_limit=5)
            text = self.rec.recognize_google(audio, language="es-MX")
            print(f"[VOZ] '{text}'"); return text
        except: return None

    @staticmethod
    def _norm(text):
        """Quita acentos y pasa a minusculas."""
        for a,b in [("á","a"),("é","e"),("í","i"),("ó","o"),("ú","u"),("ü","u"),("ñ","n")]:
            text = text.replace(a,b)
        return text.lower().strip()

    def _local_intent(self, text):
        """Detecta intents sin gastar cuota Gemini."""
        if not text: return None
        t = self._norm(text)

        # Clima
        if any(w in t for w in ["clima","temperatura","lluvia","llueve",
                                  "nublado","calor","frio","pronostico","tiempo"]):
            city = ""
            for marker in ["clima en ","temperatura en ","tiempo en ","en "]:
                if marker in t:
                    city = t.split(marker,1)[1].strip()
                    for sw in ["por favor","hoy","ahorita","actualmente"]:
                        city = city.replace(sw,"").strip()
                    break
            return {"type":"weather","value":"NONE","reply":"","city":city}

        # Lamp ON
        if any(w in t for w in ["enciende","encender","prende","prender",
                                  "ilumina","activa la luz","enciende la lampara"]):
            return {"type":"action","value":"TURN_ON_LAMP",
                    "reply":"Claro, encendi la lampara.","city":""}
        # Lamp OFF
        if any(w in t for w in ["apaga","apagar","oscurece","desactiva la luz"]):
            return {"type":"action","value":"TURN_OFF_LAMP",
                    "reply":"Listo, apague la lampara.","city":""}
        # Book
        if "abre" in t and "libro" in t:
            return {"type":"action","value":"OPEN_BOOK","reply":"Abri el libro.","city":""}
        if "cierra" in t and "libro" in t:
            return {"type":"action","value":"CLOSE_BOOK","reply":"Cerre el libro.","city":""}
        # Reset
        if any(w in t for w in ["reinicia","resetea","reset","vuelve al inicio","restablece"]):
            return {"type":"action","value":"RESET","reply":"Reinicie la escena.","city":""}
        # Pausa / reanudar
        if any(w in t for w in ["pausa","pausar","deten","detener"]):
            return {"type":"action","value":"PAUSE","reply":"Pause la simulacion.","city":""}
        if any(w in t for w in ["continua","reanuda","sigue","prosigue"]):
            return {"type":"action","value":"RESUME","reply":"Continue la simulacion.","city":""}

        return None  # no reconocido localmente → pasar a Gemini

    def _gemini_intent(self, text):
        if not self.gemini_enabled or not self.client:
            return {"type":"question","value":"NONE",
                    "reply":"Por ahora solo puedo responder comandos locales y clima.","city":""}
        prompt = f"""Eres un asistente de voz para una simulacion de lampara inteligente.
Responde SOLO con JSON valido, sin markdown.

Tipos: action | weather | question | none
Acciones validas: TURN_ON_LAMP TURN_OFF_LAMP OPEN_BOOK CLOSE_BOOK RESET PAUSE RESUME NONE

Formato:
{{"type":"...","value":"...","reply":"respuesta corta en espanol","city":"ciudad o vacio"}}

Reglas:
- prender/encender/iluminar → TURN_ON_LAMP
- apagar/oscurecer → TURN_OFF_LAMP
- abrir libro → OPEN_BOOK
- cerrar libro → CLOSE_BOOK
- reiniciar/reset → RESET
- pausar → PAUSE
- continuar/reanudar → RESUME
- pregunta de clima → weather (pon ciudad en city si la menciona)
- pregunta general → question con respuesta breve
- reply: max una oracion, sin markdown

Mensaje: "{text}"
"""
        try:
            r   = self.client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            raw = r.text.strip().replace("```json","").replace("```","").strip()
            print(f"[VOZ] Gemini raw: {raw}")
            intent = json.loads(raw)
            valid_actions = {"TURN_ON_LAMP","TURN_OFF_LAMP","OPEN_BOOK","CLOSE_BOOK",
                             "RESET","PAUSE","RESUME","NONE"}
            if intent.get("type") not in {"action","weather","question","none"}:
                intent["type"] = "none"
            if intent.get("value") not in valid_actions:
                intent["value"] = "NONE"
            return intent
        except Exception as e:
            err = str(e)
            print(f"[VOZ] Gemini error: {err}")
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                return {"type":"none","value":"NONE",
                        "reply":"Llegue al limite de Gemini, pero puedo controlar la lampara y el clima.",
                        "city":""}
            return {"type":"none","value":"NONE",
                    "reply":"Tuve un problema al procesar el comando.","city":""}

    def get_intent(self):
        text   = self.listen()
        intent = self._local_intent(text)
        if intent:
            print(f"[VOZ] Intent local: {intent}")
            return intent
        return self._gemini_intent(text)

# ── Aplicar intent ────────────────────────────────────────────────────────────
def apply_intent(intent):
    t      = intent.get("type","none")
    value  = intent.get("value","NONE")
    reply  = intent.get("reply","")
    city   = intent.get("city","")
    print(f"[INTENT] type={t} value={value} reply={reply}")

    if t == "action":
        with lock:
            if value=="TURN_ON_LAMP":
                state["lamp_on"]=True; state["book_open"]=True; state["toggled"]=True
                speak("Claro, encendi la lampara.")
            elif value=="TURN_OFF_LAMP":
                state["lamp_on"]=False; state["book_open"]=False; state["toggled"]=True
                speak("Listo, apague la lampara.")
            elif value=="OPEN_BOOK":
                state["book_open"]=True; state["toggled"]=True
                speak("Abri el libro.")
            elif value=="CLOSE_BOOK":
                state["book_open"]=False; state["lamp_on"]=False; state["toggled"]=True
                speak("Cerre el libro.")
            elif value=="RESET":
                state["reset"]=True; speak("Reinicie la escena.")
            elif value=="PAUSE":
                state["paused"]=True; speak("Pause la simulacion.")
            elif value=="RESUME":
                state["paused"]=False; speak("Continue la simulacion.")
            else:
                speak(reply or "No entendi que accion hacer.")

    elif t == "weather":
        def _w():
            w = get_weather(city if city else None)
            speak(w)
        threading.Thread(target=_w, daemon=True).start()

    elif t == "question":
        speak(reply or "No tengo una respuesta clara.")

    else:
        speak(reply or "No entendi bien el comando.")

# ── Teclado ───────────────────────────────────────────────────────────────────
def keyboard_thread():
    print("\n[SO101 Linterna v10]")
    print("  WASD=libro  O=abrir  L=luz  F=debug  V=voz  R=reset  P=pausa  Esc=salir\n")
    while True:
        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            with lock:
                if   ch=='\x1b':        state["quit"]=True; break
                elif ch in('p','P'):
                    state["paused"]=not state["paused"]
                    speak("Pausa." if state["paused"] else "Continuando.")
                elif ch in('r','R'):    state["reset"]=True; speak("Reiniciando.")
                elif ch in('v','V'):    state["voice_req"]=True
                elif ch in('f','F'):
                    state["cam_debug"]=not state["cam_debug"]
                    print(f"  [DEBUG {'ON' if state['cam_debug'] else 'OFF'}]")
                elif ch in('l','L'):
                    state["lamp_on"]=not state["lamp_on"]
                    if state["lamp_on"]: state["book_open"]=True; state["toggled"]=True
                    else:                state["book_open"]=False; state["toggled"]=True
                    speak("Lampara encendida." if state["lamp_on"] else "Lampara apagada.")
                elif ch in('o','O'):
                    state["book_open"]=not state["book_open"]; state["toggled"]=True
                    if not state["book_open"]: state["lamp_on"]=False
                    speak("Libro abierto." if state["book_open"] else "Libro cerrado.")
                elif ch in('w','W'): state["move"]+=[MOVE_SPEED,0]
                elif ch in('s','S'): state["move"]+=[-MOVE_SPEED,0]
                elif ch in('a','A'): state["move"]+=[0,MOVE_SPEED]
                elif ch in('d','D'): state["move"]+=[0,-MOVE_SPEED]
        time.sleep(0.008)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global timed_out, last_seen_time
    timed_out=False; last_seen_time=0.0

    init_tts()
    print(f"Cargando {XML_PATH} ...")
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    show_book(model,False); set_light(model,False)
    teleport(model,data,"book_red", BOOK_START[0],BOOK_START[1],BOOK_Z)
    teleport(model,data,"book_open",BOOK_START[0],BOOK_START[1],TABLE_SURFACE+0.005)
    set_joints(model,data,REST)
    mujoco.mj_forward(model,data)

    cam_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAM_NAME)
    renderer = mujoco.Renderer(model, height=CAM_H, width=CAM_W)
    print(f"[CAMARA] '{CAM_NAME}' lista.")

    voice = GeminiVoice()
    kb    = threading.Thread(target=keyboard_thread, daemon=True)
    kb.start()

    speak("Sistema listo.")
    step = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.35,0.0,0.90]
        viewer.cam.distance  = 1.45
        viewer.cam.elevation = -20
        viewer.cam.azimuth   = 150
        print("Viewer abierto. Click en TERMINAL para teclas.\n")

        while viewer.is_running():
            t0 = time.perf_counter()

            with lock:
                if state["quit"]: break
                paused    = state["paused"]
                rst       = state["reset"]
                dxy       = state["move"].copy()
                bopen     = state["book_open"]
                toggled   = state["toggled"]
                lamp      = state["lamp_on"]
                voice_req = state["voice_req"]
                debug     = state["cam_debug"]
                state["move"]=np.zeros(2); state["reset"]=False
                state["toggled"]=False;    state["voice_req"]=False

            if voice_req:
                def _vi():
                    speak("Te escucho.")
                    apply_intent(voice.get_intent())
                threading.Thread(target=_vi, daemon=True).start()

            if rst: do_reset(model,data)

            if not paused:
                if np.any(dxy!=0): move_book(model,data,dxy)

                if toggled:
                    xy = get_xy(model,data,"book_red")
                    teleport(model,data,"book_open",xy[0],xy[1],TABLE_SURFACE+0.005)
                    show_book(model,bopen)
                    set_light(model, lamp if lamp else bopen)
                    mujoco.mj_forward(model,data)

                if bopen:
                    blend(model,data,compute_ik(model,data),alpha=0.14)
                    if step%10==0:
                        detected = vision_track(model,data,renderer,cam_id,debug=debug)
                        if timed_out:
                            with lock:
                                if state["book_open"] or state["lamp_on"]:
                                    state["book_open"]=False; state["lamp_on"]=False
                                    state["toggled"]=True; bopen=False; toggled=True
                                    timed_out=False
                                    speak("No veo el libro. Apague la luz.")
                                    print("[CAM] Timeout — luz OFF")
                else:
                    blend(model,data,REST,alpha=0.06)
                    if debug and step%10==0:
                        vision_track(model,data,renderer,cam_id,debug=True)

                mujoco.mj_step(model,data)
                step+=1

            viewer.sync()
            elapsed = time.perf_counter()-t0
            if elapsed < DT: time.sleep(DT-elapsed)

    speak("Hasta luego.")
    renderer.close()
    cv2.destroyAllWindows()
    print("Hasta luego!")

if __name__=="__main__":
    main()