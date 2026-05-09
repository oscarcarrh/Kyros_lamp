"""
SO101 Smart Lantern Tracker
===========================

Features
--------
- Only RED book
- Book open/close with O
- Lamp ON/OFF with L
- Voice command with V + Gemini
- Assistant voice response with pyttsx3
- Short answers to questions
- Example weather response
- Dark scene
- Robot tracks ONLY in YAW
- Spotlight attached to gripper
- No arm stretching

Controls
--------
WASD  -> move book
Q/E   -> move book up/down
O     -> open/close book
L     -> lamp on/off
V     -> voice command with Gemini
R     -> reset
P     -> pause
ESC   -> quit
"""

import os
import sys
import time
import json
import select
import termios
import tty
import threading
import requests

from dotenv import load_dotenv

import mujoco
import mujoco.viewer
import numpy as np

try:
    import speech_recognition as sr
except ImportError:
    sr = None

try:
    from google import genai
except ImportError:
    genai = None

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None


# ─────────────────────────────────────────────
# LOAD ENV
# ─────────────────────────────────────────────

load_dotenv()


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

XML_PATH = "my_smart_lantern_scene.xml"

DT = 0.002
MOVE_SPEED = 0.003
PAN_GAIN = 4.0

TABLE_Z = 0.757
BOOK_Z = TABLE_Z + 0.015

BOOK_START = np.array([0.35, -0.18, BOOK_Z])

GEMINI_MODEL = "gemini-2.5-flash"


# ─────────────────────────────────────────────
# SHARED STATE
# ─────────────────────────────────────────────

state = {
    "move": np.zeros(3),
    "paused": False,
    "quit": False,
    "reset": False,
    "book_open": False,
    "lamp_on": False,
    "voice_request": False,
}

state_lock = threading.Lock()


# ─────────────────────────────────────────────
# TEXT TO SPEECH
# ─────────────────────────────────────────────

tts_engine = None
tts_lock = threading.Lock()


def init_tts():
    global tts_engine

    if pyttsx3 is None:
        print("[TTS] pyttsx3 no está instalado.")
        return

    try:
        tts_engine = pyttsx3.init()

        # Velocidad de habla
        tts_engine.setProperty("rate", 175)

        # Volumen
        tts_engine.setProperty("volume", 1.0)

        print("[TTS] Voz lista.")

    except Exception as e:
        print(f"[TTS] No se pudo inicializar voz: {e}")
        tts_engine = None


def speak(text):
    if not text:
        return

    print(f"[ASSISTANT] {text}")

    if tts_engine is None:
        return

    def _speak():
        with tts_lock:
            try:
                tts_engine.say(text)
                tts_engine.runAndWait()
            except Exception as e:
                print(f"[TTS] Error hablando: {e}")

    threading.Thread(target=_speak, daemon=True).start()


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def get_book_pos(model, data):
    return data.xpos[model.body("book_red").id].copy()


def set_book_pose(model, data, pos):
    jid = int(model.body("book_red").jntadr[0])
    qadr = int(model.jnt_qposadr[jid])

    data.qpos[qadr:qadr + 3] = pos

    data.qpos[qadr + 2] = max(data.qpos[qadr + 2], TABLE_Z + 0.013)

    data.qpos[qadr] = np.clip(data.qpos[qadr], -0.40, 0.80)
    data.qpos[qadr + 1] = np.clip(data.qpos[qadr + 1], -0.45, 0.45)

    mujoco.mj_forward(model, data)


def move_book(model, data, delta):
    pos = get_book_pos(model, data)
    pos += delta
    set_book_pose(model, data, pos)


def reset_scene(model, data):
    set_book_pose(model, data, BOOK_START)

    data.ctrl[:] = 0.0

    close_book(model, data)
    set_flashlight(model, False)

    with state_lock:
        state["book_open"] = False
        state["lamp_on"] = False

    mujoco.mj_forward(model, data)


# ─────────────────────────────────────────────
# BOOK OPEN / CLOSE
# ─────────────────────────────────────────────

def open_book(model, data):
    top_id = model.geom("br_cover_top").id
    bot_id = model.geom("br_cover_bot").id

    model.geom_pos[top_id][2] = 0.030
    model.geom_pos[bot_id][2] = -0.030

    mujoco.mj_forward(model, data)


def close_book(model, data):
    top_id = model.geom("br_cover_top").id
    bot_id = model.geom("br_cover_bot").id

    model.geom_pos[top_id][2] = 0.0125
    model.geom_pos[bot_id][2] = -0.0125

    mujoco.mj_forward(model, data)


# ─────────────────────────────────────────────
# FLASHLIGHT
# ─────────────────────────────────────────────

def set_flashlight(model, enabled):
    light_id = model.light("flashlight").id

    if enabled:
        model.light_diffuse[light_id] = np.array([1.0, 1.0, 0.9])
        model.light_ambient[light_id] = np.array([0.15, 0.15, 0.12])
        model.light_specular[light_id] = np.array([0.3, 0.3, 0.3])
    else:
        model.light_diffuse[light_id] = np.array([0.0, 0.0, 0.0])
        model.light_ambient[light_id] = np.array([0.0, 0.0, 0.0])
        model.light_specular[light_id] = np.array([0.0, 0.0, 0.0])


# ─────────────────────────────────────────────
# YAW ONLY TRACKING
# ─────────────────────────────────────────────

def track_book_yaw_only(model, data):
    base_pos = data.xpos[model.body("base").id].copy()
    target = get_book_pos(model, data)

    dx = target[0] - base_pos[0]
    dy = target[1] - base_pos[1]

    yaw = np.arctan2(-dy, dx)
    yaw = np.clip(yaw, -1.91986, 1.91986)

    desired = np.array([
        yaw,
        0.0,
        0.0,
        0.0,
        0.0,
        0.3
    ])

    data.ctrl[:] += PAN_GAIN * DT * (desired - data.ctrl[:])


# ─────────────────────────────────────────────
# WEATHER EXAMPLE
# ─────────────────────────────────────────────

def get_weather_summary(city=None):
    city = city or os.getenv("DEFAULT_CITY", "Monterrey,MX")
    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        return "No tengo configurada la API del clima."

    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": city,
        "appid": api_key,
        "units": "metric",
        "lang": "es",
    }

    try:
        response = requests.get(url, params=params, timeout=8)
        response.raise_for_status()
        data = response.json()

        description = data["weather"][0]["description"]
        temp = round(data["main"]["temp"])
        feels_like = round(data["main"]["feels_like"])
        humidity = data["main"]["humidity"]

        return (
            f"En {city}, está {description}, "
            f"con {temp} grados, sensación de {feels_like} "
            f"y humedad de {humidity} por ciento."
        )

    except requests.exceptions.HTTPError as e:
        print(f"[WEATHER] HTTP error: {e}")
        return "No pude consultar el clima. Revisa tu API key o la ciudad configurada."

    except requests.exceptions.RequestException as e:
        print(f"[WEATHER] Request error: {e}")
        return "No pude conectarme al servicio del clima."

    except KeyError as e:
        print(f"[WEATHER] Respuesta inesperada: falta {e}")
        return "Recibí una respuesta inesperada del clima."

    except Exception as e:
        print(f"[WEATHER] Error inesperado: {e}")
        return "No pude consultar el clima en este momento."


# ─────────────────────────────────────────────
# GEMINI VOICE CONTROLLER
# ─────────────────────────────────────────────

class GeminiVoiceController:
    def __init__(self):
        self.enabled = False
        self.gemini_enabled = False
        self.client = None

        if sr is None:
            print("[VOICE] SpeechRecognition no está instalado.")
            return

        try:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()

            with self.microphone as source:
                print("[VOICE] Ajustando ruido ambiente...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)

            self.enabled = True
            print("[VOICE] Micrófono listo. Presiona V para hablar.")

        except Exception as e:
            print(f"[VOICE] No se pudo inicializar micrófono: {e}")
            return

        # Gemini es opcional.
        # Si no hay key o se acaba la cuota, todavía funcionan comandos locales y clima.
        if genai is None:
            print("[VOICE] google-genai no está instalado. Usaré solo comandos locales.")
            return

        if not os.getenv("GEMINI_API_KEY"):
            print("[VOICE] No existe GEMINI_API_KEY en .env. Usaré solo comandos locales.")
            return

        try:
            self.client = genai.Client()
            self.gemini_enabled = True
            print("[VOICE] Gemini listo para preguntas generales.")

        except Exception as e:
            print(f"[VOICE] No se pudo inicializar Gemini: {e}")
            self.gemini_enabled = False

    def listen_text(self):
        if not self.enabled:
            return None

        try:
            with self.microphone as source:
                print("\n[VOICE] Escuchando...")
                audio = self.recognizer.listen(
                    source,
                    timeout=5,
                    phrase_time_limit=5
                )

            text = self.recognizer.recognize_google(audio, language="es-MX")
            print(f"[VOICE] Texto detectado: {text}")
            return text

        except sr.WaitTimeoutError:
            print("[VOICE] No escuché ningún comando.")
            return None

        except sr.UnknownValueError:
            print("[VOICE] No entendí el audio.")
            return None

        except Exception as e:
            print(f"[VOICE] Error escuchando voz: {e}")
            return None

    def normalize_text(self, text):
        replacements = {
            "á": "a",
            "é": "e",
            "í": "i",
            "ó": "o",
            "ú": "u",
            "ü": "u",
            "ñ": "n",
        }

        text = text.lower().strip()

        for original, replacement in replacements.items():
            text = text.replace(original, replacement)

        return text

    def extract_city_from_weather_text(self, text_lower):
        """
        Extrae ciudad de frases como:
        - como esta el clima en monterrey
        - clima en cancun
        - temperatura en ciudad de mexico
        """

        markers = [
            "clima en ",
            "temperatura en ",
            "tiempo en ",
            "lluvia en ",
            "llueve en ",
            "en ",
        ]

        for marker in markers:
            if marker in text_lower:
                city = text_lower.split(marker, 1)[1].strip()

                # Limpieza simple de palabras sobrantes
                stop_words = [
                    "por favor",
                    "hoy",
                    "ahorita",
                    "actualmente",
                    "en este momento",
                ]

                for stop_word in stop_words:
                    city = city.replace(stop_word, "").strip()

                if city:
                    return city

        return ""

    def local_text_to_intent(self, text):
        """
        Detecta comandos frecuentes sin usar Gemini.
        Esto evita gastar cuota y permite que clima funcione aunque Gemini dé 429.
        """

        if not text:
            return {
                "type": "none",
                "value": "NONE",
                "reply": "No escuché ningún comando.",
                "city": ""
            }

        text_lower = self.normalize_text(text)

        # Ignorar si por accidente escucha la frase del asistente
        if text_lower in ["te escucho", "sistema listo", "hasta luego"]:
            return {
                "type": "none",
                "value": "NONE",
                "reply": "Dime el comando después de presionar la tecla.",
                "city": ""
            }

        # Clima: primero, para que no gaste Gemini
        weather_words = [
            "clima",
            "temperatura",
            "lluvia",
            "llueve",
            "nublado",
            "calor",
            "frio",
            "pronostico",
            "tiempo"
        ]

        if any(word in text_lower for word in weather_words):
            city = self.extract_city_from_weather_text(text_lower)

            return {
                "type": "weather",
                "value": "NONE",
                "reply": "",
                "city": city
            }

        # Lámpara
        turn_on_words = [
            "enciende",
            "encender",
            "prende",
            "prender",
            "ilumina",
            "iluminar",
            "activa la luz",
            "activar la luz",
            "enciende la lampara",
            "prende la lampara"
        ]

        if any(word in text_lower for word in turn_on_words):
            return {
                "type": "action",
                "value": "TURN_ON_LAMP",
                "reply": "Claro, encendí la lámpara.",
                "city": ""
            }

        turn_off_words = [
            "apaga",
            "apagar",
            "oscurece",
            "oscurecer",
            "desactiva la luz",
            "desactivar la luz",
            "apaga la lampara"
        ]

        if any(word in text_lower for word in turn_off_words):
            return {
                "type": "action",
                "value": "TURN_OFF_LAMP",
                "reply": "Listo, apagué la lámpara.",
                "city": ""
            }

        # Libro
        if "abre" in text_lower and "libro" in text_lower:
            return {
                "type": "action",
                "value": "OPEN_BOOK",
                "reply": "Abrí el libro.",
                "city": ""
            }

        if "cierra" in text_lower and "libro" in text_lower:
            return {
                "type": "action",
                "value": "CLOSE_BOOK",
                "reply": "Cerré el libro.",
                "city": ""
            }

        # Reset
        reset_words = [
            "reinicia",
            "reiniciar",
            "resetea",
            "reset",
            "vuelve al inicio",
            "restablece"
        ]

        if any(word in text_lower for word in reset_words):
            return {
                "type": "action",
                "value": "RESET",
                "reply": "Reinicié la escena.",
                "city": ""
            }

        # Pausa
        pause_words = [
            "pausa",
            "pausar",
            "deten",
            "detener",
            "para la simulacion"
        ]

        if any(word in text_lower for word in pause_words):
            return {
                "type": "action",
                "value": "PAUSE",
                "reply": "Pausé la simulación.",
                "city": ""
            }

        resume_words = [
            "continua",
            "continuar",
            "reanuda",
            "reanudar",
            "sigue",
            "prosigue"
        ]

        if any(word in text_lower for word in resume_words):
            return {
                "type": "action",
                "value": "RESUME",
                "reply": "Continué la simulación.",
                "city": ""
            }

        return None

    def text_to_intent(self, text):
        if not text:
            return {
                "type": "none",
                "value": "NONE",
                "reply": "No escuché ningún comando.",
                "city": ""
            }

        if not self.gemini_enabled or self.client is None:
            return {
                "type": "question",
                "value": "NONE",
                "reply": "Por ahora solo puedo responder comandos locales y clima.",
                "city": ""
            }

        prompt = f"""
Eres un asistente de voz para una simulación de una lámpara inteligente con un libro y un brazo robótico.

Tu tarea es interpretar el mensaje del usuario y responder SOLO con JSON válido.

Tipos disponibles:
- action: cuando el usuario quiere controlar la simulación.
- weather: cuando el usuario pregunta por el clima.
- question: cuando el usuario hace una pregunta general.
- none: cuando no hay intención clara.

Acciones válidas:
- TURN_ON_LAMP
- TURN_OFF_LAMP
- OPEN_BOOK
- CLOSE_BOOK
- RESET
- PAUSE
- RESUME
- NONE

Formato obligatorio:
{{
  "type": "action | weather | question | none",
  "value": "ACCION_O_NONE",
  "reply": "respuesta corta en español",
  "city": "ciudad o vacío"
}}

Reglas:
- Si el usuario dice prender, encender, iluminar o activar la luz, usa:
  type=action, value=TURN_ON_LAMP.
- Si el usuario dice apagar, oscurecer o desactivar la luz, usa:
  type=action, value=TURN_OFF_LAMP.
- Si el usuario quiere abrir el libro, usa:
  type=action, value=OPEN_BOOK.
- Si el usuario quiere cerrar el libro, usa:
  type=action, value=CLOSE_BOOK.
- Si pide reiniciar, resetear o volver al inicio, usa:
  type=action, value=RESET.
- Si pide pausar, usa:
  type=action, value=PAUSE.
- Si pide continuar o reanudar, usa:
  type=action, value=RESUME.
- Si pregunta por el clima, usa:
  type=weather, value=NONE.
- Si pregunta por clima y menciona ciudad, coloca la ciudad en city.
- Si pregunta por clima sin ciudad, usa city="".
- Si es una pregunta general, responde de forma breve.
- La respuesta debe ser natural, corta y en español.
- Máximo una oración en reply.
- No uses Markdown.
- No expliques el JSON.

Mensaje del usuario:
{text}
"""

        try:
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )

            raw = response.text.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()

            print(f"[VOICE] Respuesta cruda Gemini: {raw}")

            try:
                intent = json.loads(raw)
            except json.JSONDecodeError:
                return {
                    "type": "question",
                    "value": "NONE",
                    "reply": raw[:180],
                    "city": ""
                }

            intent_type = intent.get("type", "none")
            value = intent.get("value", "NONE")
            reply = intent.get("reply", "")
            city = intent.get("city", "")

            valid_types = {"action", "weather", "question", "none"}
            valid_actions = {
                "TURN_ON_LAMP",
                "TURN_OFF_LAMP",
                "OPEN_BOOK",
                "CLOSE_BOOK",
                "RESET",
                "PAUSE",
                "RESUME",
                "NONE",
            }

            if intent_type not in valid_types:
                intent_type = "none"

            if value not in valid_actions:
                value = "NONE"

            return {
                "type": intent_type,
                "value": value,
                "reply": reply,
                "city": city
            }

        except Exception as e:
            print(f"[VOICE] Error con Gemini: {e}")

            error_text = str(e)

            if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
                return {
                    "type": "none",
                    "value": "NONE",
                    "reply": "Llegué al límite de Gemini, pero aún puedo controlar la lámpara y consultar el clima.",
                    "city": ""
                }

            return {
                "type": "none",
                "value": "NONE",
                "reply": "Tuve un problema al procesar el comando.",
                "city": ""
            }

    def get_intent(self):
        text = self.listen_text()

        local_intent = self.local_text_to_intent(text)

        if local_intent:
            print(f"[VOICE] Intención local: {local_intent}")
            return local_intent

        return self.text_to_intent(text)
# ─────────────────────────────────────────────
# APPLY INTENT
# ─────────────────────────────────────────────

def apply_action(action):
    with state_lock:
        if action == "TURN_ON_LAMP":
            state["lamp_on"] = True
            state["book_open"] = True
            print("[ACTION] Lámpara encendida")
            speak("Claro, encendí la lámpara.")

        elif action == "TURN_OFF_LAMP":
            state["lamp_on"] = False
            print("[ACTION] Lámpara apagada")
            speak("Listo, apagué la lámpara.")

        elif action == "OPEN_BOOK":
            state["book_open"] = True
            print("[ACTION] Libro abierto")
            speak("Abrí el libro.")

        elif action == "CLOSE_BOOK":
            state["book_open"] = False
            state["lamp_on"] = False
            print("[ACTION] Libro cerrado")
            speak("Cerré el libro.")

        elif action == "RESET":
            state["reset"] = True
            print("[ACTION] Reset")
            speak("Reinicié la escena.")

        elif action == "PAUSE":
            state["paused"] = True
            print("[ACTION] Pausa")
            speak("Pausé la simulación.")

        elif action == "RESUME":
            state["paused"] = False
            print("[ACTION] Reanudar")
            speak("Continué la simulación.")

        else:
            print("[ACTION] Sin acción")
            speak("No entendí qué acción hacer.")


def apply_intent(intent):
    intent_type = intent.get("type", "none")
    value = intent.get("value", "NONE")
    reply = intent.get("reply", "")

    print(f"[INTENT] type={intent_type}, value={value}, reply={reply}")

    if intent_type == "action":
        apply_action(value)

    elif intent_type == "weather":
        city = intent.get("city", "")
        weather = get_weather_summary(city if city else None)
        speak(weather)

    elif intent_type == "question":
        if reply:
            speak(reply)
        else:
            speak("No tengo una respuesta clara para eso.")

    else:
        if reply:
            speak(reply)
        else:
            speak("No entendí bien el comando.")


# ─────────────────────────────────────────────
# KEYBOARD THREAD MAC/LINUX
# ─────────────────────────────────────────────

def read_key_nonblocking():
    dr, _, _ = select.select([sys.stdin], [], [], 0)

    if dr:
        return sys.stdin.read(1)

    return None


def keyboard_thread():
    print("\nControls:")
    print("WASD -> move book")
    print("Q/E  -> up/down")
    print("O    -> open/close book")
    print("L    -> lamp on/off")
    print("V    -> voice command")
    print("R    -> reset")
    print("P    -> pause")
    print("ESC  -> quit\n")

    old_settings = termios.tcgetattr(sys.stdin)

    try:
        tty.setcbreak(sys.stdin.fileno())

        while True:
            ch = read_key_nonblocking()

            if ch is not None:
                with state_lock:
                    if ch == "\x1b":
                        state["quit"] = True
                        break

                    elif ch in ("p", "P"):
                        state["paused"] = not state["paused"]
                        print(f"[KEY] Paused: {state['paused']}")
                        speak("Pausa." if state["paused"] else "Continuando.")

                    elif ch in ("r", "R"):
                        state["reset"] = True
                        print("[KEY] Reset")
                        speak("Reiniciando escena.")

                    elif ch in ("o", "O"):
                        state["book_open"] = not state["book_open"]

                        if not state["book_open"]:
                            state["lamp_on"] = False

                        print(f"[KEY] Book open: {state['book_open']}")
                        speak("Libro abierto." if state["book_open"] else "Libro cerrado.")

                    elif ch in ("l", "L"):
                        state["lamp_on"] = not state["lamp_on"]

                        if state["lamp_on"]:
                            state["book_open"] = True

                        print(f"[KEY] Lamp on: {state['lamp_on']}")
                        speak("Lámpara encendida." if state["lamp_on"] else "Lámpara apagada.")

                    elif ch in ("v", "V"):
                        state["voice_request"] = True
                        print("[KEY] Voice request")

                    elif ch in ("w", "W"):
                        state["move"] += np.array([MOVE_SPEED, 0, 0])

                    elif ch in ("s", "S"):
                        state["move"] += np.array([-MOVE_SPEED, 0, 0])

                    elif ch in ("a", "A"):
                        state["move"] += np.array([0, MOVE_SPEED, 0])

                    elif ch in ("d", "D"):
                        state["move"] += np.array([0, -MOVE_SPEED, 0])

                    elif ch in ("q", "Q"):
                        state["move"] += np.array([0, 0, MOVE_SPEED])

                    elif ch in ("e", "E"):
                        state["move"] += np.array([0, 0, -MOVE_SPEED])

            time.sleep(0.01)

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print(f"Loading {XML_PATH}")

    init_tts()

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)

    voice_controller = GeminiVoiceController()

    kb = threading.Thread(target=keyboard_thread, daemon=True)
    kb.start()

    speak("Sistema listo.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.3, 0.0, 0.9]
        viewer.cam.distance = 1.6
        viewer.cam.elevation = -25
        viewer.cam.azimuth = 160

        while viewer.is_running():
            t0 = time.perf_counter()

            with state_lock:
                if state["quit"]:
                    break

                paused = state["paused"]
                do_reset = state["reset"]
                move = state["move"].copy()
                book_open = state["book_open"]
                lamp_on = state["lamp_on"]
                voice_request = state["voice_request"]

                state["move"] = np.zeros(3)
                state["reset"] = False
                state["voice_request"] = False

            if voice_request:
                print("[ASSISTANT] Te escucho.")
                intent = voice_controller.get_intent()
                apply_intent(intent)

            if do_reset:
                reset_scene(model, data)

            if not paused:
                if np.any(move != 0):
                    move_book(model, data, move)

                if book_open:
                    open_book(model, data)
                else:
                    close_book(model, data)

                set_flashlight(model, lamp_on)

                if lamp_on:
                    track_book_yaw_only(model, data)
                else:
                    data.ctrl[:] *= 0.95

                mujoco.mj_step(model, data)

            viewer.sync()

            elapsed = time.perf_counter() - t0
            remaining = DT - elapsed

            if remaining > 0:
                time.sleep(remaining)

    speak("Hasta luego.")
    print("Bye")


if __name__ == "__main__":
    main()