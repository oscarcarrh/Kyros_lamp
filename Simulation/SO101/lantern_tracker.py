"""
SO101 Smart Lantern Tracker
===========================

Features
--------
- Only RED book
- Book open/close with O
- Lamp ON/OFF with L
- Voice command with V + Gemini
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
import select
import termios
import tty
import threading

from dotenv import load_dotenv

import mujoco
import mujoco.viewer
import numpy as np

load_dotenv()

try:
    import speech_recognition as sr
except ImportError:
    sr = None

try:
    from google import genai
except ImportError:
    genai = None


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
# GEMINI VOICE CONTROLLER
# ─────────────────────────────────────────────

class GeminiVoiceController:
    def __init__(self):
        self.enabled = False

        if sr is None:
            print("[VOICE] SpeechRecognition no está instalado.")
            return

        if genai is None:
            print("[VOICE] google-genai no está instalado.")
            return

        if not os.getenv("GEMINI_API_KEY"):
            print("[VOICE] No existe GEMINI_API_KEY en variables de entorno.")
            return

        try:
            self.client = genai.Client()
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()

            with self.microphone as source:
                print("[VOICE] Ajustando ruido ambiente...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)

            self.enabled = True
            print("[VOICE] Control por voz listo. Presiona V para hablar.")

        except Exception as e:
            print(f"[VOICE] No se pudo inicializar voz: {e}")

    def listen_text(self):
        if not self.enabled:
            return None

        try:
            with self.microphone as source:
                print("\n[VOICE] Escuchando...")
                audio = self.recognizer.listen(
                    source,
                    timeout=5,
                    phrase_time_limit=4
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

    def text_to_action(self, text):
        if not text:
            return "NONE"

        prompt = f"""
Eres el controlador de una simulación de una lámpara inteligente con un libro.

Convierte el comando del usuario en exactamente una acción.

Acciones válidas:
- TURN_ON_LAMP
- TURN_OFF_LAMP
- OPEN_BOOK
- CLOSE_BOOK
- RESET
- NONE

Reglas:
- Si el usuario dice prender, encender, iluminar o activar la luz, responde TURN_ON_LAMP.
- Si el usuario dice apagar, oscurecer o desactivar la luz, responde TURN_OFF_LAMP.
- Si el usuario quiere abrir el libro, responde OPEN_BOOK.
- Si el usuario quiere cerrar el libro, responde CLOSE_BOOK.
- Si el usuario pide reiniciar o resetear, responde RESET.
- Si no hay intención clara, responde NONE.

Responde solamente con una de las acciones válidas.

Comando del usuario:
{text}
"""

        try:
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )

            action = response.text.strip().upper()
            print(f"[VOICE] Acción Gemini: {action}")

            valid_actions = {
                "TURN_ON_LAMP",
                "TURN_OFF_LAMP",
                "OPEN_BOOK",
                "CLOSE_BOOK",
                "RESET",
                "NONE",
            }

            if action in valid_actions:
                return action

            return "NONE"

        except Exception as e:
            print(f"[VOICE] Error con Gemini: {e}")
            return "NONE"

    def get_action(self):
        text = self.listen_text()
        return self.text_to_action(text)


def apply_voice_action(action):
    with state_lock:
        if action == "TURN_ON_LAMP":
            state["lamp_on"] = True
            state["book_open"] = True
            print("[ACTION] Lámpara encendida")

        elif action == "TURN_OFF_LAMP":
            state["lamp_on"] = False
            print("[ACTION] Lámpara apagada")

        elif action == "OPEN_BOOK":
            state["book_open"] = True
            print("[ACTION] Libro abierto")

        elif action == "CLOSE_BOOK":
            state["book_open"] = False
            state["lamp_on"] = False
            print("[ACTION] Libro cerrado")

        elif action == "RESET":
            state["reset"] = True
            print("[ACTION] Reset")

        else:
            print("[ACTION] Sin acción")


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

                    elif ch in ("r", "R"):
                        state["reset"] = True
                        print("[KEY] Reset")

                    elif ch in ("o", "O"):
                        state["book_open"] = not state["book_open"]
                        if not state["book_open"]:
                            state["lamp_on"] = False
                        print(f"[KEY] Book open: {state['book_open']}")

                    elif ch in ("l", "L"):
                        state["lamp_on"] = not state["lamp_on"]
                        if state["lamp_on"]:
                            state["book_open"] = True
                        print(f"[KEY] Lamp on: {state['lamp_on']}")

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

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)

    voice_controller = GeminiVoiceController()

    kb = threading.Thread(target=keyboard_thread, daemon=True)
    kb.start()

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
                action = voice_controller.get_action()
                apply_voice_action(action)

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

    print("Bye")


if __name__ == "__main__":
    main()