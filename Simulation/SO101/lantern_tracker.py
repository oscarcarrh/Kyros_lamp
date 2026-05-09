"""
SO101 Smart Lantern Tracker
===========================

Features
--------
- Only RED book
- Book open/close with O
- Flashlight ON only when book is open
- Dark scene
- Robot tracks ONLY in YAW
- Spotlight attached to gripper
- No arm stretching

Controls
--------
WASD  -> move book
Q/E   -> move book up/down
O     -> open/close book
R     -> reset
P     -> pause
ESC   -> quit
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import threading
import msvcrt

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

# ─────────────────────────────────────────────
# SHARED STATE
# ─────────────────────────────────────────────

state = {
    "move": np.zeros(3),
    "paused": False,
    "quit": False,
    "reset": False,
    "book_open": False,
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

    data.qpos[qadr:qadr+3] = pos

    data.qpos[qadr+2] = max(data.qpos[qadr+2], TABLE_Z + 0.013)

    data.qpos[qadr] = np.clip(data.qpos[qadr], -0.40, 0.80)
    data.qpos[qadr+1] = np.clip(data.qpos[qadr+1], -0.45, 0.45)

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
# KEYBOARD THREAD
# ─────────────────────────────────────────────

def keyboard_thread():

    print("\nControls:")
    print("WASD -> move book")
    print("Q/E  -> up/down")
    print("O    -> open/close book")
    print("R    -> reset")
    print("P    -> pause")
    print("ESC  -> quit\n")

    while True:

        if msvcrt.kbhit():

            ch = msvcrt.getwch()

            with state_lock:

                if ch == '\x1b':

                    state["quit"] = True
                    break

                elif ch in ('p', 'P'):

                    state["paused"] = not state["paused"]

                elif ch in ('r', 'R'):

                    state["reset"] = True

                elif ch in ('o', 'O'):

                    state["book_open"] = not state["book_open"]

                elif ch in ('w', 'W'):

                    state["move"] += [MOVE_SPEED, 0, 0]

                elif ch in ('s', 'S'):

                    state["move"] += [-MOVE_SPEED, 0, 0]

                elif ch in ('a', 'A'):

                    state["move"] += [0, MOVE_SPEED, 0]

                elif ch in ('d', 'D'):

                    state["move"] += [0, -MOVE_SPEED, 0]

                elif ch in ('q', 'Q'):

                    state["move"] += [0, 0, MOVE_SPEED]

                elif ch in ('e', 'E'):

                    state["move"] += [0, 0, -MOVE_SPEED]

        time.sleep(0.01)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():

    print(f"Loading {XML_PATH}")

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)

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
                state["move"] = np.zeros(3)

                book_open = state["book_open"]

                state["reset"] = False

            if do_reset:

                reset_scene(model, data)

            if not paused:

                if np.any(move != 0):

                    move_book(model, data, move)

                if book_open:

                    open_book(model, data)
                    set_flashlight(model, True)

                    track_book_yaw_only(model, data)

                else:

                    close_book(model, data)
                    set_flashlight(model, False)

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