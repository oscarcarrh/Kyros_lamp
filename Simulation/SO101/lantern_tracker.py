"""
SO101 Smart Lantern Tracker with Computer Vision
================================================

Controls
--------
WASD  -> move book
Q/E   -> move book up/down
O     -> open/close book
L     -> lamp on/off
V     -> vision debug on/off
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
import cv2

XML_PATH = "my_smart_lantern_scene.xml"

DT = 0.002
MOVE_SPEED = 0.003

TABLE_Z = 0.757
BOOK_Z = TABLE_Z + 0.015
BOOK_START = np.array([0.35, -0.18, BOOK_Z])

CAMERA_NAME = "gripper_cam"
CAM_W = 480
CAM_H = 360

YAW_GAIN = 0.003
PITCH_GAIN = 0.002

YAW_LIMIT = (-1.91986, 1.91986)
WRIST_LIMIT = (0.35, 1.15)

state = {
    "move": np.zeros(3),
    "paused": False,
    "quit": False,
    "reset": False,
    "lamp_on": False,
    "vision_debug": True,
}

HOME_POSE = np.array([
    0.0,    # shoulder_pan
    0.25,   # shoulder_lift
    -0.65,  # elbow_flex
    0.85,   # wrist_flex
    0.0,    # wrist_roll
    0.3
])

state_lock = threading.Lock()


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




def set_lamp(model, enabled):
    light_id = model.light("desk_light").id
    mat_id = model.material("lamp_yellow").id

    if enabled:
        model.light_diffuse[light_id] = np.array([2.0, 1.6, 0.9])
        model.light_specular[light_id] = np.array([0.35, 0.30, 0.20])

        model.mat_rgba[mat_id] = np.array([1.0, 0.85, 0.25, 1.0])
        model.mat_emission[mat_id] = 1.0

    else:
        model.light_diffuse[light_id] = np.array([0.0, 0.0, 0.0])
        model.light_specular[light_id] = np.array([0.0, 0.0, 0.0])

        model.mat_rgba[mat_id] = np.array([0.35, 0.30, 0.10, 1.0])
        model.mat_emission[mat_id] = 0.0


def reset_scene(model, data):
    set_book_pose(model, data, BOOK_START)

    data.ctrl[:] = HOME_POSE
    set_lamp(model, False)

    mujoco.mj_forward(model, data)


def detect_green_book(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    lower_green = np.array([40, 80, 60])
    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None, mask, 0

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    if area < 250:
        return None, None, mask, area

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None, None, mask, area

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    return cx, cy, mask, area


def vision_track_book(model, data, renderer, cam_id, debug=False):
    """
    Tracks the red book using the gripper camera.

    No MuJoCo book position is used here.
    The robot only uses image error:
        horizontal error -> shoulder pan
        vertical error   -> wrist flex
    """

    renderer.update_scene(data, camera=cam_id)
    rgb = renderer.render()

    cx, cy, mask, area = detect_green_book(rgb)

    if cx is None:
        data.ctrl[:] += 0.01 * (HOME_POSE - data.ctrl[:])

        if debug:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(
                bgr,
                "GREEN BOOK NOT DETECTED",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            cv2.imshow("gripper camera", bgr)
            cv2.imshow("green mask", mask)
            cv2.waitKey(1)

        return False

    h, w, _ = rgb.shape

    error_x = (cx - w / 2) / (w / 2)
    error_y = (cy - h / 2) / (h / 2)

    DEADBAND_X = 0.10
    DEADBAND_Y = 0.10

    MIN_TRACK_AREA = 2500

    MAX_YAW_STEP = 0.0012
    MAX_PITCH_STEP = 0.0010

    YAW_SIGN = -1.0
    YAW_RANGE = 0.8
    PITCH_SIGN = -1.0
    PITCH_RANGE = 0.45

    if area > MIN_TRACK_AREA:
        if abs(error_x) > DEADBAND_X:
            desired_yaw = HOME_POSE[0] + YAW_SIGN * YAW_RANGE * error_x
            data.ctrl[0] += 0.04 * (desired_yaw - data.ctrl[0])
        else:
            data.ctrl[0] += 0.04 * (HOME_POSE[0] - data.ctrl[0])

        

        if abs(error_y) > DEADBAND_Y:
            desired_pitch = HOME_POSE[3] + PITCH_SIGN * PITCH_RANGE * error_y
            data.ctrl[3] += 0.04 * (desired_pitch - data.ctrl[3])
        else:
            data.ctrl[3] += 0.04 * (HOME_POSE[3] - data.ctrl[3])

    data.ctrl[0] = np.clip(data.ctrl[0], YAW_LIMIT[0], YAW_LIMIT[1])
    data.ctrl[3] = np.clip(data.ctrl[3], WRIST_LIMIT[0], WRIST_LIMIT[1])

    data.ctrl[1] = HOME_POSE[1]
    data.ctrl[2] = HOME_POSE[2]
    data.ctrl[4] = HOME_POSE[4]
    data.ctrl[5] = HOME_POSE[5]

    if debug:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        cv2.circle(bgr, (cx, cy), 8, (0, 255, 0), -1)
        cv2.line(bgr, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)
        cv2.line(bgr, (0, h // 2), (w, h // 2), (255, 255, 255), 1)

        cv2.putText(
            bgr,
            f"ex={error_x:.2f}, ey={error_y:.2f}, area={area:.0f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.imshow("gripper camera", bgr)
        cv2.imshow("green mask", mask)
        cv2.waitKey(1)

    return True


def keyboard_thread():
    print("\nControls:")
    print("WASD -> move book")
    print("Q/E  -> up/down")
    print("L    -> lamp on/off")
    print("V    -> vision debug on/off")
    print("R    -> reset")
    print("P    -> pause")
    print("ESC  -> quit\n")

    while True:
        if msvcrt.kbhit():
            ch = msvcrt.getwch()

            with state_lock:
                if ch == "\x1b":
                    state["quit"] = True
                    break

                elif ch in ("p", "P"):
                    state["paused"] = not state["paused"]

                elif ch in ("r", "R"):
                    state["reset"] = True

                elif ch in ("l", "L"):
                    state["lamp_on"] = not state["lamp_on"]

                elif ch in ("v", "V"):
                    state["vision_debug"] = not state["vision_debug"]

                elif ch in ("w", "W"):
                    state["move"] += [MOVE_SPEED, 0, 0]

                elif ch in ("s", "S"):
                    state["move"] += [-MOVE_SPEED, 0, 0]

                elif ch in ("a", "A"):
                    state["move"] += [0, MOVE_SPEED, 0]

                elif ch in ("d", "D"):
                    state["move"] += [0, -MOVE_SPEED, 0]

                elif ch in ("q", "Q"):
                    state["move"] += [0, 0, MOVE_SPEED]

                elif ch in ("e", "E"):
                    state["move"] += [0, 0, -MOVE_SPEED]

        time.sleep(0.01)


def main():
    print(f"Loading {XML_PATH}")

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)

    data.ctrl[:] = HOME_POSE
    mujoco.mj_forward(model, data)

    cam_id = model.camera(CAMERA_NAME).id
    renderer = mujoco.Renderer(model, height=CAM_H, width=CAM_W)

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
                lamp_on = state["lamp_on"]
                vision_debug = state["vision_debug"]

                state["reset"] = False

            if do_reset:
                reset_scene(model, data)

            if not paused:
                if np.any(move != 0):
                    move_book(model, data, move)

                set_lamp(model, lamp_on)

                if lamp_on:
                    vision_track_book(
                        model,
                        data,
                        renderer,
                        cam_id,
                        debug=vision_debug
                    )
                else:
                    data.ctrl[:] += 0.02 * (HOME_POSE - data.ctrl[:])

                mujoco.mj_step(model, data)

            viewer.sync()

            elapsed = time.perf_counter() - t0
            remaining = DT - elapsed

            if remaining > 0:
                time.sleep(remaining)

    renderer.close()
    cv2.destroyAllWindows()
    print("Bye")


if __name__ == "__main__":
    main()