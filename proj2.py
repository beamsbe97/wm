import cv2
from picamera2 import Picamera2
import numpy as np
import picar_4wd as fc
import time
import threading
from google import genai
from google.genai import types

# --- API Config ---
client = genai.Client(api_key="AIzaSyApMtngKbEFBg-xecfmeiqzUBk4AVdYkXc")
MODEL = "gemini-3.1-flash-image-preview"

# --- Motor tuning ---
BASE_SPEED    = 3
TURN_SPEED    = 15
BYPASS_SPEED  = 15   # Extra push when ramming a light/flat obstacle

# --- Live feed config ---
DISPLAY_SIZE  = (480, 360)   # Width x Height of the OpenCV preview window
DISPLAY_FPS   = 30           # Target refresh rate of the preview (independent of AI loop)

# Predefined command set — the only strings Gemini is allowed to return
VALID_COMMANDS = {"forward", "forward_push", "ride_over", "stop"}
FALLBACK_CMD   = "stop"

# Shared state between the AI loop and the display thread
_frame_lock      = threading.Lock()
_latest_frame    = None   # Most recent camera frame (BGR, full resolution)
_latest_command  = "stop" # Most recent Gemini command, shown as overlay


def encode_image_for_gemini(image_np, size=(320, 240), quality=60):
    small = cv2.resize(image_np, size)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    ok, buffer = cv2.imencode(".jpg", small, encode_param)
    if not ok:
        raise ValueError("Image encoding failed")
    return buffer.tobytes()


# ---------------------------------------------------------------------------
# Live feed thread
# ---------------------------------------------------------------------------

CMD_COLORS = {
    "forward":      (0, 200,   0),   # green
    "forward_push": (0, 165, 255),   # orange
    "ride_over":    (255, 200,  0),  # cyan-ish
    "turn_left":    (255,  50, 50),  # blue
    "turn_right":   (50,   50, 255), # red
    "stop":         (0,    0, 200),  # dark red
}

def display_thread():
    """Continuously show the latest camera frame in an OpenCV window."""
    global _latest_frame, _latest_command

    window_name = "PiCar Live Feed"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, *DISPLAY_SIZE)

    interval = 1.0 / DISPLAY_FPS

    while True:
        with _frame_lock:
            frame = _latest_frame.copy() if _latest_frame is not None else None
            cmd   = _latest_command

        if frame is not None:
            # Resize for display
            display = cv2.resize(frame, DISPLAY_SIZE)

            # Overlay: command label
            color = CMD_COLORS.get(cmd, (200, 200, 200))
            cv2.rectangle(display, (0, 0), (DISPLAY_SIZE[0], 36), (30, 30, 30), -1)
            cv2.putText(display, f"CMD: {cmd}", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

            cv2.imshow(window_name, display)

        # q to quit (only works if the window has focus)
        if cv2.waitKey(int(interval * 1000)) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# World model
# ---------------------------------------------------------------------------

class WorldModel:
    def __init__(self):
        self.current_action = "stop"

    def get_next_command(self, image_np):
        buffer = encode_image_for_gemini(image_np)

        prompt = f"""You are the world model for a PiCar 4WD robot with a front camera.

The PiCar is a small 4-wheeled robot car, roughly 20x15 cm, weighing ~500g.

STEP 1 — Detect the hand:
Is a human hand visible anywhere in the frame?
- Yes → the robot's goal is to move forward toward it.
- No  → output: stop (no other steps needed).

STEP 2 — Environmental safety check:
Before thinking about obstacles, rule out terrain hazards:
- Ledge, drop-off, or floor discontinuity within ~15 cm → stop (overrides everything)
- Ledge detected near hand → stop (overrides everything)
- Stairs ahead → stop
- Wall or fixed structure within ~15 cm → stop

STEP 3 — Obstacle assessment (only if Step 2 is safe):
Is there an object between the robot and the hand?
If yes, classify it honestly based on what you can see:

  FLAT / RIDE-OVER (robot can drive over it):
  - Paper, thin cardboard sheet, fabric, a fallen cable, a flat mat
  - Height clearly less than ~2 cm
  → output: ride_over

  LIGHT / KNOCKABLE (robot can push it aside):
  - Plastic bottle, small cardboard box, styrofoam cup, crumpled paper ball,
    ping pong ball, small toy — anything that looks hollow, light, or easily displaced
  - Object appears movable by a gentle bump from a 500g car
  → output: forward_push

  HEAVY / IMMOVABLE (robot must go around):
  - Furniture leg, book, brick, metal object, dense wooden block, anything
    that clearly outweighs or is fixed to the floor
  - If you are uncertain whether it's light or heavy, treat it as heavy
  → choose turn_left or turn_right (pick whichever side has more open space)

  NO OBSTACLE in path → output: forward

STEP 4 — Transition dynamics:
Given the current action "forward", consider what happens in ~3 seconds:
- forward / forward_push / ride_over → robot closes distance with what's ahead
- stop       → no change
Would the predicted future state create a new hazard? If so, adjust your output.

STEP 5 — Output:
Reply with ONLY one token from this exact set (no punctuation, no explanation):
  forward | forward_push | ride_over | stop

Priority order:
1. Terrain hazard (Step 2) → stop
2. No hand visible (Step 1) → stop
3. Heavy/immovable obstacle → stop
4. Light obstacle → forward_push
5. Flat obstacle → ride_over
6. Clear path with hand visible → forward
"""

        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[
                    types.Part(text=prompt),
                    types.Part(inline_data=types.Blob(
                        mime_type="image/jpeg",
                        data=buffer
                    ))
                ]
            )
            cmd = response.text.strip().lower().split()[0]
            if cmd not in VALID_COMMANDS:
                print(f"Unexpected Gemini output '{cmd}' — defaulting to stop.")
                return FALLBACK_CMD
            return cmd

        except Exception as e:
            print(f"Gemini API error: {e}")
            return FALLBACK_CMD


# ---------------------------------------------------------------------------
# Motor execution
# ---------------------------------------------------------------------------

def drive(command):
    if command == "forward":
        fc.forward(BASE_SPEED)
        print("▶  forward")
    elif command == "forward_push":
        fc.forward(BYPASS_SPEED)
        print("▶▶ forward_push (ramming light obstacle)")
    elif command == "ride_over":
        fc.forward(BASE_SPEED)
        print("▶~ ride_over (flat obstacle)")
    elif command == "turn_left":
        fc.turn_left(TURN_SPEED)
        print("◀  turn_left")
    elif command == "turn_right":
        fc.turn_right(TURN_SPEED)
        print("▶  turn_right")
    else:
        fc.stop()
        print("■  stop")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_challenge():
    global _latest_frame, _latest_command

    wm = WorldModel()

    # Start the display thread (daemon so it dies when main exits)
    t = threading.Thread(target=display_thread, daemon=True)
    t.start()

    with Picamera2() as camera:
        camera.preview_configuration.main.size = (640, 480)
        camera.preview_configuration.main.format = "RGB888"
        camera.configure("preview")
        camera.start()
        print("Camera started. Waiting for hand...")

        try:
            while True:
                img = camera.capture_array()

                # Push latest frame to display thread
                with _frame_lock:
                    _latest_frame = img.copy()

                command = wm.get_next_command(img)
                print(f"Gemini → {command}")

                # Update overlay label before driving
                with _frame_lock:
                    _latest_command = command

                #drive(command)
                wm.current_action = command

                time.sleep(1)

        finally:
            fc.stop()
            camera.close()
            cv2.destroyAllWindows()
            print("Stopped.")


if __name__ == "__main__":
    run_challenge()