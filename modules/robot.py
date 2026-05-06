import time
import csv
import os
import threading
import pyttsx3
from vision import VisionModule, VisionResult

# ── TTS Engine ────────────────────────────────────────────────────────────────
tts = pyttsx3.init()
tts.setProperty("rate", 160)
_tts_lock = threading.Lock()

def speak(text: str):
    """Non-blocking TTS announcement."""
    def _speak():
        with _tts_lock:
            tts.say(text)
            tts.runAndWait()
    threading.Thread(target=_speak, daemon=True).start()

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE = "detections.csv"

def log_detection(label: str, confidence: float, known_name: str = None):
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "label", "confidence", "known_name"])
        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            label,
            f"{confidence:.2f}",
            known_name or ""
        ])

# ── Motor Control (stub — replace with your driver) ──────────────────────────
def steer_towards(bbox, frame_w: int):
    """
    Turn robot to centre the target in frame.
    bbox = (x1, y1, x2, y2)
    Replace the print() calls with your actual motor commands.
    """
    x1, _, x2, _ = bbox
    cx = (x1 + x2) / 2
    centre = frame_w / 2
    offset = cx - centre          # negative = target is left, positive = right
    dead_zone = frame_w * 0.1    # 10% of frame width

    if abs(offset) < dead_zone:
        print("Motors: STRAIGHT")
        # motors.forward()
    elif offset < 0:
        print(f"Motors: TURN LEFT  (offset {offset:.0f}px)")
        # motors.left()
    else:
        print(f"Motors: TURN RIGHT (offset {offset:.0f}px)")
        # motors.right()

# ── Announcement Throttle ─────────────────────────────────────────────────────
last_announced: dict[str, float] = {}
ANNOUNCE_COOLDOWN = 10  # seconds between repeat announcements

def maybe_announce(key: str, message: str):
    now = time.time()
    if now - last_announced.get(key, 0) > ANNOUNCE_COOLDOWN:
        last_announced[key] = now
        speak(message)

# ── Main Loop ─────────────────────────────────────────────────────────────────
def main():
    vision = VisionModule(width=640, height=480)
    vision.start()

    # Optional: register known faces before starting
    # vision.register_face("Alice", "faces/alice.jpg")

    print("Robot running. Press Ctrl+C to stop.")
    prev_labels = set()

    try:
        while True:
            result: VisionResult = vision.get_latest()

            # ── Track primary target (largest detection) ──────────────────
            target = result.primary_target
            if target and target.is_person:
                steer_towards(target.bbox, result.frame_w)

            # ── Announce & log new detections ─────────────────────────────
            current_labels = set()
            for det in result.detections:
                key = det.known_name or det.label
                current_labels.add(key)

                # Only announce/log if this is a new detection this cycle
                if key not in prev_labels:
                    if det.known_name:
                        msg = f"I can see {det.known_name}"
                    elif det.is_person:
                        msg = "Unknown person detected"
                    else:
                        msg = f"I can see a {det.label}"

                    maybe_announce(key, msg)
                    log_detection(det.label, det.confidence, det.known_name)
                    print(f"[DETECT] {msg} ({det.confidence:.0%})")

            prev_labels = current_labels
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        vision.stop()

if __name__ == "__main__":
    main()