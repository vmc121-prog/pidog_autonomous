import time
import csv
import os
import threading
import pyttsx3
from modules.vision import VisionModule, VisionResult

class RobotController:
    def __init__(self, width=640, height=480):
        self.vision = VisionModule(width=width, height=height)
        self._prev_labels = set()
        self._last_announced: dict[str, float] = {}
        self.ANNOUNCE_COOLDOWN = 10

        # TTS
        self._tts = pyttsx3.init()
        self._tts.setProperty("rate", 160)
        self._tts_lock = threading.Lock()

    def start(self):
        self.vision.start()

    def stop(self):
        self.vision.stop()

    def update(self):
        """Call this every tick from your main loop."""
        result: VisionResult = self.vision.get_latest()

        # Track
        target = result.primary_target
        if target and target.is_person:
            self._steer_towards(target.bbox, result.frame_w)

        # Announce + log
        current_labels = set()
        for det in result.detections:
            key = det.known_name or det.label
            current_labels.add(key)
            if key not in self._prev_labels:
                msg = (
                    f"I can see {det.known_name}" if det.known_name
                    else "Unknown person detected" if det.is_person
                    else f"I can see a {det.label}"
                )
                self._maybe_announce(key, msg)
                self._log_detection(det.label, det.confidence, det.known_name)

        self._prev_labels = current_labels

    def _steer_towards(self, bbox, frame_w: int):
        x1, _, x2, _ = bbox
        cx = (x1 + x2) / 2
        offset = cx - frame_w / 2
        dead_zone = frame_w * 0.1

        if abs(offset) < dead_zone:
            pass  # motors.forward()
        elif offset < 0:
            pass  # motors.left()
        else:
            pass  # motors.right()

    def _maybe_announce(self, key: str, message: str):
        now = time.time()
        if now - self._last_announced.get(key, 0) > self.ANNOUNCE_COOLDOWN:
            self._last_announced[key] = now
            threading.Thread(target=self._speak, args=(message,), daemon=True).start()

    def _speak(self, text: str):
        with self._tts_lock:
            self._tts.say(text)
            self._tts.runAndWait()

    def _log_detection(self, label, confidence, known_name=None):
        log_file = "detections.csv"
        file_exists = os.path.exists(log_file)
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "label", "confidence", "known_name"])
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                label,
                f"{confidence:.2f}",
                known_name or ""
            ])