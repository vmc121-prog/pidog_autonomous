#!/usr/bin/env python3
"""
PiDog Autonomous Brain
======================
Main state machine loop. Reads all sensors, arbitrates between behaviour
modules by priority, and drives actuators.

Priority order (lowest number = highest priority):
  1. Obstacle avoidance
  2. Follow (person / object)
  3. Voice / sound reactive
  4. Emotion / mood FSM
  5. Mission / patrol
"""
import json
import time
import threading
import signal
import sys
import argparse
import logging

from modules.vision_viewer   import start_viewer
from modules.obstacle        import ObstacleModule
from modules.follow          import FollowModule
from modules.voice           import VoiceModule
from modules.emotion         import EmotionModule
from modules.mission         import MissionModule
from modules.speech          import SpeechModule
from modules.vision          import VisionModule
from modules.logging_config  import setup_logging

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

# ── Logging ───────────────────────────────────────────────────────────────────
setup_logging(level=logging.DEBUG if args.debug else logging.INFO)
log = logging.getLogger("Main")
log.info("Main starting")

# ── PiDog import (falls back to mock if library not present) ──────────────────
try:
    from pidog import Pidog
except ImportError:
    print("[WARN] pidog library not found - running in simulation mode")
    from modules.mock_pidog import MockPidog as Pidog

# ── Constants ─────────────────────────────────────────────────────────────────
LOOP_HZ     = 20
LOOP_PERIOD = 1.0 / LOOP_HZ


# ── JSON encoder that handles custom sensor objects ───────────────────────────
class SensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return str(obj)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=== PiDog Autonomous System Starting ===")

    # Hardware
    dog = Pidog()
    dog.do_action("stand", speed=50)
    time.sleep(1)

    # Vision — single instance shared by all modules
    vm = VisionModule(camera_index=0)
    vm.start()

    # Start web viewer in background so it doesn't block the main loop
    threading.Thread(
        target=start_viewer,
        args=(vm,),
        daemon=True,
    ).start()

    # Speech
    speech = SpeechModule()

    # Behaviour modules — all share the same vm instance
    modules = [
        ObstacleModule(dog, speech,     priority=1),
        FollowModule(dog, speech, vm,   priority=2),
        VoiceModule(dog, speech,        priority=3),
        EmotionModule(dog, speech, vm,  priority=4),
        MissionModule(dog, speech,      priority=5),
    ]

    # Start background threads
    speech.start()
    for m in modules:
        m.start()

    speech.say("PiDog online. Ready to explore.", priority=9)

    # ── Graceful shutdown handler ─────────────────────────────────────────────
    running = True

    def _shutdown(sig, frame):
        nonlocal running
        print("\n[INFO] Shutting down...")
        running = False

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Main loop ─────────────────────────────────────────────────────────────
    log.info("Entering main loop (Ctrl+C to stop)")

    while running:
        t_start = time.time()

        # Gather shared sensor snapshot
        sensor_data = {
            "distance_cm": dog.ultrasonic_distance if hasattr(dog, "ultrasonic_distance") else 999,
            "vision":      vm.get_latest(),
            "battery_pct": dog.get_battery_percentage() if hasattr(dog, "get_battery_percentage") else 100,
            "is_touched":  dog.is_touched() if hasattr(dog, "is_touched") else False,
        }

        log.debug("Full sensor dump:\n%s", json.dumps(sensor_data, indent=2, cls=SensorEncoder))

        # Find highest-priority module that wants to act
        for module in sorted(modules, key=lambda m: m.priority):
            try:
                if module.should_act(sensor_data):
                    module.act(sensor_data)
                    break
            except NotImplementedError:
                print(f"[WARN] {module.__class__.__name__} missing should_act()")
            except Exception as e:
                print(f"[WARN] {module.__class__.__name__} failed: {e}")

        # Pace the loop
        elapsed = time.time() - t_start
        time.sleep(max(0, LOOP_PERIOD - elapsed))

    # ── Clean shutdown ────────────────────────────────────────────────────────
    log.info("Stopping modules...")
    for m in modules:
        m.stop()
    vm.stop()
    speech.stop()
    dog.do_action("lie", speed=50)
    speech.say("Goodbye!", priority=9)
    time.sleep(1)
    dog.close()
    log.info("Goodbye!")


if __name__ == "__main__":
    main()