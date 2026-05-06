"""
Obstacle Avoidance Module (Priority 1)
=======================================
Uses the ultrasonic sensor to detect obstacles and steer clear.
Takes control whenever something is within DANGER_CM.
"""

import time
import random
from modules import BaseModule
from modules.logging_config import setup_logging
import logging
import argparse, logging

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

setup_logging(level=logging.DEBUG if args.debug else logging.INFO)
# setup_logging()           # call once at startup

log = logging.getLogger("Main")
log.info("Obstacle module starting")

DANGER_CM  = 30    # stop and turn
CAUTION_CM = 60    # slow / prepare to turn
SCAN_PAUSE = 0.4   # seconds to pause before turning


class ObstacleModule(BaseModule):
    def __init__(self, dog, speech, priority=1):
        super().__init__(dog, speech, priority)
        self._last_alert_t = 0
        self._alert_cooldown = 5.0  # seconds between speech alerts

    def should_act(self, sensor_data: dict) -> bool:
        return sensor_data.get("distance_cm", 999) < DANGER_CM

    def act(self, sensor_data: dict):
        dist = sensor_data.get("distance_cm", 999)

        # Speech alert (rate-limited)
        now = time.time()
        if now - self._last_alert_t > self._alert_cooldown:
            self.speech.say(f"Obstacle detected at {int(dist)} centimetres.", priority=2)
            self._last_alert_t = now

        # Stop
        self.dog.do_action("stop", speed=80)
        time.sleep(0.1)

        # Scan left and right to find clear direction
        self.dog.do_action("turn_left", speed=50)
        time.sleep(SCAN_PAUSE)
        left_dist = self._read_distance()

        self.dog.do_action("turn_right", speed=50)
        time.sleep(SCAN_PAUSE * 2)
        right_dist = self._read_distance()

        # Return to centre, pick better side
        self.dog.do_action("stop", speed=80)
        time.sleep(0.1)

        if left_dist > right_dist and left_dist > DANGER_CM:
            self.dog.do_action("turn_left", speed=60)
            self.speech.say("Turning left.", priority=2)
        elif right_dist > DANGER_CM:
            self.dog.do_action("turn_right", speed=60)
            self.speech.say("Turning right.", priority=2)
        else:
            # Both sides blocked — back up
            self.dog.do_action("backward", speed=50)
            self.speech.say("Both sides blocked. Reversing.", priority=2)
            time.sleep(0.8)

        time.sleep(0.4)

    def _read_distance(self) -> float:
        try:
            return self.dog.ultrasonic.get_distance()
        except Exception:
            return 999
