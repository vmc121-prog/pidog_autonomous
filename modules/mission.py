"""
Mission / Patrol Module (Priority 5)
=======================================
Executes timed waypoint patrol routes.
Routes are defined as lists of (action, duration_seconds) tuples.
Obstacle avoidance still interrupts via the priority system.
"""

import time
from modules import BaseModule
from modules.logging_config import setup_logging
import logging
import argparse, logging

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

setup_logging(level=logging.DEBUG if args.debug else logging.INFO)
# setup_logging()           # call once at startup

log = logging.getLogger("mission")
log.info("Mission module starting")

# Default patrol route — customise freely
DEFAULT_PATROL = [
    ("forward",    2.0),
    ("turn_left",  1.2),
    ("forward",    2.0),
    ("turn_left",  1.2),
    ("forward",    2.0),
    ("turn_left",  1.2),
    ("forward",    2.0),
    ("turn_left",  1.2),   # square loop
]


class MissionModule(BaseModule):
    def __init__(self, dog, speech, priority=5, patrol=None):
        super().__init__(dog, speech, priority)
        self._patrol      = patrol or DEFAULT_PATROL
        self._active      = False
        self._step        = 0
        self._step_start  = 0

    def start_patrol(self):
        self._active     = True
        self._step       = 0
        self._step_start = time.time()
        self.speech.say("Starting patrol route.", priority=6)
        print("[Mission] Patrol started.")

    def stop_patrol(self):
        self._active = False
        self.dog.do_action("stop", speed=80)
        self.speech.say("Patrol complete.", priority=6)
        print("[Mission] Patrol stopped.")

    def should_act(self, sensor_data: dict) -> bool:
        return self._active

    def act(self, sensor_data: dict):
        if self._step >= len(self._patrol):
            self.stop_patrol()
            return

        action, duration = self._patrol[self._step]
        now = time.time()

        if now - self._step_start >= duration:
            # Advance to next step
            self._step      += 1
            self._step_start = now
            if self._step < len(self._patrol):
                next_action, _ = self._patrol[self._step]
                print(f"[Mission] Step {self._step}: {next_action}")
                self.dog.do_action(next_action, speed=50)
        else:
            # Continue current step (keep issuing the action)
            self.dog.do_action(action, speed=50)
