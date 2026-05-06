"""
Follow Module (Priority 2)
===========================
When vision detects a target, this module:
  1. Pans/tilts the head to keep the target centred (PID)
  2. Walks forward/backward to maintain a set distance
  3. Announces known faces by name
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

log = logging.getLogger("follow")
log.info("Follow module starting")

# Target box size as fraction of frame (proxy for distance)
TARGET_SIZE_NEAR = 0.35   # too close — back up
TARGET_SIZE_FAR  = 0.10   # too far  — walk forward

# Head pan/tilt PID gains
KP_PAN  = 0.04
KP_TILT = 0.03

# Only follow these COCO labels (add more as needed)
FOLLOW_LABELS = {"person", "cat", "dog", "sports ball"}


class FollowModule(BaseModule):
    def __init__(self, dog, speech, vision, priority=2):
        super().__init__(dog, speech, priority)
        self._vision = vision
        self._last_greeted = {}   # name -> timestamp
        self._greet_cooldown = 30.0

        self._pan_angle  = 0
        self._tilt_angle = 0

    def should_act(self, sensor_data: dict) -> bool:
        vision = sensor_data.get("vision")
        if vision is None:
            return False
        target = vision.primary_target
        return target is not None and target.label in FOLLOW_LABELS

    def act(self, sensor_data: dict):
        vision = sensor_data["vision"]
        target = vision.primary_target
        if target is None:
            return

        # Greet known persons
        if target.known_name:
            self._maybe_greet(target.known_name)

        # Head tracking
        cx = (target.bbox[0] + target.bbox[2]) / 2
        cy = (target.bbox[1] + target.bbox[3]) / 2
        frame_cx = vision.frame_w / 2
        frame_cy = vision.frame_h / 2

        pan_error  = (cx - frame_cx) / frame_cx    # -1 .. +1
        tilt_error = (cy - frame_cy) / frame_cy

        self._pan_angle  = max(-40, min(40, self._pan_angle  - pan_error  * KP_PAN  * 1000))
        self._tilt_angle = max(-20, min(20, self._tilt_angle + tilt_error * KP_TILT * 1000))

        try:
            self.dog.head_move([[self._pan_angle, 0, self._tilt_angle]], immediately=True, speed=80)
        except Exception:
            pass

        # Body follow — use bounding box area as distance proxy
        bw = target.bbox[2] - target.bbox[0]
        bh = target.bbox[3] - target.bbox[1]
        box_frac = (bw * bh) / (vision.frame_w * vision.frame_h)

        if box_frac < TARGET_SIZE_FAR:
            self.dog.do_action("forward", speed=40)
        elif box_frac > TARGET_SIZE_NEAR:
            self.dog.do_action("backward", speed=30)
        else:
            self.dog.do_action("stop", speed=80)

    def _maybe_greet(self, name: str):
        now = time.time()
        last = self._last_greeted.get(name, 0)
        if now - last > self._greet_cooldown:
            self.speech.say(f"Hello {name}, good to see you!", priority=5)
            self._last_greeted[name] = now
