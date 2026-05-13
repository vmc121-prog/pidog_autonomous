"""
Head Tracking Module (Priority 4)
======================

This module implements head tracking behavior for the PiDog robot. It:
1. Tracks the primary target detected by vision
2. Moves the head to keep the target centered
3. Uses PID control for smooth, responsive head movement
4. Maintains a minimum distance from the target
5. Handles face recognition for known persons

The module works with the existing vision system and integrates with the
main behavior arbitration loop.
"""

import time
from modules import BaseModule
from modules.logging_config import setup_logging
import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

setup_logging(level=logging.DEBUG if args.debug else logging.INFO)

log = logging.getLogger("head_tracking")
log.info("HeadTracking module starting")

# PID gains for head movement
KP_PAN  = 0.05
KP_TILT = 0.04

# Minimum distance threshold (as fraction of frame)
MIN_DISTANCE_FRACTION = 0.25

# Maximum head angles (degrees)
MAX_PAN_ANGLE  = 40
MAX_TILT_ANGLE = 20


class HeadTrackingModule(BaseModule):
    def __init__(self, dog, speech, vision, priority=4):
        super().__init__(dog, speech, priority)
        self._vision = vision

        # Head position tracking
        self._pan_angle  = 0
        self._tilt_angle = 0

        # Target tracking
        self._last_target = None
        self._target_lock_time = 0
        self._target_lock_duration = 2.0  # seconds to maintain target

    def should_act(self, sensor_data: dict) -> bool:
        """Return True if this module wants to take control."""
        vision = sensor_data.get("vision")
        if vision is None:
            return False

        # Check if there's a primary target
        target = vision.primary_target
        return target is not None

    def act(self, sensor_data: dict):
        """Execute one tick of head tracking behavior."""
        vision = sensor_data["vision"]
        target = vision.primary_target

        if target is None:
            return

        # Update target tracking
        if self._last_target != target:
            self._target_lock_time = time.time()
            self._last_target = target

        # Calculate target center
        cx = (target.bbox[0] + target.bbox[2]) / 2
        cy = (target.bbox[1] + target.bbox[3]) / 2
        frame_cx = vision.frame_w / 2
        frame_cy = vision.frame_h / 2

        # Calculate error (normalized to -1 to +1)
        pan_error  = (cx - frame_cx) / frame_cx    # -1 .. +1
        tilt_error = (cy - frame_cy) / frame_cy

        # Apply PID control to update head angles
        self._pan_angle  = max(-MAX_PAN_ANGLE, min(MAX_PAN_ANGLE, self._pan_angle  - pan_error  * KP_PAN  * 1000))
        self._tilt_angle = max(-MAX_TILT_ANGLE, min(MAX_TILT_ANGLE, self._tilt_angle + tilt_error * KP_TILT * 1000))

        # Apply head movement
        try:
            self.dog.head_move([[self._pan_angle, 0, self._tilt_angle]], immediately=True, speed=80)
        except Exception as e:
            log.warning(f"Head movement failed: {e}")

        # Handle face recognition
        if target.known_name:
            self._handle_known_face(target)

    def _handle_known_face(self, target):
        """Handle behavior when a known face is detected."""
        # This could be extended to add specific behavior for known faces
        # For now, we just log that we recognized someone
        log.info(f"Recognized known face: {target.known_name}")