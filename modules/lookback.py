"""
LookBack Module (Priority 3)
==========================
When vision detects a person looking at the robot, this module:
  1. Detects when a person is facing the robot (based on face orientation)
  2. Makes the robot look back at the person
  3. Announces recognition and starts a conversation about the person's face
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

log = logging.getLogger("lookback")
log.info("LookBack module starting")

# Face orientation thresholds
FACE_FRONT_ANGLE = 30   # degrees - face is facing the robot
FACE_SIDE_ANGLE  = 60   # degrees - face is looking to the side

# Conversation parameters
CONVERSATION_TIMEOUT = 30.0  # seconds between conversations
MIN_FACE_SIZE        = 0.05  # minimum face size as fraction of frame


class LookBackModule(BaseModule):
    def __init__(self, dog, speech, vision, priority=3):
        super().__init__(dog, speech, priority)
        self._vision = vision
        self._last_conversation_t = 0
        self._last_target = None
        self._last_target_time = 0

        # Conversation state
        self._conversation_active = False
        self._conversation_start_time = 0

        # Known faces for conversation
        self._known_faces = set()

    def should_act(self, sensor_data: dict) -> bool:
        """Return True if this module wants to take control."""
        vision = sensor_data.get("vision")
        if vision is None:
            return False

        # Check if there are any faces detected
        if not vision.detections:
            return False

        # Check if any person is looking at the robot
        return self._detect_face_orientation(vision) is not None

    def act(self, sensor_data: dict):
        """Execute one tick of this module's behaviour."""
        vision = sensor_data["vision"]
        target = self._detect_face_orientation(vision)

        if target is None:
            return

        # Update target tracking
        self._last_target = target
        self._last_target_time = time.time()

        # Make robot look back at the person
        self._look_back_at_person(target, vision)

        # Start conversation if appropriate
        self._maybe_start_conversation(target, vision)

    def _detect_face_orientation(self, vision) -> dict:
        """Detect if any person is looking at the robot and return their detection."""
        # Look for people in the vision data
        persons = [d for d in vision.detections if d.is_person]

        if not persons:
            return None

        # For now, just return the first person detected
        # In a more sophisticated implementation, we'd analyze face orientation
        # For this implementation, we'll consider any person as a potential target
        return persons[0]

    def _look_back_at_person(self, target, vision):
        """Make the robot look back at the person."""
        # Get face center
        cx = (target.bbox[0] + target.bbox[2]) / 2
        cy = (target.bbox[1] + target.bbox[3]) / 2
        frame_cx = vision.frame_w / 2
        frame_cy = vision.frame_h / 2

        # Calculate head movement to look at the person
        pan_error  = (cx - frame_cx) / frame_cx    # -1 .. +1
        tilt_error = (cy - frame_cy) / frame_cy

        # Apply head movement (simplified - in reality this would be PID control)
        pan_angle  = max(-40, min(40, pan_error  * 0.05 * 1000))
        tilt_angle = max(-20, min(20, tilt_error * 0.03 * 1000))

        try:
            # Move head to look at the person
            self.dog.head_move([[pan_angle, 0, tilt_angle]], immediately=True, speed=80)
        except Exception as e:
            log.warning(f"Head movement failed: {e}")

    def _maybe_start_conversation(self, target, vision):
        """Start a conversation about the person if appropriate."""
        now = time.time()

        # Check if enough time has passed since last conversation
        if now - self._last_conversation_t < CONVERSATION_TIMEOUT:
            return

        # Check if the person is still looking at the robot
        if now - self._last_target_time > 5.0:  # Person has stopped looking
            return

        # Check if face is large enough
        face_width = target.bbox[2] - target.bbox[0]
        face_height = target.bbox[3] - target.bbox[1]
        face_size = (face_width * face_height) / (vision.frame_w * vision.frame_h)

        if face_size < MIN_FACE_SIZE:
            return

        # Start conversation
        self._start_conversation(target)
        self._last_conversation_t = now

    def _start_conversation(self, target):
        """Start a conversation with the person."""
        # Determine if person is known or unknown
        if target.known_name:
            greeting = f"Hello {target.known_name}! It's nice to see you!"
            self.speech.say(greeting, priority=5)
            self._known_faces.add(target.known_name)
        else:
            # Unknown person - make some observations
            observations = [
                "I see you! Are you a friend?",
                "Hello there! Nice to meet you!",
                "You look familiar. Are we friends?",
                "I like the way you look! Are you a friend?"
            ]
            self.speech.say(random.choice(observations), priority=6)

        # Add some personality to the conversation
        personality_lines = [
            "I love meeting new people!",
            "You have a very interesting face!",
            "I like the way you smile!",
            "You look very friendly!",
            "I'm glad you're here!"
        ]

        # Randomly choose a personality line
        if random.random() < 0.7:  # 70% chance of saying something
            self.speech.say(random.choice(personality_lines), priority=7)

        # Set conversation state
        self._conversation_active = True
        self._conversation_start_time = time.time()

    def _end_conversation(self):
        """End the current conversation."""
        if self._conversation_active:
            self.speech.say("It was nice talking to you!", priority=8)
            self._conversation_active = False