"""
Emotion / Mood Module (Priority 4)
=====================================
Finite state machine with moods: idle, curious, happy, excited, tired, cautious.

Transitions driven by:
  - Time since last stimulus
  - Battery level
  - Vision detections (known person → excited, stranger → cautious)
  - Touch sensor
  - Random drift for personality variation
"""

import time
import random
from modules import BaseModule

# Mood definitions: (oled_expression, led_colour_rgb, idle_action, speech_lines)
MOODS = {
    "idle": {
        "expr":   "neutral",
        "led":    (20, 20, 20),
        "action": "stand",
        "lines":  ["I am here.", "Waiting...", "All quiet."],
    },
    "curious": {
        "expr":   "curious",
        "led":    (0, 80, 200),
        "action": "look_around",
        "lines":  ["What is that?", "Interesting.", "Let me see..."],
    },
    "happy": {
        "expr":   "happy",
        "led":    (0, 200, 80),
        "action": "wag_tail",
        "lines":  ["This is great!", "I love exploring.", "Woof!"],
    },
    "excited": {
        "expr":   "excited",
        "led":    (255, 160, 0),
        "action": "wave_hand",
        "lines":  ["Oh wow!", "Hello! Hello!", "I see a friend!"],
    },
    "tired": {
        "expr":   "sleepy",
        "led":    (60, 0, 60),
        "action": "sit_down",
        "lines":  ["I need a rest.", "So sleepy...", "Just five minutes."],
    },
    "cautious": {
        "expr":   "alert",
        "led":    (200, 100, 0),
        "action": "stand",
        "lines":  ["Someone is there.", "I am watching.", "Stay back."],
    },
}

IDLE_SPEECH_INTERVAL  = 45   # seconds between idle utterances
MOOD_PERSIST_MIN      = 8    # minimum seconds to stay in a mood
BORED_THRESHOLD       = 30   # seconds of no stimulus → curious


class EmotionModule(BaseModule):
    def __init__(self, dog, speech, vision, priority=4):
        super().__init__(dog, speech, priority)
        self._vision  = vision
        self._mood    = "idle"
        self._mood_t  = time.time()
        self._last_speech_t    = time.time()
        self._last_stimulus_t  = time.time()

    def should_act(self, sensor_data: dict) -> bool:
        # Always willing to update mood, but yields to higher priority modules
        return True

    def act(self, sensor_data: dict):
        self._update_mood(sensor_data)
        self._apply_mood()
        self._maybe_speak()

    def idle(self, sensor_data: dict):
        """Called by main loop when no other module acted."""
        self.act(sensor_data)

    def notify_stimulus(self):
        self._last_stimulus_t = time.time()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _update_mood(self, sensor_data: dict):
        now       = time.time()
        battery   = sensor_data.get("battery_pct", 100)
        touched   = sensor_data.get("is_touched", False)
        vision    = sensor_data.get("vision")
        in_mood_t = now - self._mood_t

        if in_mood_t < MOOD_PERSIST_MIN:
            return   # stay in current mood

        # Priority rules for transitions
        if battery < 15:
            self._set_mood("tired")
            return

        if touched:
            self._set_mood("happy")
            self._last_stimulus_t = now
            return

        if vision:
            known = vision.known_persons
            strangers = [d for d in vision.persons if not d.known_name]
            if known:
                self._set_mood("excited")
                self._last_stimulus_t = now
                return
            if strangers:
                self._set_mood("cautious")
                self._last_stimulus_t = now
                return

        secs_idle = now - self._last_stimulus_t
        if secs_idle > BORED_THRESHOLD and self._mood == "idle":
            self._set_mood("curious")
            return

        if secs_idle < 5:
            self._set_mood("happy")
            return

        # Random drift between idle and curious
        if self._mood == "curious" and random.random() < 0.1:
            self._set_mood("idle")

    def _set_mood(self, mood: str):
        if mood != self._mood:
            print(f"[Emotion] Mood: {self._mood} → {mood}")
            self._mood   = mood
            self._mood_t = time.time()

    def _apply_mood(self):
        mood_cfg = MOODS.get(self._mood, MOODS["idle"])

        # LED colour
        try:
            r, g, b = mood_cfg["led"]
            self.dog.set_rgb_led(r, g, b)
        except Exception:
            pass

        # OLED face
        try:
            self.dog.set_face(mood_cfg["expr"])
        except Exception:
            pass

    def _maybe_speak(self):
        now = time.time()
        if now - self._last_speech_t < IDLE_SPEECH_INTERVAL:
            return
        mood_cfg = MOODS.get(self._mood, MOODS["idle"])
        line     = random.choice(mood_cfg["lines"])
        self.speech.say(line, priority=8)
        self._last_speech_t = now
