"""
pidog_actions.py
================
Translates voice command strings into PiDog actions and sounds.

Usage (standalone, no dog hardware):
    python pidog_actions.py

Usage (from main script):
    from modules.pidog_actions import PiDogActionModule
    actions = PiDogActionModule(dog)          # pass your Pidog() instance
    actions.execute("sit")                    # blocking=False by default
    actions.execute("forward", blocking=True)
"""

import threading
import logging
import time

log = logging.getLogger("PiDogActions")

# ─────────────────────────────────────────────────────────────────────────────
# Action definitions
# Each entry: [method_name, head_pitch_adjust, speed]
#   head_pitch_adjust = -1  → leave head at its current pitch
#   head_pitch_adjust = 0   → reset to 0
#   head_pitch_adjust = N   → set to N degrees
# ─────────────────────────────────────────────────────────────────────────────
_ACTIONS = [
    # name                  pitch  speed
    ["stand",               0,     50],
    ["sit",                -30,    50],
    ["lie",                 0,     20],
    ["lie_with_hands_out",  0,     20],
    ["trot",                0,     95],
    ["forward",             0,     98],
    ["backward",            0,     98],
    ["turn_left",           0,     98],
    ["turn_right",          0,     98],
    ["doze_off",           -30,    90],
    ["stretch",             20,    20],
    ["push_up",            -30,    50],
    ["shake_head",         -1,     90],
    ["tilting_head",       -1,     60],
    ["wag_tail",           -1,    100],
]

# Sounds the dog can play (filename stem, no extension)
_SOUNDS = [
    "angry",
    "confused_1",
    "confused_2",
    "confused_3",
    "growl_1",
    "growl_2",
    "howling",
    "pant",
    "single_bark_1",
    "single_bark_2",
    "snoring",
    "woohoo",
]

# ─────────────────────────────────────────────────────────────────────────────
# Voice-command → action / sound mapping
# Keys are lower-case; values are (type, name) tuples.
#   type = "action"  → call dog.<name>(speed=...) and adjust head pitch
#   type = "sound"   → call dog.speak(<name>)
# Add as many synonyms as you like.
# ─────────────────────────────────────────────────────────────────────────────
COMMAND_MAP: dict[str, tuple[str, str]] = {
    # ── actions ──────────────────────────────────────────────────────────────
    "stand":                ("action", "stand"),
    "stand up":             ("action", "stand"),
    "get up":               ("action", "stand"),
    "up":                   ("action", "stand"),

    "sit":                  ("action", "sit"),
    "sit down":             ("action", "sit"),

    "lie":                  ("action", "lie"),
    "lie down":             ("action", "lie"),
    "lay down":             ("action", "lie"),
    "down":                 ("action", "lie"),

    "lie with hands out":   ("action", "lie_with_hands_out"),
    "paws out":             ("action", "lie_with_hands_out"),
    "sphinx":               ("action", "lie_with_hands_out"),

    "trot":                 ("action", "trot"),
    "jog":                  ("action", "trot"),
    "march":                ("action", "trot"),

    "forward":              ("action", "forward"),
    "go forward":           ("action", "forward"),
    "walk forward":         ("action", "forward"),
    "move forward":         ("action", "forward"),
    "advance":              ("action", "forward"),
    "go":                   ("action", "forward"),
    "walk":                 ("action", "forward"),

    "backward":             ("action", "backward"),
    "go backward":          ("action", "backward"),
    "walk backward":        ("action", "backward"),
    "move backward":        ("action", "backward"),
    "back":                 ("action", "backward"),
    "back up":              ("action", "backward"),
    "reverse":              ("action", "backward"),

    "turn left":            ("action", "turn_left"),
    "left":                 ("action", "turn_left"),
    "go left":              ("action", "turn_left"),

    "turn right":           ("action", "turn_right"),
    "right":                ("action", "turn_right"),
    "go right":             ("action", "turn_right"),

    "doze off":             ("action", "doze_off"),
    "doze":                 ("action", "doze_off"),
    "sleep":                ("action", "doze_off"),
    "nap":                  ("action", "doze_off"),

    "stretch":              ("action", "stretch"),
    "stretch out":          ("action", "stretch"),

    "push up":              ("action", "push_up"),
    "push ups":             ("action", "push_up"),
    "do push ups":          ("action", "push_up"),

    "shake head":           ("action", "shake_head"),
    "shake your head":      ("action", "shake_head"),
    "no":                   ("action", "shake_head"),

    "tilt head":            ("action", "tilting_head"),
    "tilting head":         ("action", "tilting_head"),
    "head tilt":            ("action", "tilting_head"),

    "wag tail":             ("action", "wag_tail"),
    "wag your tail":        ("action", "wag_tail"),
    "wag":                  ("action", "wag_tail"),

    # ── sounds ────────────────────────────────────────────────────────────────
    "angry":                ("sound", "angry"),
    "be angry":             ("sound", "angry"),
    "growl":                ("sound", "growl_1"),
    "growl 1":              ("sound", "growl_1"),
    "growl 2":              ("sound", "growl_2"),
    "growl one":            ("sound", "growl_1"),
    "growl two":            ("sound", "growl_2"),
    "confused":             ("sound", "confused_1"),
    "confused 1":           ("sound", "confused_1"),
    "confused 2":           ("sound", "confused_2"),
    "confused 3":           ("sound", "confused_3"),
    "howl":                 ("sound", "howling"),
    "howling":              ("sound", "howling"),
    "pant":                 ("sound", "pant"),
    "panting":              ("sound", "pant"),
    "bark":                 ("sound", "single_bark_1"),
    "single bark":          ("sound", "single_bark_1"),
    "single bark 1":        ("sound", "single_bark_1"),
    "single bark 2":        ("sound", "single_bark_2"),
    "snore":                ("sound", "snoring"),
    "snoring":              ("sound", "snoring"),
    "woohoo":               ("sound", "woohoo"),
    "woo hoo":              ("sound", "woohoo"),
    "yay":                  ("sound", "woohoo"),
    "celebrate":            ("sound", "woohoo"),
}

# Build a quick lookup from action name → (pitch, speed)
_ACTION_PARAMS: dict[str, tuple[int, int]] = {
    name: (pitch, speed) for name, pitch, speed in _ACTIONS
}


class PiDogActionModule:
    """
    Wraps a Pidog instance and executes actions / sounds from voice commands.

    Parameters
    ----------
    dog : Pidog
        Live Pidog() instance.  Pass None only for dry-run / unit-testing.
    default_duration : float
        How many seconds a movement action runs before stopping (when the dog
        method is a continuous loop rather than a one-shot animation).
    """

    def __init__(self, dog, default_duration: float = 1.5):
        self._dog = dog
        self._duration = default_duration
        self._lock = threading.Lock()          # one action at a time
        self._current_thread: threading.Thread | None = None
        log.info("PiDogActionModule ready")

    # ── Public API ────────────────────────────────────────────────────────────

    def execute(self, voice_text: str, blocking: bool = False) -> bool:
        """
        Resolve *voice_text* to an action or sound and run it.

        Returns True if a matching command was found, False otherwise.
        """
        key = voice_text.strip().lower()
        entry = COMMAND_MAP.get(key)

        if entry is None:
            log.info(f"No command mapping for: '{voice_text}'")
            print(f"[PiDogActions] Unknown command: '{voice_text}'")
            return False

        kind, name = entry
        if kind == "action":
            self._dispatch_action(name, blocking)
        else:
            self._dispatch_sound(name, blocking)
        return True

    def stop(self):
        """Ask the dog to stop whatever it's doing."""
        if self._dog is None:
            print("[PiDogActions] DRY-RUN stop()")
            return
        try:
            self._dog.stop_and_lie()
        except Exception as e:
            log.warning(f"stop() failed: {e}")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _dispatch_action(self, name: str, blocking: bool):
        t = threading.Thread(
            target=self._run_action,
            args=(name,),
            daemon=True,
            name=f"pidog-action-{name}",
        )
        if blocking:
            t.start()
            t.join()
        else:
            with self._lock:
                # Optionally interrupt previous non-blocking action here
                self._current_thread = t
            t.start()

    def _run_action(self, name: str):
        pitch, speed = _ACTION_PARAMS[name]
        print(f"[PiDogActions] ACTION  → {name}  (pitch={pitch}, speed={speed})")

        if self._dog is None:
            # Dry-run mode
            time.sleep(0.2)
            return

        # Adjust head pitch first (unless -1 which means "leave it")
        if pitch != -1:
            try:
                self._dog.head_move([[0, 0, pitch]], immediately=True, speed=60)
                self._dog.wait_head_done()
            except Exception as e:
                log.warning(f"head_move failed: {e}")

        # All PiDog actions go through do_action()
        try:
            self._dog.do_action(name, speed=speed)
            self._dog.wait_all_done()
        except Exception as e:
            log.error(f"Action '{name}' raised: {e}")

    def _dispatch_sound(self, name: str, blocking: bool):
        t = threading.Thread(
            target=self._run_sound,
            args=(name,),
            daemon=True,
            name=f"pidog-sound-{name}",
        )
        if blocking:
            t.start()
            t.join()
        else:
            t.start()

    def _run_sound(self, name: str):
        print(f"[PiDogActions] SOUND   → {name}")

        if self._dog is None:
            time.sleep(0.2)
            return

        try:
            # PiDog plays sounds via sound_effect_play(); fall back to speak()
            if hasattr(self._dog, "sound_effect_play"):
                self._dog.sound_effect_play(name)
            else:
                self._dog.speak(name)
        except Exception as e:
            log.error(f"Sound '{name}' raised: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test — no hardware needed
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    module = PiDogActionModule(dog=None)

    test_commands = [
        "sit down",
        "stand up",
        "wag your tail",
        "go left",
        "bark",
        "howl",
        "yay",
        "no",
        "sleep",
        "do push ups",
        "paws out",
        "unknown command",
    ]

    print("\n── PiDogActionModule dry-run ──────────────────────────────")
    for cmd in test_commands:
        found = module.execute(cmd, blocking=True)
        print(f"  '{cmd}' → {'✓ matched' if found else '✗ no match'}")
    print("── done ───────────────────────────────────────────────────\n")