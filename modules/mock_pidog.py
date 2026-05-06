"""
Mock PiDog — for testing the autonomy stack without hardware.
Prints all actions to the terminal.
"""

import time
import random
from modules.logging_config import setup_logging
import logging
import argparse, logging

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

setup_logging(level=logging.DEBUG if args.debug else logging.INFO)
# setup_logging()           # call once at startup

log = logging.getLogger("Main")
log.info("MockPidog module starting")

class MockUltrasonic:
    def get_distance(self):
        # Simulate occasional obstacles
        return random.choice([80, 80, 80, 80, 25, 80, 80])


class MockPidog:
    def __init__(self):
        self.ultrasonic = MockUltrasonic()
        print("[MockPidog] Running in simulation mode.")

    def do_action(self, action, speed=50):
        print(f"[MockPidog] Action: {action} (speed={speed})")

    def head_move(self, angles, immediately=False, speed=80):
        print(f"[MockPidog] Head move: {angles}")

    def set_rgb_led(self, r, g, b):
        print(f"[MockPidog] LED: rgb({r},{g},{b})")

    def set_face(self, expression):
        print(f"[MockPidog] Face: {expression}")

    def get_battery_percentage(self):
        return 85

    def is_touched(self):
        return False

    def close(self):
        print("[MockPidog] Shutdown.")
