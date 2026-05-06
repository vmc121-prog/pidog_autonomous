"""
Speech Module
=============
Two-tier TTS:
  - espeak-ng  : fast, offline, for urgent/navigation alerts  (priority 1-3)
  - Coqui TTS  : natural neural voice, for personality speech (priority 4-9)

Falls back gracefully if Coqui is not installed.
"""

import subprocess
import queue
import threading
import os
import sys
from modules.logging_config import setup_logging
import logging
import argparse, logging

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

setup_logging(level=logging.DEBUG if args.debug else logging.INFO)
# setup_logging()           # call once at startup

log = logging.getLogger("Speech")
log.info("Speech module starting")
COQUI_AVAILABLE = False
try:
    from TTS.api import TTS
    COQUI_AVAILABLE = True
except ImportError:
    pass


class SpeechModule:
    ESPEAK_THRESHOLD = 4   # priority <= this uses espeak-ng

    def __init__(self, rate=150, pitch=60, coqui_model="tts_models/en/ljspeech/tacotron2-DDC"):
        self._queue   = queue.PriorityQueue()
        self._thread  = None
        self._running = False
        self._rate    = rate
        self._pitch   = pitch
        self._counter = 0        # tie-break for equal priorities
        self._lock    = threading.Lock()

        self._coqui = None
        if COQUI_AVAILABLE:
            try:
                print("[Speech] Loading Coqui TTS model (first run downloads ~500MB)...")
                self._coqui = TTS(model_name=coqui_model, progress_bar=False)
                print("[Speech] Coqui TTS ready.")
            except Exception as e:
                print(f"[Speech] Coqui TTS load failed ({e}), falling back to espeak-ng only.")

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        # Unblock the worker
        self._queue.put((0, 0, None))

    def say(self, text: str, priority: int = 5):
        """
        Queue a speech utterance.
        priority 1 = most urgent (obstacle alerts)
        priority 9 = least urgent (idle chatter)
        """
        with self._lock:
            self._counter += 1
            counter = self._counter
        self._queue.put((priority, counter, text))

    def say_urgent(self, text: str):
        self.say(text, priority=1)

    def _worker(self):
        while self._running:
            try:
                priority, _, text = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if text is None:
                break

            self._speak(text, priority)

    def _speak(self, text: str, priority: int):
        if priority <= self.ESPEAK_THRESHOLD or self._coqui is None:
            self._espeak(text)
        else:
            try:
                self._coqui_speak(text)
            except Exception as e:
                print(f"[Speech] Coqui error: {e}, falling back to espeak")
                self._espeak(text)

    def _espeak(self, text: str):
        try:
            subprocess.run(
                ["espeak-ng", "-s", str(self._rate), "-p", str(self._pitch), text],
                timeout=10,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            print(f"[Speech] espeak-ng not found. Text: {text}")
        except Exception as e:
            print(f"[Speech] espeak-ng error: {e}")

    def _coqui_speak(self, text: str):
        wav_path = "/tmp/pidog_speech.wav"
        self._coqui.tts_to_file(text=text, file_path=wav_path)
        subprocess.run(["aplay", "-q", wav_path], timeout=15)
