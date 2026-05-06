import threading
import time
import queue
import json
import audioop

from modules import BaseModule

from vosk import Model, KaldiRecognizer
import pyaudio
from modules.logging_config import setup_logging
import logging
import argparse, logging

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

setup_logging(level=logging.DEBUG if args.debug else logging.INFO)
# setup_logging()           # call once at startup

log = logging.getLogger("Voice")

log.info("Voice module starting")


VOSK_MODEL_PATH = "models/vosk-model-small-en-us"

SAMPLE_RATE  = 16000   # Vosk requires 16 kHz
CAPTURE_RATE = 48000   # Google Voice HAT only supports 48000 Hz
CHUNK        = 2048

CLAP_THRESHOLD = 3000

COMMANDS = {
    "hello":   (None,         "Hello there!",       5),
    "sit":     ("sit",        "Sitting down.",       4),
    "stand":   ("stand",      "Standing up.",        5),
    "come":    ("forward",    "Coming over.",        50),
    "stay":    ("stop",       "Staying put.",        4),
    "spin":    ("turn_left",  "Spinning!",           7),
    "shake":   ("wave_hand",  "Shake on it.",        7),
    "patrol":  ("patrol",     "Starting patrol.",    6),
    "stop":    ("stop",       "Stopping.",           4),
}


class VoiceModule(BaseModule):

    def __init__(self, dog, speech, priority=3):
        super().__init__(dog, speech, priority)

        self._command_queue = queue.Queue()
        self._clap_queue    = queue.Queue()

        self._pa     = None
        self._stream = None

        self._vosk_rec = None

        self._running_audio = False
        self._audio_thread  = None

        self._clap_window    = 1.5
        self._last_clap      = 0
        self._resample_state = None   # audioop resampler state

    # -------------------------
    # DEVICE SELECTION
    # -------------------------
    def _find_input_device(self):
        for i in range(self._pa.get_device_count()):
            dev = self._pa.get_device_info_by_index(i)
            if dev.get("maxInputChannels", 0) > 0:
                print(f"[Voice] Mic device {i}: {dev['name']}")
                return i
        return None

    # -------------------------
    # START
    # -------------------------
    def start(self):
        super().start()

        self._pa = pyaudio.PyAudio()

        device_index = self._find_input_device()
        if device_index is None:
            print("[Voice] No mic found")
            return

        # FIX 1: open at 48000 Hz - the only rate the HAT supports
        try:
            self._stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=CAPTURE_RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=CHUNK,
            )
        except Exception as e:
            print(f"[Voice] Mic open failed: {e}")
            return

        # FIX 2: restrict grammar to command words only - much better accuracy
        model = Model(VOSK_MODEL_PATH)
        self._vosk_rec = KaldiRecognizer(model, SAMPLE_RATE)
        grammar = json.dumps(list(COMMANDS.keys()) + ["[unk]"])
        self._vosk_rec.SetGrammar(grammar)
        log.info(f"[Voice] Grammar restricted to: {list(COMMANDS.keys())}")

        self._running_audio = True
        self._audio_thread  = threading.Thread(
            target=self._audio_loop,
            daemon=True,
        )
        self._audio_thread.start()

        print("[Voice] Unified audio engine started")

    # -------------------------
    # STOP
    # -------------------------
    def stop(self):
        super().stop()
        self._running_audio = False

        if self._stream:
            self._stream.stop_stream()
            self._stream.close()

        if self._pa:
            self._pa.terminate()

    # -------------------------
    # ARBITRATION HOOK
    # -------------------------
    def should_act(self, sensor_data: dict) -> bool:
        return (
            not self._command_queue.empty()
            or not self._clap_queue.empty()
        )

    def act(self, sensor_data: dict):
        # Clap detection
        while not self._clap_queue.empty():
            self._clap_queue.get()
            self.speech.say("Double clap detected. Sitting.", priority=3)
            self.dog.do_action("sit", speed=60)

        # Voice commands
        while not self._command_queue.empty():
            cmd = self._command_queue.get()
            if cmd in COMMANDS:
                action, phrase, pri = COMMANDS[cmd]
                self.speech.say(phrase, priority=pri)
                if str(action) == "forward":
                    self.dog.do_action("forward", speed=70)
                    time.sleep(3.0)
                    self.dog.do_action("stop")
                elif action:
                    self.dog.do_action(action, speed=60)

    # -------------------------
    # SINGLE AUDIO PIPELINE
    # -------------------------
    def _audio_loop(self):
        print("[Voice] Audio loop running")

        while self._running_audio:
            try:
                raw = self._stream.read(CHUNK, exception_on_overflow=False)
            except Exception:
                continue

            # Clap detection on raw audio
            rms = audioop.rms(raw, 2)
            now = time.time()
            if rms > CLAP_THRESHOLD:
                if now - self._last_clap < self._clap_window:
                    self._clap_queue.put("double_clap")
                    self._last_clap = 0
                else:
                    self._last_clap = now

            # FIX 3: resample 48000 -> 16000 before passing to Vosk
            resampled, self._resample_state = audioop.ratecv(
                raw, 2, 1,
                CAPTURE_RATE, SAMPLE_RATE,
                self._resample_state,
            )

            if self._vosk_rec.AcceptWaveform(resampled):
                result = json.loads(self._vosk_rec.Result())
                text   = result.get("text", "").strip().lower()

                if text and text != "[unk]":
                    print(f"[Voice] Recognised: {text}")
                    for keyword in COMMANDS:
                        if keyword in text:
                            print(f"[Voice] Command: {text} -> {keyword}")
                            self._command_queue.put(keyword)
                            break

