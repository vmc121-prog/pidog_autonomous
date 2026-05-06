# PiDog Autonomous — Installation Guide

## What's included

```
pidog_autonomous/
├── main.py              ← Run this to start everything
├── register_face.py     ← Register known faces
├── requirements.txt
├── modules/
│   ├── obstacle.py      ← Obstacle avoidance (priority 1)
│   ├── follow.py        ← Person / object follow (priority 2)
│   ├── voice.py         ← Voice commands + clap detection (priority 3)
│   ├── emotion.py       ← Mood state machine (priority 4)
│   ├── mission.py       ← Patrol routes (priority 5)
│   ├── speech.py        ← Text-to-speech (espeak-ng + optional Coqui)
│   ├── vision.py        ← Camera, detection, face recognition
│   └── mock_pidog.py    ← Simulated hardware for testing on a PC
├── models/              ← Place downloaded models here
├── faces/               ← Auto-created, stores registered face data
└── sounds/              ← Optional custom sound files
```

---

## Step 1 — Flash and set up Raspberry Pi

Use **Raspberry Pi OS (64-bit, Bullseye or Bookworm)**.
Enable camera, SSH, and I2C in `raspi-config`.

```bash
sudo raspi-config
# Interface Options → Camera → Enable
# Interface Options → I2C   → Enable
# Interface Options → SSH   → Enable
```

---

## Step 2 — Install the SunFounder PiDog library

Follow SunFounder's official instructions:
https://docs.sunfounder.com/projects/pidog/en/latest/

```bash
cd ~
git clone https://github.com/sunfounder/pidog.git
cd pidog
sudo python3 install.py
```

Test it works:
```bash
cd ~/pidog/examples
python3 simple_demo.py
```

---

## Step 3 — Install system packages

```bash
sudo apt update && sudo apt install -y \
    espeak-ng \
    portaudio19-dev \
    python3-opencv \
    cmake \
    libboost-all-dev \
    libatlas-base-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    libopenblas-dev \
    gfortran
```

---

## Step 4 — Copy project files

```bash
# From your computer:
scp pidog_autonomous.zip pi@<PI_IP>:~
# Then on the Pi:
unzip pidog_autonomous.zip
cd pidog_autonomous
```

Or clone/copy manually via USB stick.

---

## Step 5 — Install Python dependencies

```bash
cd ~/pidog_autonomous

# Core packages
pip install opencv-python-headless numpy ultralytics vosk pyaudio \
    --break-system-packages

# Face recognition (needs cmake — takes ~10 min to compile dlib)
pip install face-recognition --break-system-packages
```

### If pip install face-recognition fails:
```bash
pip install dlib --break-system-packages   # compile from source (~15 min)
pip install face-recognition --break-system-packages
```

---

## Step 6 — Download the Vosk speech model

```bash
cd ~/pidog_autonomous/models
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
mv vosk-model-small-en-us-0.15 vosk-model-small-en-us
```

---

## Step 7 — YOLOv8 model (auto-downloads on first run)

YOLOv8n (~6MB) downloads automatically the first time vision starts.
If you want to pre-download it:

```bash
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

For a lighter but less accurate alternative, download MobileNet-SSD TFLite:
```bash
cd ~/pidog_autonomous/models
wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
mv detect.tflite mobilenet_ssd.tflite
```
Then in `modules/vision.py` comment out the ultralytics import and uncomment tflite_runtime.

---

## Step 8 — Register known faces (optional)

```bash
# From a photo:
python3 register_face.py --name "Dave" --image ~/dave.jpg

# Live from camera:
python3 register_face.py --name "Dave" --capture
```

Run once per person. Data is stored in `faces/known_faces.pkl`.

---

## Step 9 — Test without hardware (PC / any machine)

```bash
python3 main.py
```

If the `pidog` library is not found it automatically switches to
`MockPidog` — all actions print to the terminal. Good for verifying
the logic before touching the hardware.

---

## Step 10 — Run on PiDog

```bash
python3 main.py
```

Press `Ctrl+C` to stop cleanly. PiDog will lie down before shutting off.

---

## Optional — Run on boot

```bash
crontab -e
# Add this line:
@reboot sleep 10 && cd /home/pi/pidog_autonomous && python3 main.py >> /home/pi/pidog.log 2>&1
```

---

## Optional — Coqui TTS (better voice quality)

```bash
pip install TTS --break-system-packages
```

Then in `modules/speech.py` uncomment the Coqui import.
First run downloads ~500MB of model weights.
Synthesis takes 1–3s per sentence — suitable for personality speech only.

---

## Voice commands

Say these out loud near PiDog's microphone:

| Say           | Action                          |
|---------------|---------------------------------|
| "sit"         | Sit down                        |
| "stand"       | Stand up                        |
| "come"        | Walk toward you                 |
| "stay"        | Stop and hold                   |
| "spin"        | Turn left 360°                  |
| "shake"       | Wave a paw                      |
| "patrol"      | Start the patrol route          |

**Double-clap** → sit down (works even without Vosk)

---

## Customising the patrol route

Edit `modules/mission.py` → `DEFAULT_PATROL`:

```python
DEFAULT_PATROL = [
    ("forward",    3.0),   # walk forward 3 seconds
    ("turn_right", 1.5),   # turn right
    ("forward",    2.0),
    ("turn_left",  1.0),
    # add as many steps as you like
]
```

Available actions match PiDog's `do_action()` strings:
`forward`, `backward`, `turn_left`, `turn_right`, `stand`,
`sit_down`, `lie_down`, `wag_tail`, `wave_hand`, `look_around`

---

## Tuning tips

- **Obstacle threshold**: edit `DANGER_CM` in `modules/obstacle.py`
- **Follow distance**: edit `TARGET_SIZE_FAR / TARGET_SIZE_NEAR` in `modules/follow.py`
- **Mood speech frequency**: edit `IDLE_SPEECH_INTERVAL` in `modules/emotion.py`
- **Clap sensitivity**: edit `CLAP_THRESHOLD` in `modules/voice.py` (lower = more sensitive)
- **Inference speed**: edit `INFERENCE_EVERY_N` in `modules/vision.py` (higher = fewer frames processed)

---

## Troubleshooting

**"No module named pidog"** → SunFounder library not installed. See Step 2. MockPidog will be used automatically.

**Camera not opening** → Check `raspi-config` camera is enabled. Try `libcamera-hello` to verify.

**espeak-ng not found** → `sudo apt install espeak-ng`

**Vosk model not found** → Check path: `models/vosk-model-small-en-us/` must exist (see Step 6)

**face-recognition compile fails** → Install cmake + build-essential first, then retry dlib.

**YOLOv8 too slow** → Switch to MobileNet-SSD TFLite (see Step 7), or increase `INFERENCE_EVERY_N`.
