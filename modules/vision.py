"""
Vision Module
=============
Runs in a background thread.
  - Frame capture via picamera2 (Pi 5 native) with OpenCV fallback
  - Object detection: YOLOv8n (preferred) or MobileNet-SSD TFLite fallback
  - Face recognition: face_recognition library (dlib-based)

get_latest() returns the most recent VisionResult to the main loop.
get_frame()  returns the most recent raw BGR frame (used by vision_viewer).
"""

import threading
import time
import os
import pickle
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import cv2
import numpy as np
import logging

# Logging is configured by main.py before this module is imported.
log = logging.getLogger("Vision")

# ── Picamera2 (Pi 5 native camera stack) ─────────────────────────────────────
PICAMERA2_AVAILABLE = False
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    pass

# ── Optional detection/recognition imports ────────────────────────────────────
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    pass

FACE_REC_AVAILABLE = False
try:
    import face_recognition
    FACE_REC_AVAILABLE = True
except ImportError:
    pass

TFLITE_AVAILABLE = False
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    try:
        import tensorflow.lite as tflite
        TFLITE_AVAILABLE = True
    except ImportError:
        pass


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2
    is_person: bool = False
    known_name: Optional[str] = None   # set if face recognised


@dataclass
class VisionResult:
    detections: List[Detection] = field(default_factory=list)
    frame_w: int = 640
    frame_h: int = 480
    timestamp: float = 0.0

    @property
    def persons(self):
        return [d for d in self.detections if d.is_person]

    @property
    def known_persons(self):
        return [d for d in self.detections if d.known_name]

    @property
    def primary_target(self) -> Optional[Detection]:
        """Largest bounding box detection."""
        if not self.detections:
            return None
        return max(self.detections, key=lambda d: (d.bbox[2]-d.bbox[0])*(d.bbox[3]-d.bbox[1]))


FACES_DB = "faces/known_faces.pkl"
INFERENCE_EVERY_N = 8   # run inference every N frames


class VisionModule:
    def __init__(self, camera_index=0, width=640, height=480, model_path=None):
        self._cam_idx  = camera_index
        self._width    = width
        self._height   = height
        self._thread   = None
        self._running  = False
        self._lock     = threading.Lock()
        self._latest   = VisionResult(timestamp=time.time())
        self._latest_frame = None   # raw BGR frame for vision_viewer
        self._frame_n  = 0

        self._detector = self._init_detector(model_path)

        self._known_encodings = []
        self._known_names     = []
        self._load_faces()

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info("Vision started.")

    def stop(self):
        self._running = False

    def get_latest(self) -> VisionResult:
        """Returns the most recent detection result."""
        with self._lock:
            return self._latest

    def get_frame(self):
        """Returns the most recent raw BGR frame, or None if not yet available.
        Used by vision_viewer so it doesn't need to open the camera itself."""
        with self._lock:
            return self._latest_frame

    def register_face(self, name: str, image_path: str):
        """Add a known face from an image file."""
        img = face_recognition.load_image_file(image_path)
        encs = face_recognition.face_encodings(img)
        if not encs:
            log.warning(f"No face found in {image_path}")
            return
        self._known_encodings.append(encs[0])
        self._known_names.append(name)
        self._save_faces()
        log.info(f"Registered face: {name}")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _loop(self):
        if PICAMERA2_AVAILABLE:
            self._loop_picamera2()
        else:
            self._loop_opencv()

    def _loop_picamera2(self):
        """Capture loop using picamera2 — for Pi 5 / libcamera stack."""
        log.info("Using picamera2 for capture")
        cam = Picamera2()
        config = cam.create_video_configuration(
            main={"size": (self._width, self._height), "format": "RGB888"}
        )
        cam.configure(config)
        cam.start()
        log.info(f"picamera2 started at {self._width}x{self._height}")

        try:
            while self._running:
                # Capture returns RGB — convert to BGR for OpenCV
                rgb_frame = cam.capture_array()
                frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

                # Always store latest raw frame for vision_viewer
                with self._lock:
                    self._latest_frame = frame.copy()

                self._frame_n += 1
                if self._frame_n % INFERENCE_EVERY_N != 0:
                    time.sleep(0.01)
                    continue

                detections = []
                if self._detector:
                    detections = self._detect(frame)

                if FACE_REC_AVAILABLE and self._known_encodings:
                    detections = self._recognise_faces(frame, detections)

                result = VisionResult(
                    detections=detections,
                    frame_w=self._width,
                    frame_h=self._height,
                    timestamp=time.time(),
                )
                with self._lock:
                    self._latest = result

        finally:
            cam.stop()
            log.info("picamera2 stopped.")

    def _loop_opencv(self):
        """Fallback capture loop using OpenCV VideoCapture."""
        log.info("Using OpenCV VideoCapture for capture (picamera2 not available)")
        cap = cv2.VideoCapture(self._cam_idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

        if not cap.isOpened():
            log.error("Could not open camera.")
            return

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                with self._lock:
                    self._latest_frame = frame.copy()

                self._frame_n += 1
                if self._frame_n % INFERENCE_EVERY_N != 0:
                    time.sleep(0.01)
                    continue

                detections = []
                if self._detector:
                    detections = self._detect(frame)

                if FACE_REC_AVAILABLE and self._known_encodings:
                    detections = self._recognise_faces(frame, detections)

                result = VisionResult(
                    detections=detections,
                    frame_w=self._width,
                    frame_h=self._height,
                    timestamp=time.time(),
                )
                with self._lock:
                    self._latest = result
        finally:
            cap.release()

    def _init_detector(self, model_path):
        if YOLO_AVAILABLE:
            mp = model_path or "yolov8n.pt"
            log.info(f"Loading YOLOv8n ({mp})...")
            try:
                return ("yolo", YOLO(mp))
            except Exception as e:
                log.warning(f"YOLO load failed: {e}")

        if TFLITE_AVAILABLE:
            mp = model_path or "models/mobilenet_ssd.tflite"
            if os.path.exists(mp):
                log.info(f"Loading MobileNet-SSD TFLite ({mp})...")
                interp = tflite.Interpreter(model_path=mp)
                interp.allocate_tensors()
                return ("tflite", interp)

        log.warning("No detector available — vision running without detection.")
        return None

    def _detect(self, frame) -> List[Detection]:
        kind, model = self._detector
        detections  = []

        if kind == "yolo":
            results = model(frame, verbose=False, conf=0.45)[0]
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = results.names[int(box.cls[0])]
                conf  = float(box.conf[0])
                detections.append(Detection(
                    label=label,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    is_person=(label == "person"),
                ))

        elif kind == "tflite":
            detections = self._tflite_detect(frame, model)

        return detections

    def _tflite_detect(self, frame, interp) -> List[Detection]:
        in_details  = interp.get_input_details()
        out_details = interp.get_output_details()
        h, w = in_details[0]["shape"][1:3]
        resized = cv2.resize(frame, (w, h))
        inp     = np.expand_dims(resized, axis=0).astype(np.uint8)
        interp.set_tensor(in_details[0]["index"], inp)
        interp.invoke()

        boxes   = interp.get_tensor(out_details[0]["index"])[0]
        classes = interp.get_tensor(out_details[1]["index"])[0]
        scores  = interp.get_tensor(out_details[2]["index"])[0]

        COCO_LABELS = {0: "person", 15: "cat", 16: "dog", 32: "sports ball",
                       39: "bottle", 56: "chair", 57: "couch"}
        detections = []
        fh, fw = frame.shape[:2]
        for i in range(len(scores)):
            if scores[i] < 0.45:
                continue
            y1, x1, y2, x2 = boxes[i]
            label = COCO_LABELS.get(int(classes[i]), f"obj_{int(classes[i])}")
            detections.append(Detection(
                label=label,
                confidence=float(scores[i]),
                bbox=(int(x1*fw), int(y1*fh), int(x2*fw), int(y2*fh)),
                is_person=(label == "person"),
            ))
        return detections

    def _recognise_faces(self, frame, detections: List[Detection]) -> List[Detection]:
        small  = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locs   = face_recognition.face_locations(rgb)
        encs   = face_recognition.face_encodings(rgb, locs)

        for enc, loc in zip(encs, locs):
            top, right, bottom, left = [v*2 for v in loc]
            matches   = face_recognition.compare_faces(self._known_encodings, enc, tolerance=0.5)
            distances = face_recognition.face_distance(self._known_encodings, enc)
            name      = None
            if any(matches):
                best = int(np.argmin(distances))
                if matches[best]:
                    name = self._known_names[best]

            merged = False
            for d in detections:
                if d.is_person:
                    ox1, oy1, ox2, oy2 = d.bbox
                    cx = (left + right) // 2
                    cy = (top + bottom) // 2
                    if ox1 <= cx <= ox2 and oy1 <= cy <= oy2:
                        d.known_name = name
                        merged = True
                        break

            if not merged:
                detections.append(Detection(
                    label="person",
                    confidence=0.9,
                    bbox=(left, top, right, bottom),
                    is_person=True,
                    known_name=name,
                ))

        return detections

    def _load_faces(self):
        if os.path.exists(FACES_DB):
            with open(FACES_DB, "rb") as f:
                data = pickle.load(f)
                self._known_encodings = data.get("encodings", [])
                self._known_names     = data.get("names", [])
            log.info(f"Loaded {len(self._known_names)} known face(s): {self._known_names}")

    def _save_faces(self):
        os.makedirs("faces", exist_ok=True)
        with open(FACES_DB, "wb") as f:
            pickle.dump({"encodings": self._known_encodings, "names": self._known_names}, f)