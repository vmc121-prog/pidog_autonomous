#!/usr/bin/env python3
"""
Register a known face for PiDog to recognise.

Usage:
  python register_face.py --name "Dave" --image path/to/photo.jpg

  # Or capture from camera live:
  python register_face.py --name "Dave" --capture
"""

import argparse
import sys
import os
from modules.logging_config import setup_logging
import logging
import argparse, logging

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

setup_logging(level=logging.DEBUG if args.debug else logging.INFO)
# setup_logging()           # call once at startup

log = logging.getLogger("register_face")
log.info("RegisterFace module starting")

sys.path.insert(0, os.path.dirname(__file__))


def main():
    parser = argparse.ArgumentParser(description="Register a face for PiDog recognition")
    parser.add_argument("--name",    required=True, help="Person's name")
    parser.add_argument("--image",   help="Path to a photo file")
    parser.add_argument("--capture", action="store_true", help="Capture from camera")
    args = parser.parse_args()

    from modules.vision import VisionModule
    vision = VisionModule.__new__(VisionModule)
    vision._known_encodings = []
    vision._known_names     = []
    vision._load_faces()

    if args.capture:
        import cv2
        print("Press SPACE to capture, ESC to cancel.")
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            cv2.imshow("Capture face — SPACE to save", frame)
            key = cv2.waitKey(1)
            if key == 32:   # SPACE
                path = f"/tmp/face_capture_{args.name}.jpg"
                cv2.imwrite(path, frame)
                cap.release()
                cv2.destroyAllWindows()
                vision.register_face(args.name, path)
                print(f"Registered '{args.name}' from camera capture.")
                break
            elif key == 27:  # ESC
                cap.release()
                cv2.destroyAllWindows()
                print("Cancelled.")
                break

    elif args.image:
        vision.register_face(args.name, args.image)
        print(f"Registered '{args.name}' from {args.image}.")

    else:
        print("Provide --image <path> or --capture")
        sys.exit(1)


if __name__ == "__main__":
    main()
