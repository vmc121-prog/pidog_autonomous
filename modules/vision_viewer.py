"""
Vision Viewer Server
====================
Streams the camera feed with detection overlays to a browser.
Captures print() output and shows it as a live log panel.

Usage:
    python vision_viewer.py [--port 5050] [--debug]

Then open http://<robot-ip>:5050 on any machine on your network.
"""

import threading
import time
import sys
import collections
import argparse
import logging

logging.getLogger("werkzeug").setLevel(logging.WARNING)

import cv2
import numpy as np
from flask import Flask, Response, render_template_string, jsonify

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--port",  type=int, default=5050)
parser.add_argument("--debug", action="store_true")
args, _ = parser.parse_known_args()

# ── Logging ───────────────────────────────────────────────────────────────────
log = logging.getLogger("VisionViewer")


# ── Print interceptor — captures all print() calls ───────────────────────────
class PrintCapture:
    """Replaces sys.stdout so print() calls are captured AND still shown in terminal."""
    MAX_LINES = 200

    def __init__(self, original_stdout):
        self._orig  = original_stdout
        self._lock  = threading.Lock()
        self._lines = collections.deque(maxlen=self.MAX_LINES)

    def write(self, text):
        self._orig.write(text)
        stripped = text.strip()
        if stripped:
            ts = time.strftime("%H:%M:%S")
            with self._lock:
                self._lines.append(f"[{ts}] {stripped}")

    def flush(self):
        self._orig.flush()

    def get_lines(self):
        with self._lock:
            return list(self._lines)


print_capture = PrintCapture(sys.stdout)
sys.stdout = print_capture


# ── Frame buffer — shared between render thread and Flask ────────────────────
class FrameBuffer:
    def __init__(self):
        self._lock  = threading.Lock()
        self._frame = None  # JPEG bytes

    def set(self, jpeg_bytes):
        with self._lock:
            self._frame = jpeg_bytes

    def get(self):
        with self._lock:
            return self._frame


frame_buffer = FrameBuffer()


# ── Drawing helpers ───────────────────────────────────────────────────────────
COLOURS = {
    "person":  (0,   200, 100),
    "cat":     (255, 180,   0),
    "dog":     (0,   180, 255),
    "default": (180,  60, 255),
}

def colour_for(label: str):
    return COLOURS.get(label.lower(), COLOURS["default"])

def draw_detections(frame, vision_result):
    """Draw bounding boxes, labels and confidence onto frame."""
    overlay = frame.copy()

    for d in vision_result.detections:
        x1, y1, x2, y2 = d.bbox
        col = colour_for(d.label)

        cv2.rectangle(overlay, (x1, y1), (x2, y2), col, 2)

        name       = d.known_name or d.label
        conf       = f"{d.confidence:.0%}"
        label_text = f"{name}  {conf}"

        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness  = 1
        (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, thickness)

        pad = 4
        lx1, ly1 = x1, max(0, y1 - th - pad * 2)
        lx2, ly2 = x1 + tw + pad * 2, y1
        cv2.rectangle(overlay, (lx1, ly1), (lx2, ly2), col, -1)
        cv2.putText(overlay, label_text,
                    (lx1 + pad, ly2 - pad),
                    font, font_scale, (10, 10, 10), thickness, cv2.LINE_AA)

        # Extra highlight for recognised faces
        if d.known_name:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 200), 3)

    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # HUD
    cv2.putText(frame, time.strftime("%H:%M:%S"),
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{len(vision_result.detections)} detection(s)",
                (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    return frame


def frame_to_jpeg(frame) -> bytes:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes()


# ── Render loop — uses frames from VisionModule, no second camera open ────────
def vision_render_loop(vision_module, fps=15):
    """
    Pulls the latest raw frame and detection result from VisionModule,
    draws overlays, and pushes a JPEG into frame_buffer for Flask to serve.
    Never opens the camera itself — VisionModule owns the camera.
    """
    interval = 1.0 / fps
    log.info(f"Render loop started at {fps} fps")
    print(f"[VisionViewer] Render loop started at {fps} fps")

    while True:
        t0 = time.time()
        try:
            frame = vision_module.get_frame()   # raw BGR from VisionModule
            if frame is not None:
                result = vision_module.get_latest()
                drawn  = draw_detections(frame.copy(), result)
                frame_buffer.set(frame_to_jpeg(drawn))
        except Exception as e:
            log.warning(f"Render loop error: {e}")

        time.sleep(max(0, interval - (time.time() - t0)))


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PiDog Vision</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;600&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:       #0a0c0f;
    --panel:    #10141a;
    --border:   #1e2a38;
    --accent:   #00e5a0;
    --accent2:  #0af;
    --warn:     #f0b429;
    --text:     #c8d8e8;
    --dim:      #4a6070;
    --mono:     'Share Tech Mono', monospace;
    --sans:     'Exo 2', sans-serif;
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-weight: 300;
    height: 100vh;
    display: grid;
    grid-template-rows: 48px 1fr;
    grid-template-columns: 1fr 340px;
    gap: 1px;
    background-color: var(--border);
    overflow: hidden;
  }

  header {
    grid-column: 1 / -1;
    background: var(--panel);
    display: flex;
    align-items: center;
    padding: 0 20px;
    gap: 16px;
    border-bottom: 1px solid var(--border);
  }

  .logo {
    font-family: var(--mono);
    font-size: 15px;
    color: var(--accent);
    letter-spacing: 2px;
    text-transform: uppercase;
  }

  .dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 8px var(--accent);
    animation: pulse 2s ease-in-out infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
  }

  .status-bar {
    margin-left: auto;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--dim);
    letter-spacing: 1px;
  }

  .feed-panel {
    background: #000;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    overflow: hidden;
  }

  .feed-panel img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
    display: block;
  }

  .corner {
    position: absolute;
    width: 20px; height: 20px;
    border-color: var(--accent2);
    border-style: solid;
    opacity: 0.6;
  }
  .corner.tl { top: 12px; left: 12px;  border-width: 2px 0 0 2px; }
  .corner.tr { top: 12px; right: 12px; border-width: 2px 2px 0 0; }
  .corner.bl { bottom: 12px; left: 12px;  border-width: 0 0 2px 2px; }
  .corner.br { bottom: 12px; right: 12px; border-width: 0 2px 2px 0; }

  .scan-line {
    position: absolute;
    left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent2), transparent);
    opacity: 0.3;
    animation: scan 4s linear infinite;
  }
  @keyframes scan {
    0%   { top: 0%; }
    100% { top: 100%; }
  }

  .log-panel {
    background: var(--panel);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .log-header {
    padding: 10px 14px;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--accent);
    letter-spacing: 2px;
    border-bottom: 1px solid var(--border);
    text-transform: uppercase;
    flex-shrink: 0;
  }

  #log-lines {
    flex: 1;
    overflow-y: auto;
    padding: 8px 0;
    font-family: var(--mono);
    font-size: 11px;
    line-height: 1.7;
  }

  #log-lines::-webkit-scrollbar { width: 4px; }
  #log-lines::-webkit-scrollbar-track { background: transparent; }
  #log-lines::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

  .log-line {
    padding: 1px 14px;
    border-left: 2px solid transparent;
    color: var(--text);
    word-break: break-all;
    animation: fadeIn 0.2s ease;
  }
  @keyframes fadeIn { from { opacity: 0; transform: translateX(4px); } to { opacity: 1; } }

  .log-line.new   { border-left-color: var(--accent); color: #fff; }
  .log-line .ts   { color: var(--dim); margin-right: 6px; }
  .log-line.warn  { color: var(--warn); }
  .log-line.error { color: #f55; }

  .log-footer {
    padding: 6px 14px;
    font-family: var(--mono);
    font-size: 10px;
    color: var(--dim);
    border-top: 1px solid var(--border);
    flex-shrink: 0;
    display: flex;
    justify-content: space-between;
  }
</style>
</head>
<body>

<header>
  <div class="dot"></div>
  <div class="logo">PiDog // Vision</div>
  <div class="status-bar" id="status">CONNECTING...</div>
</header>

<div class="feed-panel">
  <div class="corner tl"></div>
  <div class="corner tr"></div>
  <div class="corner bl"></div>
  <div class="corner br"></div>
  <div class="scan-line"></div>
  <img id="feed" src="/stream" alt="Camera feed">
</div>

<div class="log-panel">
  <div class="log-header">// stdout log</div>
  <div id="log-lines"></div>
  <div class="log-footer">
    <span id="log-count">0 lines</span>
    <span id="log-time">--:--:--</span>
  </div>
</div>

<script>
  const feed = document.getElementById('feed');
  const status = document.getElementById('status');
  const logLines = document.getElementById('log-lines');
  const logCount = document.getElementById('log-count');
  const logTime  = document.getElementById('log-time');

  let lastLineCount = 0;
  let autoScroll = true;

  logLines.addEventListener('scroll', () => {
    const atBottom = logLines.scrollHeight - logLines.scrollTop - logLines.clientHeight < 30;
    autoScroll = atBottom;
  });

  feed.onload  = () => { status.textContent = 'LIVE'; status.style.color = 'var(--accent)'; };
  feed.onerror = () => { status.textContent = 'NO SIGNAL'; status.style.color = '#f55'; };

  async function fetchLogs() {
    try {
      const r = await fetch('/logs');
      const data = await r.json();
      const lines = data.lines;

      if (lines.length !== lastLineCount) {
        const frag = document.createDocumentFragment();
        lines.forEach((line, i) => {
          const div = document.createElement('div');
          const isNew   = i >= lastLineCount;
          const isWarn  = line.toLowerCase().includes('warn');
          const isError = line.toLowerCase().includes('error') || line.toLowerCase().includes('failed');

          div.className = 'log-line'
            + (isNew   ? ' new'   : '')
            + (isWarn  ? ' warn'  : '')
            + (isError ? ' error' : '');

          const match = line.match(/^(\[\d{2}:\d{2}:\d{2}\])\s(.*)$/);
          if (match) {
            div.innerHTML = `<span class="ts">${match[1]}</span>${match[2]}`;
          } else {
            div.textContent = line;
          }
          frag.appendChild(div);
        });

        logLines.innerHTML = '';
        logLines.appendChild(frag);
        lastLineCount = lines.length;
        logCount.textContent = lines.length + ' lines';
        if (autoScroll) logLines.scrollTop = logLines.scrollHeight;

        setTimeout(() => {
          document.querySelectorAll('.log-line.new').forEach(el => el.classList.remove('new'));
        }, 1500);
      }

      logTime.textContent = new Date().toLocaleTimeString();
    } catch(e) {
      logTime.textContent = 'ERROR';
    }
  }

  fetchLogs();
  setInterval(fetchLogs, 1000);
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/stream")
def stream():
    def generate():
        while True:
            jpeg = frame_buffer.get()
            if jpeg:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
            time.sleep(1 / 15)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/logs")
def logs():
    return jsonify(lines=print_capture.get_lines())


# ── Entry point ───────────────────────────────────────────────────────────────
def start_viewer(vision_module, fps=15, port=None):
    """
    Call this from your main PiDog script AFTER creating your VisionModule.

        from modules.vision_viewer import start_viewer
        vm = VisionModule(camera_index=0)
        vm.start()
        threading.Thread(target=start_viewer, args=(vm,), daemon=True).start()
    """
    p = port or args.port

    threading.Thread(
        target=vision_render_loop,
        args=(vision_module,),
        kwargs={"fps": fps},
        daemon=True,
    ).start()

    print(f"[VisionViewer] Web viewer at http://0.0.0.0:{p}")
    print(f"[VisionViewer] Open http://<robot-ip>:{p} in your browser")
    app.run(host="0.0.0.0", port=p, debug=False, threaded=True)


if __name__ == "__main__":
    # Standalone test — no camera needed, checks UI is working
    print("[VisionViewer] Standalone test mode")

    class FakeResult:
        detections = []

    class FakeVision:
        def get_latest(self): return FakeResult()
        def get_frame(self):  return None

    start_viewer(FakeVision(), port=args.port)