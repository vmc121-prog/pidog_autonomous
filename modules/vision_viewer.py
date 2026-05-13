"""
Vision Viewer Server
====================
Streams the camera feed with detection overlays to a browser.
Captures print() output and shows it as a live log panel.
Supports voice commands via Web Speech API → /command endpoint.

Usage:
    python vision_viewer.py [--port 5050] [--debug]

Then open http://<robot-ip>:5050 on any machine on your network.

Voice commands require Chrome or Edge. For network access (non-localhost),
use an SSH tunnel so the page is served from localhost:
    ssh -L 5050:localhost:5050 pi@<robot-ip>
Then open http://localhost:5050 in Chrome.

Consuming voice commands in your main script
--------------------------------------------
    from modules.vision_viewer import start_viewer, command_queue

    def handle_commands(dog):
        while True:
            try:
                cmd = command_queue.get(timeout=0.5)
                if "sit"     in cmd: dog.sit()
                elif "stand" in cmd: dog.stand()
                elif "forward" in cmd: dog.forward(...)
                # etc.
            except queue.Empty:
                pass

    threading.Thread(target=handle_commands, args=(dog,), daemon=True).start()
"""

import queue
import threading
import time
import sys
import collections
import argparse
import logging

logging.getLogger("werkzeug").setLevel(logging.WARNING)

import cv2
import numpy as np
from flask import Flask, Response, render_template_string, jsonify, request

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--port",  type=int, default=5050)
parser.add_argument("--debug", action="store_true")
args, _ = parser.parse_known_args()

# ── Logging ───────────────────────────────────────────────────────────────────
log = logging.getLogger("VisionViewer")

# ── Command queue — consume this in your main PiDog script ───────────────────
command_queue = queue.Queue()


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


# ── Render loop ───────────────────────────────────────────────────────────────
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
            frame = vision_module.get_frame()
            if frame is not None:
                result = vision_module.get_latest()
                drawn  = draw_detections(frame.copy(), result)
                frame_buffer.set(frame_to_jpeg(drawn))
        except Exception as e:
            log.warning(f"Render loop error: {e}")

        time.sleep(max(0, interval - (time.time() - t0)))


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

HTML = r"""
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
    --danger:   #f55;
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
    grid-template-rows: 48px 1fr 90px;
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

  /* ── Feed panel ── */
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

  /* ── Log panel ── */
  .log-panel {
    background: var(--panel);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    grid-row: 2 / 4;
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
  .log-line.error { color: var(--danger); }
  .log-line.cmd   { border-left-color: var(--accent2); color: var(--accent2); }

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

  /* ── Voice command bar ── */
  .voice-bar {
    grid-column: 1 / 2;
    background: var(--panel);
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 0 20px;
    border-top: 1px solid var(--border);
    position: relative;
    overflow: hidden;
  }

  /* animated listening background glow */
  .voice-bar::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at center, rgba(0,229,160,0.08) 0%, transparent 70%);
    opacity: 0;
    transition: opacity 0.4s ease;
    pointer-events: none;
  }
  .voice-bar.listening::before { opacity: 1; }

  .mic-btn {
    flex-shrink: 0;
    width: 48px; height: 48px;
    border-radius: 50%;
    border: 2px solid var(--accent);
    background: transparent;
    color: var(--accent);
    font-size: 20px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: background 0.2s, box-shadow 0.2s, border-color 0.2s;
    position: relative;
    z-index: 1;
  }

  .mic-btn:hover {
    background: rgba(0,229,160,0.1);
    box-shadow: 0 0 12px rgba(0,229,160,0.3);
  }

  .mic-btn.listening {
    background: var(--accent);
    color: #000;
    border-color: var(--accent);
    box-shadow: 0 0 24px var(--accent), 0 0 48px rgba(0,229,160,0.3);
    animation: mic-pulse 1s ease-in-out infinite;
  }

  .mic-btn.error {
    border-color: var(--danger);
    color: var(--danger);
  }

  @keyframes mic-pulse {
    0%, 100% { box-shadow: 0 0 24px var(--accent), 0 0 48px rgba(0,229,160,0.3); }
    50%       { box-shadow: 0 0 8px var(--accent); }
  }

  /* waveform bars — shown while listening */
  .waveform {
    display: flex;
    align-items: center;
    gap: 3px;
    height: 32px;
    opacity: 0;
    transition: opacity 0.3s;
  }
  .waveform.active { opacity: 1; }
  .waveform span {
    display: block;
    width: 3px;
    border-radius: 2px;
    background: var(--accent);
    height: 8px;
    animation: wave 0.8s ease-in-out infinite;
  }
  .waveform span:nth-child(1) { animation-delay: 0.0s; }
  .waveform span:nth-child(2) { animation-delay: 0.1s; }
  .waveform span:nth-child(3) { animation-delay: 0.2s; }
  .waveform span:nth-child(4) { animation-delay: 0.3s; }
  .waveform span:nth-child(5) { animation-delay: 0.15s; }
  .waveform span:nth-child(6) { animation-delay: 0.05s; }

  @keyframes wave {
    0%, 100% { height: 6px;  opacity: 0.5; }
    50%       { height: 28px; opacity: 1;   }
  }

  .voice-status {
    font-family: var(--mono);
    font-size: 12px;
    color: var(--dim);
    letter-spacing: 1px;
    flex: 1;
    transition: color 0.2s;
  }
  .voice-status.listening { color: var(--accent); }
  .voice-status.error     { color: var(--danger); }

  /* last-command chip */
  .last-cmd {
    font-family: var(--mono);
    font-size: 11px;
    padding: 4px 10px;
    border-radius: 20px;
    border: 1px solid var(--border);
    color: var(--dim);
    max-width: 200px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    transition: all 0.3s;
  }
  .last-cmd.flash {
    border-color: var(--accent2);
    color: var(--accent2);
    background: rgba(0,170,255,0.08);
  }

  /* no-speech-api warning */
  .no-api-warn {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--warn);
    letter-spacing: 1px;
    padding: 0 20px;
    text-align: center;
    flex: 1;
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

<!-- Voice command bar -->
<div class="voice-bar" id="voiceBar">
  <button class="mic-btn" id="micBtn" onclick="toggleVoice()" title="Hold to speak a command">
    🎤
  </button>
  <div class="waveform" id="waveform">
    <span></span><span></span><span></span>
    <span></span><span></span><span></span>
  </div>
  <div class="voice-status" id="voiceStatus">VOICE READY — CLICK MIC</div>
  <div class="last-cmd" id="lastCmd">no command yet</div>
</div>

<script>
  // ── Feed ──────────────────────────────────────────────────────────────────
  const feed   = document.getElementById('feed');
  const status = document.getElementById('status');
  feed.onload  = () => { status.textContent = 'LIVE'; status.style.color = 'var(--accent)'; };
  feed.onerror = () => { status.textContent = 'NO SIGNAL'; status.style.color = '#f55'; };

  // ── Log polling ───────────────────────────────────────────────────────────
  const logLines = document.getElementById('log-lines');
  const logCount = document.getElementById('log-count');
  const logTime  = document.getElementById('log-time');
  let lastLineCount = 0;
  let autoScroll    = true;

  logLines.addEventListener('scroll', () => {
    autoScroll = logLines.scrollHeight - logLines.scrollTop - logLines.clientHeight < 30;
  });

  async function fetchLogs() {
    try {
      const r    = await fetch('/logs');
      const data = await r.json();
      const lines = data.lines;

      if (lines.length !== lastLineCount) {
        const frag = document.createDocumentFragment();
        lines.forEach((line, i) => {
          const div    = document.createElement('div');
          const isNew  = i >= lastLineCount;
          const isWarn = line.toLowerCase().includes('warn');
          const isErr  = line.toLowerCase().includes('error') || line.toLowerCase().includes('failed');
          const isCmd  = line.includes('[VoiceCmd]');

          div.className = 'log-line'
            + (isNew  ? ' new'   : '')
            + (isWarn ? ' warn'  : '')
            + (isErr  ? ' error' : '')
            + (isCmd  ? ' cmd'   : '');

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

  // ── Voice commands ────────────────────────────────────────────────────────
  const voiceBar    = document.getElementById('voiceBar');
  const micBtn      = document.getElementById('micBtn');
  const waveform    = document.getElementById('waveform');
  const voiceStatus = document.getElementById('voiceStatus');
  const lastCmdEl   = document.getElementById('lastCmd');

  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
  let recog     = null;
  let listening = false;

  if (!SpeechRec) {
    // Replace bar contents with a warning — browser doesn't support API
    voiceBar.innerHTML = `
      <div class="no-api-warn">
        ⚠ Web Speech API not available — use Chrome or Edge.
        For network access use an SSH tunnel: <br>
        <code>ssh -L 5050:localhost:5050 pi@&lt;robot-ip&gt;</code>
        then open <code>http://localhost:5050</code>
      </div>`;
  }

  function setListeningUI(on) {
    listening = on;
    micBtn.className    = 'mic-btn' + (on ? ' listening' : '');
    voiceBar.className  = 'voice-bar' + (on ? ' listening' : '');
    waveform.className  = 'waveform' + (on ? ' active' : '');
    voiceStatus.className = 'voice-status' + (on ? ' listening' : '');
    voiceStatus.textContent = on ? 'LISTENING...' : 'VOICE READY — CLICK MIC';
    micBtn.textContent = on ? '⏹' : '🎤';
  }

  function setErrorUI(msg) {
    micBtn.className      = 'mic-btn error';
    voiceStatus.className = 'voice-status error';
    voiceStatus.textContent = msg;
    waveform.className    = 'waveform';
    setTimeout(() => {
      micBtn.className      = 'mic-btn';
      voiceStatus.className = 'voice-status';
      voiceStatus.textContent = 'VOICE READY — CLICK MIC';
    }, 3000);
  }

  async function sendCommand(text) {
    try {
      const r = await fetch('/command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      const data = await r.json();
      lastCmdEl.textContent = '"' + text + '"';
      lastCmdEl.className   = 'last-cmd flash';
      setTimeout(() => { lastCmdEl.className = 'last-cmd'; }, 2000);
    } catch(e) {
      setErrorUI('SEND FAILED');
    }
  }

  function startListening() {
    if (!SpeechRec) return;
    recog = new SpeechRec();
    recog.continuous      = false;
    recog.interimResults  = false;
    recog.lang            = 'en-GB';

    recog.onstart  = () => setListeningUI(true);
    recog.onend    = () => setListeningUI(false);
    recog.onerror  = (e) => {
      setListeningUI(false);
      if (e.error === 'not-allowed') {
        setErrorUI('MIC ACCESS DENIED');
      } else if (e.error === 'no-speech') {
        setErrorUI('NO SPEECH DETECTED');
      } else {
        setErrorUI('ERROR: ' + e.error.toUpperCase());
      }
    };

    recog.onresult = (e) => {
      const text = e.results[0][0].transcript.trim().toLowerCase();
      if (text) sendCommand(text);
    };

    try {
      recog.start();
    } catch(e) {
      setErrorUI('COULD NOT START MIC');
    }
  }

  function stopListening() {
    recog?.stop();
    setListeningUI(false);
  }

  function toggleVoice() {
    if (!SpeechRec) return;
    listening ? stopListening() : startListening();
  }
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

@app.route("/command", methods=["POST"])
def command():
    """
    Receives a voice command from the browser as JSON: {"text": "sit down"}
    Puts the text on command_queue for the main script to consume.
    Returns: {"status": "ok", "received": "<text>"}
    """
    log.info("Received command via /command endpoint")
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    log.info(f"Received command: "+ data =" / "+text)
    if text:
        print(f"[VoiceCmd] {text}")
        command_queue.put(text)
        return jsonify(status="ok", received=text)
    return jsonify(status="empty", error="No text received"), 400


# ── Entry point ───────────────────────────────────────────────────────────────
def start_viewer(vision_module, fps=15, port=None):
    """
    Call this from your main PiDog script AFTER creating your VisionModule.

        from modules.vision_viewer import start_viewer, command_queue
        vm = VisionModule(camera_index=0)
        vm.start()
        threading.Thread(target=start_viewer, args=(vm,), daemon=True).start()

    Voice commands arrive on command_queue as lowercase strings.
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
    print(f"[VisionViewer] Voice commands require Chrome/Edge (or SSH tunnel for HTTPS)")
    app.run(host="0.0.0.0", port=p, debug=False, threaded=True)


if __name__ == "__main__":
    # Standalone test — no camera needed, checks UI + voice button are working
    print("[VisionViewer] Standalone test mode")

    class FakeResult:
        detections = []

    class FakeVision:
        def get_latest(self): return FakeResult()
        def get_frame(self):  return None

    # Print queued commands so you can verify the endpoint works
    def _print_commands():
        while True:
            try:
                cmd = command_queue.get(timeout=1)
                print(f"[TEST] Command received: {cmd}")
            except queue.Empty:
                pass

    threading.Thread(target=_print_commands, daemon=True).start()
    start_viewer(FakeVision(), port=args.port)