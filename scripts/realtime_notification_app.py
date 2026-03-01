"""
Realtime Notification Web App
=============================
Runs ElderWatch fall detection and serves a lightweight web UI with:
- Hero section and branding (ElderWatch)
- Mobile-app style notification panel
- Live video preview and realtime alert updates

Usage:
  python3 scripts/realtime_notification_app.py --source 0
  python3 scripts/realtime_notification_app.py --source data/set1/fall-01-cam0.mp4
Then open: http://127.0.0.1:8000
"""

import argparse
import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import yaml

from agent_loop import AlertLevel, ElderWatchAgent, FallEvent

logger = logging.getLogger("elderwatch.notify")


@dataclass
class NotificationEvent:
    confidence: float
    pose_summary: str
    timestamp: float


class GemmaNotifier:
    """Wraps local Gemma generation for short caregiver notifications."""

    def __init__(self, model_path: str, max_new_tokens: int = 48):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.enabled = False
        self.tokenizer = None
        self.model = None
        self._load()

    def _load(self):
        if not self.model_path or not os.path.exists(self.model_path):
            logger.warning("Gemma model path not found: %s", self.model_path)
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            logger.warning("transformers/torch not installed; using fallback text")
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
                local_files_only=True,
            ).to(device)
            self.model.eval()
            self.enabled = True
            logger.info("Loaded Gemma model from %s on %s", self.model_path, device)
        except Exception as exc:
            logger.warning("Failed to load Gemma model; using fallback text: %s", exc)
            self.enabled = False

    def generate(self, event: NotificationEvent) -> str:
        if not self.enabled:
            return (
                "Potential fall event detected. "
                "Please assess the patient immediately and initiate safety protocol."
            )

        prompt = (
            "You are a clinical safety assistant for a fall monitoring system.\\n"
            "Write one concise, professional caregiver notification.\\n"
            "Healthcare tone. No markdown. 24 words max.\\n"
            "Include immediate action guidance.\\n\\n"
            f"Fall confidence: {event.confidence:.2f}\\n"
            f"Pose summary: {event.pose_summary}\\n"
            "Notification:"
        )

        try:
            import torch

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            generated = text[len(prompt):].strip()
            if not generated:
                generated = (
                    "Potential fall event detected. "
                    "Please assess the patient immediately and initiate safety protocol."
                )
            return generated.splitlines()[0][:180]
        except Exception as exc:
            logger.warning("Gemma generation failed; using fallback text: %s", exc)
            return (
                "Potential fall event detected. "
                "Please assess the patient immediately and initiate safety protocol."
            )


class AppState:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_jpg = None
        self.last_frame_ts = 0.0
        self.current_prob = 0.0
        self.current_alert = AlertLevel.NONE.value
        self.notifications = [{
            "message": (
                "System initialized. Monitoring active. No fall events detected."
            ),
            "confidence": None,
            "ts": time.time(),
        }]
        self.notification_time = 0.0


def build_html() -> bytes:
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ElderWatch</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #f7f0e3;
      --bg-soft: #efe4d0;
      --card: #fff9ef;
      --ink: #141414;
      --muted: #47423a;
      --brand: #1a1a1a;
      --accent: #e9dcc5;
      --line: #d8c7a9;
      --shadow: 0 12px 30px rgba(60, 45, 20, 0.12);
      --radius: 22px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: 'Manrope', sans-serif;
      background: radial-gradient(1100px 420px at 10% -10%, #f1e2c6, transparent 65%), var(--bg);
      color: var(--ink);
    }
    .page {
      max-width: 1240px;
      margin: 0 auto;
      padding: 28px 20px 36px;
    }
    .hero {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 24px;
      align-items: flex-end;
    }
    .brand {
      font-weight: 800;
      font-size: clamp(2.1rem, 4vw, 2.9rem);
      letter-spacing: -0.04em;
      margin: 0;
      color: var(--brand);
    }
    .subtitle {
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 1rem;
    }
    .badge {
      background: var(--accent);
      color: var(--ink);
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 14px;
      font-weight: 700;
      font-size: .85rem;
      white-space: nowrap;
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 380px;
      gap: 20px;
      align-items: start;
    }
    .video-card {
      background: var(--card);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 14px;
      min-height: 440px;
      border: 1px solid var(--line);
    }
    .video-frame {
      width: 100%;
      border-radius: 16px;
      display: block;
      aspect-ratio: 16/9;
      object-fit: cover;
      background: #111;
    }
    .phone {
      background: #f4ead6;
      color: var(--ink);
      border-radius: 32px;
      padding: 14px;
      box-shadow: var(--shadow);
      border: 3px solid #cdb894;
      width: 100%;
      max-width: 380px;
      min-height: 640px;
    }
    .phone-screen {
      background: linear-gradient(180deg, #fff9ee 0%, #f7edd9 100%);
      border-radius: 24px;
      min-height: 604px;
      padding: 18px;
      position: relative;
      overflow: hidden;
      border: 1px solid #e3d1b1;
    }
    .notch {
      width: 120px;
      height: 26px;
      border-radius: 20px;
      background: #d6c4a2;
      margin: 0 auto 16px;
    }
    .panel-title {
      font-weight: 800;
      font-size: 1.08rem;
      letter-spacing: -0.01em;
    }
    .panel-muted {
      color: #534c42;
      font-size: .86rem;
      margin-top: 2px;
    }
    .stat {
      margin-top: 16px;
      background: #f2e5cf;
      border: 1px solid #d9c8a8;
      border-radius: 14px;
      padding: 12px;
    }
    .stat-row {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      margin: 6px 0;
      font-size: .9rem;
      color: #201d18;
    }
    .pill {
      border-radius: 999px;
      font-size: .78rem;
      font-weight: 700;
      padding: 5px 10px;
      background: #d8c6a5;
      color: #1a1a1a;
      text-transform: lowercase;
    }
    .pill.alert {
      background: #2a2a2a;
      color: #fff5ea;
    }
    .pill.none {
      background: #c5b08d;
      color: #141414;
    }
    .queue-title {
      margin-top: 16px;
      font-size: .9rem;
      font-weight: 700;
      color: #2f2a23;
    }
    .queue {
      list-style: none;
      margin: 10px 0 0;
      padding: 0;
      display: grid;
      gap: 10px;
      max-height: 330px;
      overflow: auto;
    }
    .notif-card {
      background: #fff6e8;
      border: 1px solid #d8c5a3;
      border-left: 4px solid #1e1e1e;
      border-radius: 12px;
      padding: 10px 10px;
    }
    .notif-meta {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      font-size: .74rem;
      color: #52493c;
      margin-bottom: 6px;
    }
    .notif-text {
      font-size: .88rem;
      line-height: 1.42;
      color: #121212;
      white-space: pre-wrap;
    }
    .hint {
      position: absolute;
      bottom: 16px;
      left: 18px;
      right: 18px;
      color: #5e5549;
      font-size: .78rem;
      text-align: center;
    }
    @media (max-width: 980px) {
      .layout { grid-template-columns: 1fr; }
      .phone { max-width: 100%; min-height: 560px; }
      .phone-screen { min-height: 530px; }
      .queue { max-height: 250px; }
    }
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div>
        <h1 class="brand">ElderWatch</h1>
        <p class="subtitle">Realtime fall monitoring with clinically styled caregiver notifications.</p>
      </div>
      <div class="badge">Live Agent Session</div>
    </section>

    <section class="layout">
      <div class="video-card">
        <img id="video" class="video-frame" alt="Live feed" src="/frame.jpg" />
      </div>

      <div class="phone">
        <div class="phone-screen">
          <div class="notch"></div>
          <div class="panel-title">ElderWatch</div>
          <div class="panel-muted">Caregiver Notification Queue</div>

          <div class="stat">
            <div class="stat-row"><span>LSTM p(fall)</span><strong id="prob">0.000</strong></div>
            <div class="stat-row"><span>Alert State</span><span id="alert" class="pill none">none</span></div>
          </div>

          <div class="queue-title">Notification Queue</div>
          <ul id="notif-list" class="queue"></ul>
          <div class="hint">Press Ctrl+C in terminal to stop server</div>
        </div>
      </div>
    </section>
  </div>

  <script>
    const videoEl = document.getElementById('video');
    const probEl = document.getElementById('prob');
    const alertEl = document.getElementById('alert');
    const notifListEl = document.getElementById('notif-list');

    function renderQueue(items) {
      notifListEl.innerHTML = '';
      if (!items || items.length === 0) {
        const li = document.createElement('li');
        li.className = 'notif-card';
        li.innerHTML = '<div class=\"notif-text\">Monitoring active. No event notifications yet.</div>';
        notifListEl.appendChild(li);
        return;
      }

      for (const item of items) {
        const li = document.createElement('li');
        li.className = 'notif-card';

        const meta = document.createElement('div');
        meta.className = 'notif-meta';
        const ts = new Date((item.ts || 0) * 1000).toLocaleTimeString();
        const conf = (item.confidence === null || item.confidence === undefined)
          ? 'info'
          : ('conf ' + Number(item.confidence).toFixed(2));
        meta.textContent = ts + ' • ' + conf;

        const txt = document.createElement('div');
        txt.className = 'notif-text';
        txt.textContent = item.message || '';

        li.appendChild(meta);
        li.appendChild(txt);
        notifListEl.appendChild(li);
      }
    }

    async function tick() {
      videoEl.src = '/frame.jpg?t=' + Date.now();

      try {
        const r = await fetch('/state?t=' + Date.now());
        if (r.ok) {
          const s = await r.json();
          probEl.textContent = Number(s.prob || 0).toFixed(3);
          const alertState = s.alert || 'none';
          alertEl.textContent = alertState;
          alertEl.className = 'pill ' + (alertState === 'none' ? 'none' : 'alert');
          renderQueue(s.notifications || []);
        }
      } catch (_) {
        // Keep previous state on transient fetch errors.
      }
    }

    setInterval(tick, 200);
    tick();
  </script>
</body>
</html>
"""
    return html.encode("utf-8")


def make_handler(app_state: AppState):
    html_blob = build_html()

    class Handler(BaseHTTPRequestHandler):
        def _send_bytes(self, code: int, content_type: str, payload: bytes):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path.startswith("/state"):
                with app_state.lock:
                    data = {
                        "prob": app_state.current_prob,
                        "alert": app_state.current_alert,
                        "notifications": app_state.notifications,
                        "frame_ts": app_state.last_frame_ts,
                    }
                payload = json.dumps(data).encode("utf-8")
                self._send_bytes(HTTPStatus.OK, "application/json", payload)
                return

            if self.path.startswith("/frame.jpg"):
                with app_state.lock:
                    jpg = app_state.latest_jpg
                if jpg is None:
                    blank = 255 * (cv2.UMat(360, 640, cv2.CV_8UC3).get() * 0)
                    cv2.putText(blank, "Waiting for video frames...", (120, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2)
                    ok, enc = cv2.imencode(".jpg", blank, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    jpg = enc.tobytes() if ok else b""
                self._send_bytes(HTTPStatus.OK, "image/jpeg", jpg)
                return

            if self.path == "/" or self.path.startswith("/index"):
                self._send_bytes(HTTPStatus.OK, "text/html; charset=utf-8", html_blob)
                return

            self._send_bytes(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", b"Not Found")

        def log_message(self, fmt, *args):
            return

    return Handler


class ReusableServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def run_app(video_source, config_path: str, cooldown_s: float, host: str, port: int):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    app_state = AppState()
    agent = ElderWatchAgent(config)
    notifier = GemmaNotifier(config.get("model_paths", {}).get("gemma_270m", ""))
    notif_queue = queue.Queue()
    stop_event = threading.Event()
    novelty_state = {
        "last_conf": None,
        "last_ts": 0.0,
        "last_message": "",
    }

    # Only queue a new notification when it is meaningfully different.
    min_conf_delta = 0.12
    min_novelty_gap_s = 3.0

    def on_alert(event: FallEvent):
        notif_queue.put(
            NotificationEvent(
                confidence=event.confidence,
                pose_summary=event.pose_summary,
                timestamp=event.timestamp,
            )
        )

    def worker():
        while not stop_event.is_set():
            try:
                event = notif_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            now = time.time()
            if cooldown_s > 0:
                with app_state.lock:
                    prev_ts = app_state.notification_time
                if (now - prev_ts) < cooldown_s:
                    continue

            message = notifier.generate(event)

            with app_state.lock:
                last_conf = novelty_state["last_conf"]
                last_ts = novelty_state["last_ts"]
                last_message = novelty_state["last_message"]

            conf_is_new = (
                last_conf is None or abs(event.confidence - last_conf) >= min_conf_delta
            )
            message_is_new = message.strip() != (last_message or "").strip()
            time_is_new = (now - last_ts) >= min_novelty_gap_s

            if not (conf_is_new or message_is_new or time_is_new):
                continue

            with app_state.lock:
                app_state.notifications.insert(0, {
                    "message": message,
                    "confidence": round(event.confidence, 3),
                    "ts": now,
                })
                app_state.notifications = app_state.notifications[:12]
                app_state.notification_time = now
                novelty_state["last_conf"] = event.confidence
                novelty_state["last_ts"] = now
                novelty_state["last_message"] = message
            logger.warning("Queued notification: conf=%.3f", event.confidence)

    def on_frame(frame, pose_frame):
        ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            return
        with app_state.lock:
            app_state.latest_jpg = enc.tobytes()
            app_state.last_frame_ts = time.time()
            app_state.current_prob = agent.state.current_fall_prob
            app_state.current_alert = agent.state.current_alert.value

    handler_cls = make_handler(app_state)
    server = ReusableServer((host, port), handler_cls)

    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()

    agent.on_alert = on_alert
    agent.on_frame = on_frame

    agent_thread = threading.Thread(target=agent.start, args=(video_source,), daemon=True)
    agent_thread.start()

    logger.warning("Web app live at http://%s:%d", host, port)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        agent.stop()
        server.shutdown()
        server.server_close()
        worker_thread.join(timeout=1.0)
        agent_thread.join(timeout=2.0)


def main():
    parser = argparse.ArgumentParser(description="Realtime ElderWatch notification web app")
    parser.add_argument(
        "--source",
        default="0",
        help="Video source: 0 for webcam, path for file",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--cooldown-s",
        type=float,
        default=0.0,
        help="Minimum seconds between panel notification updates",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("elderwatch").setLevel(logging.ERROR)
    logging.getLogger("absl").setLevel(logging.ERROR)

    run_app(source, args.config, args.cooldown_s, args.host, args.port)


if __name__ == "__main__":
    main()
