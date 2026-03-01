"""
Realtime Notification App
=========================
Runs ElderWatch LSTM fall detection and shows realtime notifications in a
simple side panel next to the video stream.

Usage:
  python scripts/realtime_notification_app.py --source 0
  python scripts/realtime_notification_app.py --source path/to/video.mp4
"""

import argparse
import logging
import os
import queue
import threading
import time
import textwrap
from dataclasses import dataclass

import cv2
import numpy as np
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
                f"Fall detected (confidence {event.confidence:.2f}). "
                "Please check immediately."
            )

        prompt = (
            "You are an elderly fall monitoring assistant.\n"
            "Write one short caregiver push notification.\n"
            "No markdown. 18 words max.\n\n"
            f"Fall confidence: {event.confidence:.2f}\n"
            f"Pose summary: {event.pose_summary}\n"
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
                    f"Fall detected (confidence {event.confidence:.2f}). "
                    "Please check immediately."
                )
            return generated.splitlines()[0][:180]
        except Exception as exc:
            logger.warning("Gemma generation failed; using fallback text: %s", exc)
            return (
                f"Fall detected (confidence {event.confidence:.2f}). "
                "Please check immediately."
            )


def render_ui(frame: np.ndarray, agent: ElderWatchAgent, message: str) -> np.ndarray:
    """Render video + rectangular notification panel."""
    panel_w = 380
    h, w = frame.shape[:2]
    canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
    canvas[:, :w] = frame
    panel = canvas[:, w:]
    panel[:] = (28, 28, 28)

    cv2.rectangle(panel, (14, 14), (panel_w - 14, h - 14), (90, 90, 90), 2)
    cv2.putText(panel, "ElderWatch", (28, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (240, 240, 240), 2)
    cv2.putText(panel, "Notifications", (28, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (190, 190, 190), 1)

    prob_text = f"LSTM p(fall): {agent.state.current_fall_prob:.3f}"
    alert_text = f"Alert: {agent.state.current_alert.value}"
    alert_color = (100, 220, 100) if agent.state.current_alert == AlertLevel.NONE else (70, 70, 240)
    cv2.putText(panel, prob_text, (28, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(panel, alert_text, (28, 148),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, alert_color, 1)

    cv2.putText(panel, "Latest notification:", (28, 188),
                cv2.FONT_HERSHEY_SIMPLEX, 0.56, (230, 230, 230), 1)

    wrapped = textwrap.wrap(message, width=34)[:8]
    y = 220
    for line in wrapped:
        cv2.putText(panel, line, (28, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1)
        y += 26

    cv2.putText(panel, "Press q to quit", (28, h - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (170, 170, 170), 1)
    return canvas


def run_app(video_source, config_path: str, cooldown_s: float):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    agent = ElderWatchAgent(config)
    notifier = GemmaNotifier(config.get("model_paths", {}).get("gemma_270m", ""))
    notif_queue = queue.Queue()
    stop_event = threading.Event()
    notif_state = {
        "message": "Monitoring in realtime.",
        "last_ts": 0.0,
    }
    state_lock = threading.Lock()

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
            with state_lock:
                prev_ts = notif_state["last_ts"]
            if (now - prev_ts) < cooldown_s:
                continue

            message = notifier.generate(event)
            with state_lock:
                notif_state["message"] = message
                notif_state["last_ts"] = now
            logger.warning("UI notification updated: %s", message)

    def on_frame(frame, pose_frame):
        with state_lock:
            message = notif_state["message"]
        ui = render_ui(frame, agent, message)
        cv2.imshow("ElderWatch Realtime Notifications", ui)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            agent.stop()

    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()

    agent.on_alert = on_alert
    agent.on_frame = on_frame

    try:
        agent.start(video_source)
    finally:
        stop_event.set()
        worker_thread.join(timeout=1.0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Realtime ElderWatch notification UI")
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
        default=8.0,
        help="Minimum seconds between panel notification updates",
    )
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("elderwatch").setLevel(logging.ERROR)
    logging.getLogger("absl").setLevel(logging.ERROR)

    run_app(source, args.config, args.cooldown_s)


if __name__ == "__main__":
    main()
