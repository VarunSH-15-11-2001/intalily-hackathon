"""
ElderWatch Agent Loop
=====================
Ties together the three concurrent loops:
  Loop 1: Perception (MediaPipe PoseLandmarker) — always on
  Loop 2: Fall Detection (LSTM) — always on
  Loop 3: Context Assessment (Gemma) — event-driven

Uses the NEW MediaPipe Tasks API (PoseLandmarker) which works with
mediapipe >= 0.10.22.

This module provides the AgentLoop class that runs in real-time
on a video stream (webcam, RTSP, or file).
"""

import time
import threading
import queue
import logging
import os
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable
from collections import deque

import cv2
import numpy as np
import torch

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from train_lstm import FallDetectorLSTM

logger = logging.getLogger("elderwatch")

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
DEFAULT_MODEL_PATH = "models/pose_landmarker.task"


def ensure_pose_model(model_path: str = DEFAULT_MODEL_PATH):
    """Download pose landmarker model if not present."""
    if os.path.exists(model_path):
        return model_path
    print(f"Downloading pose landmarker model to {model_path}...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, model_path)
    print("Done.")
    return model_path


# ─── Data Types ─────────────────────────────────────────────────────

class AlertLevel(Enum):
    NONE = "none"
    MONITOR = "monitor"
    ALERT = "alert"
    ESCALATE = "escalate"
    EMERGENCY = "emergency"


@dataclass
class PoseFrame:
    """Single frame of pose data."""
    timestamp: float
    keypoints: np.ndarray       # (33, 4)
    bbox: Optional[tuple] = None
    frame: Optional[np.ndarray] = None


@dataclass
class FallEvent:
    """Detected fall event."""
    timestamp: float
    confidence: float
    pose_window: np.ndarray
    trigger_frame: Optional[np.ndarray] = None
    pose_summary: str = ""
    scene_description: str = ""
    alert_level: AlertLevel = AlertLevel.ALERT
    resolved: bool = False


@dataclass
class AgentState:
    """Current state of the monitoring agent."""
    is_running: bool = False
    current_alert: AlertLevel = AlertLevel.NONE
    last_fall_event: Optional[FallEvent] = None
    last_movement_time: float = 0.0
    frames_processed: int = 0
    falls_detected: int = 0
    status_message: str = "Initializing..."


# ─── Agent Loop ─────────────────────────────────────────────────────

class ElderWatchAgent:
    """
    Main agent that orchestrates the three loops.

    Usage:
        agent = ElderWatchAgent(config)
        agent.on_alert = my_callback
        agent.start(video_source=0)  # 0 for webcam
    """

    def __init__(self, config: dict):
        self.config = config
        self.state = AgentState()
        self.pose_buffer = deque(maxlen=config["lstm"]["sequence_length"])
        self.event_queue = queue.Queue()

        # Callbacks
        self.on_alert: Optional[Callable[[FallEvent], None]] = None
        self.on_escalation: Optional[Callable[[FallEvent], None]] = None
        self.on_scene_description: Optional[Callable[[FallEvent], None]] = None
        self.on_state_change: Optional[Callable[[AgentState], None]] = None
        self.on_frame: Optional[Callable[[np.ndarray, PoseFrame], None]] = None

        # Components (initialized in _setup)
        self._landmarker = None
        self._lstm_model = None
        self._norm_stats = None
        self._device = None
        self._lock = threading.Lock()
        self._is_video_source = False

    def _setup(self, is_video: bool = False):
        """Initialize ML models."""
        logger.info("Loading models...")
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._is_video_source = is_video

        # MediaPipe PoseLandmarker (new Tasks API)
        pose_model_path = self.config.get("model_paths", {}).get(
            "pose_landmarker", DEFAULT_MODEL_PATH
        )
        ensure_pose_model(pose_model_path)

        base_options = mp_python.BaseOptions(model_asset_path=pose_model_path)

        if is_video:
            # VIDEO mode for file playback
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            # LIVE_STREAM mode for webcam with async callback
            self._latest_pose_result = None
            self._latest_pose_timestamp = -1

            def pose_callback(result, output_image, timestamp_ms):
                self._latest_pose_result = result
                self._latest_pose_timestamp = timestamp_ms

            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.LIVE_STREAM,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                result_callback=pose_callback,
            )

        self._landmarker = vision.PoseLandmarker.create_from_options(options)

        # LSTM Fall Detector
        lstm_cfg = self.config["lstm"]
        self._lstm_model = FallDetectorLSTM(
            input_size=lstm_cfg["input_size"],
            hidden_size=lstm_cfg["hidden_size"],
            num_layers=lstm_cfg["num_layers"],
            dropout=0,
        ).to(self._device)

        model_path = self.config["model_paths"]["lstm"]
        checkpoint = torch.load(model_path, map_location=self._device,
                                weights_only=True)
        self._lstm_model.load_state_dict(checkpoint["model_state_dict"])
        self._lstm_model.eval()

        # Normalization stats
        norm_path = f"{self.config['data']['processed_dir']}/norm_stats.npz"
        stats = np.load(norm_path)
        self._norm_mean = torch.FloatTensor(stats["mean"]).to(self._device)
        self._norm_std = torch.FloatTensor(stats["std"]).to(self._device)

        logger.info(f"Models loaded on {self._device}")

    # ─── Loop 1: Perception ─────────────────────────────────────────

    def _extract_pose_video(self, frame: np.ndarray,
                             timestamp_ms: int) -> PoseFrame:
        """Extract pose from a single frame in VIDEO mode. ~15ms."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            landmarks = result.pose_landmarks[0]
            keypoints = np.array([
                [lm.x, lm.y, lm.z, lm.visibility]
                for lm in landmarks
            ], dtype=np.float32)
        else:
            keypoints = np.zeros((33, 4), dtype=np.float32)

        return PoseFrame(
            timestamp=time.time(),
            keypoints=keypoints,
            frame=frame,
        )

    def _extract_pose_live(self, frame: np.ndarray,
                            timestamp_ms: int) -> PoseFrame:
        """Extract pose from a single frame in LIVE_STREAM mode."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Async — sends frame, result arrives via callback
        self._landmarker.detect_async(mp_image, timestamp_ms)

        # Use latest available result
        if (self._latest_pose_result is not None and
                self._latest_pose_result.pose_landmarks and
                len(self._latest_pose_result.pose_landmarks) > 0):
            landmarks = self._latest_pose_result.pose_landmarks[0]
            keypoints = np.array([
                [lm.x, lm.y, lm.z, lm.visibility]
                for lm in landmarks
            ], dtype=np.float32)
        else:
            keypoints = np.zeros((33, 4), dtype=np.float32)

        return PoseFrame(
            timestamp=time.time(),
            keypoints=keypoints,
            frame=frame,
        )

    def _extract_pose(self, frame: np.ndarray,
                       timestamp_ms: int) -> PoseFrame:
        """Route to correct extraction method based on source type."""
        if self._is_video_source:
            return self._extract_pose_video(frame, timestamp_ms)
        else:
            return self._extract_pose_live(frame, timestamp_ms)

    # ─── Loop 2: Fall Detection ─────────────────────────────────────

    @torch.no_grad()
    def _detect_fall(self) -> Optional[float]:
        """Run LSTM on current pose buffer. ~10ms."""
        seq_len = self.config["lstm"]["sequence_length"]
        if len(self.pose_buffer) < seq_len:
            return None

        window = np.array([
            pf.keypoints.flatten() for pf in self.pose_buffer
        ], dtype=np.float32)

        x = torch.FloatTensor(window).unsqueeze(0).to(self._device)
        x = (x - self._norm_mean) / self._norm_std

        logit = self._lstm_model(x)
        prob = torch.sigmoid(logit).item()
        return prob

    def _compute_pose_summary(self) -> str:
        """Generate a text summary of the pose window for the controller."""
        if len(self.pose_buffer) < 2:
            return "Insufficient data"

        keypoints = np.array([pf.keypoints for pf in self.pose_buffer])

        hip_y = keypoints[:, 23:25, 1].mean(axis=1)
        shoulder_y = keypoints[:, 11:13, 1].mean(axis=1)

        hip_drop = hip_y[-1] - hip_y[0]
        shoulder_drop = shoulder_y[-1] - shoulder_y[0]

        velocities = np.diff(keypoints[:, :, :2], axis=0)
        avg_velocity = np.linalg.norm(velocities, axis=-1).mean()
        final_velocity = np.linalg.norm(velocities[-1], axis=-1).mean()

        final_kp = keypoints[-1]
        xs = final_kp[:, 0][final_kp[:, 3] > 0.5]
        ys = final_kp[:, 1][final_kp[:, 3] > 0.5]

        if len(xs) > 0 and len(ys) > 0:
            width = xs.max() - xs.min()
            height = ys.max() - ys.min()
            aspect = width / max(height, 0.01)
        else:
            aspect = 0

        return (
            f"Hip Y-drop: {hip_drop:+.3f}, Shoulder Y-drop: {shoulder_drop:+.3f}. "
            f"Avg velocity: {avg_velocity:.4f}, Final velocity: {final_velocity:.4f}. "
            f"Body aspect ratio: {aspect:.2f}. "
            f"{'Subject horizontal.' if aspect > 1.5 else 'Subject upright.' if aspect < 0.7 else 'Subject transitioning.'}"
        )

    # ─── Loop 3: Context Assessment ─────────────────────────────────

    def _handle_fall_event(self, confidence: float, pose_frame: PoseFrame):
        """Fast path: immediately create alert. Slow path: queue VLM."""
        pose_summary = self._compute_pose_summary()

        event = FallEvent(
            timestamp=time.time(),
            confidence=confidence,
            pose_window=np.array([pf.keypoints for pf in self.pose_buffer]),
            trigger_frame=pose_frame.frame,
            pose_summary=pose_summary,
            alert_level=AlertLevel.ALERT if confidence > 0.8 else AlertLevel.MONITOR,
        )

        with self._lock:
            self.state.last_fall_event = event
            self.state.falls_detected += 1
            self.state.current_alert = event.alert_level
            self.state.status_message = (
                f"FALL DETECTED (confidence: {confidence:.2f})"
            )

        logger.warning(
            f"FALL DETECTED | confidence={confidence:.2f} | "
            f"alert={event.alert_level.value}"
        )

        if self.on_alert:
            self.on_alert(event)

        # Slow path — VLM assessment (simulated)
        threading.Thread(
            target=self._run_vlm_assessment, args=(event,), daemon=True
        ).start()

        # Start stillness monitor
        threading.Thread(
            target=self._monitor_stillness, args=(event,), daemon=True
        ).start()

    def _run_vlm_assessment(self, event: FallEvent):
        """Simulate Gemma 3n scene assessment."""
        time.sleep(0.1)

        event.scene_description = (
            f"[VLM Assessment] Fall detected with confidence {event.confidence:.2f}. "
            f"Pose analysis: {event.pose_summary}"
        )

        logger.info(f"VLM assessment complete: {event.scene_description[:80]}...")

        if self.on_scene_description:
            self.on_scene_description(event)

    def _monitor_stillness(self, event: FallEvent):
        """Monitor for continued stillness after fall."""
        escalation_timeout = self.config["agent"]["escalation_timeout_s"]
        stillness_threshold = self.config["agent"]["stillness_threshold"]

        start_time = time.time()

        while (time.time() - start_time) < escalation_timeout:
            time.sleep(1)

            if event.resolved:
                return

            if len(self.pose_buffer) >= 2:
                recent = list(self.pose_buffer)[-2:]
                movement = np.linalg.norm(
                    recent[-1].keypoints - recent[-2].keypoints
                )
                if movement > stillness_threshold:
                    event.resolved = True
                    with self._lock:
                        self.state.current_alert = AlertLevel.NONE
                        self.state.status_message = "Movement resumed — incident resolved"
                    logger.info("Movement detected — fall event resolved")
                    return

        if not event.resolved:
            event.alert_level = AlertLevel.ESCALATE
            with self._lock:
                self.state.current_alert = AlertLevel.ESCALATE
                self.state.status_message = (
                    f"ESCALATION — no movement for {escalation_timeout}s"
                )

            logger.critical(f"ESCALATION — no movement for {escalation_timeout}s")

            if self.on_escalation:
                self.on_escalation(event)

    # ─── Main Loop ──────────────────────────────────────────────────

    def start(self, video_source=0):
        """
        Start the agent on a video source.

        Args:
            video_source: 0 for webcam, path for video file,
                         RTSP URL for IP camera
        """
        is_video = isinstance(video_source, str)
        self._setup(is_video=is_video)

        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {video_source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        threshold = self.config["lstm"]["fall_threshold"]

        self.state.is_running = True
        self.state.status_message = "Monitoring — no events"
        self.state.last_movement_time = time.time()

        logger.info(f"Agent started | source={video_source} | fps={fps}")

        start_time = time.time()

        try:
            while self.state.is_running:
                ret, frame = cap.read()
                if not ret:
                    if isinstance(video_source, str):
                        logger.info("Video ended")
                        break
                    continue

                # Compute timestamp
                if is_video:
                    timestamp_ms = int(self.state.frames_processed * 1000 / fps)
                else:
                    timestamp_ms = int((time.time() - start_time) * 1000)

                # Loop 1: Perception
                t0 = time.perf_counter()
                pose_frame = self._extract_pose(frame, timestamp_ms)
                t1 = time.perf_counter()

                self.pose_buffer.append(pose_frame)
                self.state.frames_processed += 1

                # Loop 2: Fall Detection
                fall_prob = self._detect_fall()
                t2 = time.perf_counter()

                # Visualization callback
                if self.on_frame:
                    self.on_frame(frame, pose_frame)

                # Loop 3: Trigger on fall
                if fall_prob is not None and fall_prob > threshold:
                    if self.state.current_alert == AlertLevel.NONE:
                        self._handle_fall_event(fall_prob, pose_frame)

                # Track movement
                if len(self.pose_buffer) >= 2:
                    recent = list(self.pose_buffer)[-2:]
                    movement = np.linalg.norm(
                        recent[-1].keypoints - recent[-2].keypoints
                    )
                    if movement > self.config["agent"]["stillness_threshold"]:
                        self.state.last_movement_time = time.time()

                        if (self.state.current_alert != AlertLevel.NONE and
                                self.state.last_fall_event):
                            self.state.last_fall_event.resolved = True
                            self.state.current_alert = AlertLevel.NONE
                            self.state.status_message = "Monitoring — no events"

                # Periodic logging (every 10 frames)
                if self.state.frames_processed % 10 == 0:
                    pose_ms = (t1 - t0) * 1000
                    lstm_ms = (t2 - t1) * 1000
                    prob_str = f"{fall_prob:.3f}" if fall_prob is not None else "N/A"
                    logger.info(
                        f"Frame {self.state.frames_processed} | "
                        f"Pose: {pose_ms:.1f}ms | LSTM: {lstm_ms:.1f}ms | "
                        f"P(fall): {prob_str}"
                    )

        except KeyboardInterrupt:
            logger.info("Stopped by user")
        finally:
            self._landmarker.close()
            cap.release()
            self.state.is_running = False
            logger.info(
                f"Agent stopped | Frames: {self.state.frames_processed} | "
                f"Falls: {self.state.falls_detected}"
            )

    def stop(self):
        """Stop the agent loop."""
        self.state.is_running = False


# ─── Demo Runner ────────────────────────────────────────────────────

def run_demo(video_source, config_path: str = "configs/config.yaml"):
    """Run agent with visual display for demo."""
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    agent = ElderWatchAgent(config)

    # Set up callbacks
    def on_alert(event: FallEvent):
        print(f"\n{'='*60}")
        print(f"FALL ALERT | Confidence: {event.confidence:.2f}")
        print(f"   {event.pose_summary}")
        print(f"{'='*60}\n")

    def on_escalation(event: FallEvent):
        print(f"\n{'!'*60}")
        print(f"ESCALATION — No movement detected!")
        print(f"{'!'*60}\n")

    def on_scene(event: FallEvent):
        print(f"\nScene Assessment: {event.scene_description}\n")

    def on_frame(frame, pose_frame: PoseFrame):
        annotated = frame.copy()
        status = agent.state.status_message
        alert = agent.state.current_alert
        color = (0, 255, 0) if alert == AlertLevel.NONE else (0, 0, 255)

        cv2.putText(annotated, status, (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(annotated, f"Frame: {agent.state.frames_processed}",
                     (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw keypoints if detected
        h, w = frame.shape[:2]
        kp = pose_frame.keypoints
        for i in range(33):
            if kp[i, 3] > 0.5:  # visibility threshold
                x, y = int(kp[i, 0] * w), int(kp[i, 1] * h)
                cv2.circle(annotated, (x, y), 3, (0, 255, 255), -1)

        cv2.imshow("ElderWatch", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            agent.stop()

    agent.on_alert = on_alert
    agent.on_escalation = on_escalation
    agent.on_scene_description = on_scene
    agent.on_frame = on_frame

    agent.start(video_source)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Run ElderWatch agent")
    parser.add_argument("--source", default="0",
                        help="Video source: 0 for webcam, path for file")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    run_demo(source, args.config)