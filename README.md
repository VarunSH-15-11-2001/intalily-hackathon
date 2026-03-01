# ElderWatch

Agentic, on-device fall monitoring for elderly safety.

ElderWatch combines real-time pose perception, temporal fall detection, and event handling into a single monitoring loop. The system is designed to run locally, with optional Gemma 270M fine-tuning assets for language-based context workflows.

## What We Implemented

- Real-time pose extraction using MediaPipe Pose landmarks (33 keypoints × 4 values).
- Sequence-based fall detection with a BiLSTM + attention classifier.
- End-to-end training pipeline: dataset download, pose extraction, windowing, augmentation, and LSTM training.
- Agent loop that continuously monitors video, triggers fall events, and manages alert state transitions.
- Optional Gemma 270M data generation + fine-tuning scripts.

## Simple Agentic Architecture

```
Video Source (webcam / file)
    │
    ▼
Perception Agent
(MediaPipe pose keypoints)
    │
    ▼
Fall Detection Agent
(BiLSTM + temporal attention)
    │
   fall probability
    │
    ▼
Response Agent
(event creation, alert escalation, stillness monitoring,
 optional context assessment)
```

## Run Commands

Assuming dependencies are installed and data/models are already in place.

Run core agent on webcam:

Run core agent on video file:

```bash
python3 scripts/agent_loop.py --source data/fall-01-cam0.mp4
```

Run realtime notification web app (video + mobile-style alert panel):

```bash
python3 scripts/realtime_notification_app.py --source data/set1/fall-01-cam0.mp4
```

Open in browser:

```bash
http://127.0.0.1:8000
```

Set-specific source behavior:

```bash
# set1 source: crops to right half
python3 scripts/agent_loop.py --source data/set1/fall-01-cam0.mp4

# set2 source: skips first 9 seconds
python3 scripts/agent_loop.py --source data/set2/fall-01-cam0.mp4
```

## Training Pipeline

```bash
python scripts/download_urfall.py --include-csv
python scripts/extract_poses.py --urfall-dir data/urfall --output-dir data/poses
python scripts/prepare_dataset.py --poses-dir data/poses --output-dir data/processed
python scripts/train_lstm.py --config configs/config.yaml
```

## Gemma 270M (Optional)

```bash
python scripts/generate_gemma_data.py --output data/gemma_270m_finetune.jsonl
python scripts/finetune_gemma.py --data data/gemma_270m_finetune.jsonl
```

Only Gemma 270M is used in this project configuration.

## Project Structure

```
elderwatch/
├── configs/
│   └── config.yaml
├── data/
│   └── processed/
├── models/
│   ├── fall_detector_lstm.pt
│   └── pose_landmarker.task
├── scripts/
│   ├── agent_loop.py
│   ├── download_urfall.py
│   ├── extract_poses.py
│   ├── prepare_dataset.py
│   ├── train_lstm.py
│   ├── generate_gemma_data.py
│   └── finetune_gemma.py
├── requirements.txt
└── README.md
```
