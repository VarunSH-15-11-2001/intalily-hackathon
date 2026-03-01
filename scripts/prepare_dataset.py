"""
Dataset Preparation (v3 — frame-level labels)
==============================================
Uses the URFall per-frame annotations to label EACH WINDOW correctly:
  - Label -1 = not lying (standing, walking, sitting)
  - Label  0 = transitioning (actively falling) — we treat as FALL
  - Label  1 = lying on ground — we treat as FALL

A window is labeled as FALL (1) only if a significant portion of its frames
are labeled 0 or 1. Otherwise it's ADL (0).

This prevents the old problem where standing-before-a-fall frames in a
"fall video" were incorrectly labeled as falls.

Also splits at the VIDEO level to prevent data leakage.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import argparse
from collections import defaultdict
import random
import yaml


# MediaPipe left/right landmark pairs for mirroring
MIRROR_PAIRS = [
    (1, 4), (2, 5), (3, 6),
    (7, 8),
    (9, 10),
    (11, 12), (13, 14), (15, 16),
    (17, 18), (19, 20), (21, 22),
    (23, 24), (25, 26), (27, 28),
    (29, 30), (31, 32),
]


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_frame_labels(urfall_dir: str) -> dict:
    """
    Load per-frame labels from URFall feature CSVs.

    Returns:
        dict mapping (sequence_name, frame_num) -> label
        where label: -1 = not lying, 0 = transitioning, 1 = lying
    """
    urfall_path = Path(urfall_dir)

    falls_csv = urfall_path / "urfall-cam0-falls.csv"
    adls_csv = urfall_path / "urfall-cam0-adls.csv"

    labels = {}  # {sequence_name: {frame_num: label}}

    # Parse falls CSV: seq_name, frame, label, ...features
    if falls_csv.exists():
        df = pd.read_csv(falls_csv, header=None)
        for _, row in df.iterrows():
            seq = row.iloc[0].strip()    # e.g., "fall-01"
            frame = int(row.iloc[1])
            label = int(row.iloc[2])     # -1, 0, or 1
            if seq not in labels:
                labels[seq] = {}
            labels[seq][frame] = label
        print(f"  Loaded fall labels: {len(labels)} sequences")
    else:
        print(f"  WARNING: {falls_csv} not found!")

    # Parse ADLs CSV — all frames are -1 (not lying)
    if adls_csv.exists():
        df = pd.read_csv(adls_csv, header=None)
        for _, row in df.iterrows():
            seq = row.iloc[0].strip()
            frame = int(row.iloc[1])
            label = int(row.iloc[2])
            if seq not in labels:
                labels[seq] = {}
            labels[seq][frame] = label
        print(f"  Loaded ADL labels: total {len(labels)} sequences")
    else:
        print(f"  WARNING: {adls_csv} not found!")

    return labels


def get_sequence_name(video_path: str) -> str:
    """
    Extract sequence name from video filename.
    'fall-01-cam0' -> 'fall-01'
    'adl-01-cam0' -> 'adl-01'
    """
    stem = Path(video_path).stem
    # Remove '-cam0' suffix
    parts = stem.split("-cam0")[0] if "-cam0" in stem else stem
    return parts


def label_window(frame_labels: dict, start_frame: int, end_frame: int,
                  fall_ratio_threshold: float = 0.3) -> int:
    """
    Label a window based on the per-frame annotations.

    A window is labeled as FALL (1) if more than fall_ratio_threshold
    of its frames have label 0 (transitioning) or 1 (lying).

    Args:
        frame_labels: {frame_num: label} for this sequence
        start_frame: first frame index of the window (1-based)
        end_frame: last frame index of the window (1-based)
        fall_ratio_threshold: fraction of fall frames needed to label window as fall

    Returns:
        0 (ADL/no fall) or 1 (fall)
    """
    if not frame_labels:
        return 0  # No labels available — assume ADL

    fall_count = 0
    total = 0

    for f in range(start_frame, end_frame + 1):
        if f in frame_labels:
            total += 1
            if frame_labels[f] >= 0:  # 0 = transitioning, 1 = lying
                fall_count += 1

    if total == 0:
        return 0

    return 1 if (fall_count / total) >= fall_ratio_threshold else 0


def create_labeled_windows(poses: np.ndarray, frame_labels: dict,
                            seq_len: int, stride: int,
                            fall_ratio_threshold: float = 0.3
                            ) -> tuple:
    """
    Create sliding windows with per-window labels based on frame annotations.

    Returns:
        windows: (num_windows, seq_len, 132)
        labels: (num_windows,) — 0 or 1
    """
    num_frames = poses.shape[0]
    if num_frames < seq_len:
        pad = np.zeros((seq_len - num_frames, *poses.shape[1:]),
                       dtype=poses.dtype)
        poses = np.concatenate([poses, pad], axis=0)
        num_frames = seq_len

    flat = poses.reshape(num_frames, -1)

    windows = []
    labels = []

    for start in range(0, num_frames - seq_len + 1, stride):
        end = start + seq_len
        windows.append(flat[start:end])

        # Frame numbers are 1-based in the CSV
        label = label_window(frame_labels, start + 1, end,
                              fall_ratio_threshold)
        labels.append(label)

    return np.array(windows, dtype=np.float32), np.array(labels, dtype=np.float32)


# ─── Augmentation ───────────────────────────────────────────────────

def augment_mirror(poses: np.ndarray) -> np.ndarray:
    mirrored = poses.copy()
    for left, right in MIRROR_PAIRS:
        mirrored[:, left], mirrored[:, right] = (
            poses[:, right].copy(), poses[:, left].copy()
        )
    mirrored[:, :, 0] = 1.0 - mirrored[:, :, 0]
    return mirrored


def augment_speed(poses: np.ndarray, factor: float) -> np.ndarray:
    num_frames = poses.shape[0]
    new_len = int(num_frames / factor)
    if new_len < 2:
        return poses
    indices = np.linspace(0, num_frames - 1, new_len).astype(int)
    return poses[indices]


def augment_noise(poses: np.ndarray, std: float = 0.01) -> np.ndarray:
    noisy = poses.copy()
    noise = np.random.randn(*poses.shape).astype(np.float32) * std
    noise[:, :, 3] = 0
    noisy += noise
    return noisy


def augment_dropout(poses: np.ndarray, drop_prob: float = 0.05) -> np.ndarray:
    dropped = poses.copy()
    mask = np.random.random((poses.shape[0], poses.shape[1])) < drop_prob
    dropped[mask] = 0
    return dropped


def get_augmented_windows(poses: np.ndarray, frame_labels: dict,
                           seq_len: int, stride: int,
                           augment: bool, aug_cfg: dict,
                           fall_ratio_threshold: float = 0.3) -> tuple:
    """
    Create windows from a video with optional augmentation.
    NOTE: Augmentation only changes poses, not labels. Frame labels stay aligned
    for the base variant. For augmented variants (mirror, speed), we apply the
    same frame labels since the activity doesn't change.
    """
    all_X, all_y = [], []

    # Base
    X, y = create_labeled_windows(poses, frame_labels, seq_len, stride,
                                   fall_ratio_threshold)
    all_X.append(X)
    all_y.append(y)

    if not augment:
        return np.concatenate(all_X), np.concatenate(all_y)

    # Mirror
    if aug_cfg.get("mirror", True):
        X_m, y_m = create_labeled_windows(augment_mirror(poses), frame_labels,
                                           seq_len, stride, fall_ratio_threshold)
        all_X.append(X_m)
        all_y.append(y_m)

    # Speed variations
    for factor in aug_cfg.get("speed_factors", [1.0]):
        if factor != 1.0:
            sped = augment_speed(poses, factor)
            # Re-label with adjusted frame mapping
            # For simplicity, use same labels (activity doesn't change)
            X_s, y_s = create_labeled_windows(sped, frame_labels,
                                               seq_len, stride, fall_ratio_threshold)
            all_X.append(X_s)
            all_y.append(y_s)

    # Noise on base
    if aug_cfg.get("noise_std", 0) > 0:
        X_n, y_n = create_labeled_windows(
            augment_noise(poses, aug_cfg["noise_std"]),
            frame_labels, seq_len, stride, fall_ratio_threshold
        )
        all_X.append(X_n)
        all_y.append(y_n)

    # Dropout on base
    if aug_cfg.get("random_drop_prob", 0) > 0:
        X_d, y_d = create_labeled_windows(
            augment_dropout(poses, aug_cfg["random_drop_prob"]),
            frame_labels, seq_len, stride, fall_ratio_threshold
        )
        all_X.append(X_d)
        all_y.append(y_d)

    return np.concatenate(all_X), np.concatenate(all_y)


# ─── Main Pipeline ──────────────────────────────────────────────────

def prepare_dataset(poses_dir: str, output_dir: str,
                     config_path: str = "configs/config.yaml"):
    """
    Full dataset preparation:
    1. Load per-frame labels from URFall feature CSVs
    2. Split at VIDEO level
    3. Create per-window labels
    4. Augment training set only
    """
    cfg = load_config(config_path)
    lstm_cfg = cfg["lstm"]
    aug_cfg = cfg["augmentation"]

    poses_path = Path(poses_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load metadata
    with open(poses_path / "metadata.json") as f:
        metadata = json.load(f)

    # Load per-frame labels
    urfall_dir = cfg["data"]["urfall_dir"]
    print("Loading per-frame labels...")
    frame_labels = load_frame_labels(urfall_dir)

    # Group by type
    fall_videos = [m for m in metadata if m["label"] == "fall"]
    adl_videos = [m for m in metadata if m["label"] == "adl"]

    print(f"\nFound {len(fall_videos)} fall videos, {len(adl_videos)} ADL videos")

    # Shuffle deterministically
    random.seed(42)
    random.shuffle(fall_videos)
    random.shuffle(adl_videos)

    # Split at video level: 70/15/15
    def split_videos(videos):
        n = len(videos)
        n_train = max(1, int(0.7 * n))
        n_val = max(1, int(0.15 * n))
        train = videos[:n_train]
        val = videos[n_train:n_train + n_val]
        test = videos[n_train + n_val:]
        return train, val, test

    fall_train, fall_val, fall_test = split_videos(fall_videos)
    adl_train, adl_val, adl_test = split_videos(adl_videos)

    print(f"\nVideo-level split:")
    print(f"  Falls  — train: {len(fall_train)}, val: {len(fall_val)}, test: {len(fall_test)}")
    print(f"  ADL    — train: {len(adl_train)}, val: {len(adl_val)}, test: {len(adl_test)}")

    train_videos = fall_train + adl_train
    val_videos = fall_val + adl_val
    test_videos = fall_test + adl_test

    def process_split(video_metas, augment: bool, split_name: str):
        all_X, all_y = [], []

        for meta in video_metas:
            video_name = Path(meta["video"]).stem
            pose_file = poses_path / f"{video_name}_poses.npy"

            # Custom videos are saved with "custom-" prefix
            if not pose_file.exists() and meta.get("source") == "custom":
                pose_file = poses_path / f"custom-{video_name}_poses.npy"

            if not pose_file.exists():
                print(f"  Skipping {video_name} — not found")
                continue

            poses = np.load(pose_file)

            # Get sequence name for label lookup
            seq_name = get_sequence_name(meta["video"])
            seq_labels = frame_labels.get(seq_name, {})

            if not seq_labels:
                # No frame-level labels available
                is_custom = meta.get("source") == "custom"

                if meta["label"] == "adl":
                    # ADL — all frames are no-fall
                    seq_labels = {f: -1 for f in range(1, len(poses) + 1)}
                elif is_custom and meta["label"] == "fall":
                    # Custom fall video — assume all frames are fall
                    # (custom clips should be short clips of actual falls)
                    seq_labels = {f: 1 for f in range(1, len(poses) + 1)}
                else:
                    print(f"  WARNING: No frame labels for {seq_name}, skipping")
                    continue

            stride = lstm_cfg["stride"] if augment else lstm_cfg["sequence_length"]

            X, y = get_augmented_windows(
                poses, seq_labels,
                seq_len=lstm_cfg["sequence_length"],
                stride=stride,
                augment=augment,
                aug_cfg=aug_cfg if augment else {},
            )
            all_X.append(X)
            all_y.append(y)

        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)

        # Shuffle within split
        idx = np.random.permutation(len(X))
        X, y = X[idx], y[idx]

        n_fall = int((y == 1).sum())
        n_adl = int((y == 0).sum())
        print(f"  {split_name}: {len(X)} windows "
              f"(fall: {n_fall}, ADL: {n_adl}, "
              f"fall ratio: {n_fall/max(len(X),1):.1%})")
        return X, y

    print("\nProcessing splits...")
    X_train, y_train = process_split(train_videos, augment=True, split_name="Train")
    X_val, y_val = process_split(val_videos, augment=False, split_name="Val")
    X_test, y_test = process_split(test_videos, augment=False, split_name="Test")

    # Normalization from training set only
    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)
    std[std < 1e-6] = 1.0

    # Save
    np.savez_compressed(output_path / "train.npz", X=X_train, y=y_train)
    np.savez_compressed(output_path / "val.npz", X=X_val, y=y_val)
    np.savez_compressed(output_path / "test.npz", X=X_test, y=y_test)
    np.savez(output_path / "norm_stats.npz", mean=mean, std=std)

    split_info = {
        "train_videos": [m["video"] for m in train_videos],
        "val_videos": [m["video"] for m in val_videos],
        "test_videos": [m["video"] for m in test_videos],
    }
    with open(output_path / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare LSTM dataset")
    parser.add_argument("--poses-dir", default="data/poses")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    prepare_dataset(args.poses_dir, args.output_dir, args.config)