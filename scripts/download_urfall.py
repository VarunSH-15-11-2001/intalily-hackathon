"""
Dataset Preparation (v2 — video-level split)
=============================================
Converts raw pose sequences into sliding-window samples for LSTM training.

CRITICAL FIX: Splits at the VIDEO level first, then windows within each split.
This prevents data leakage where windows from the same video appear in both
train and test sets — which causes 100% accuracy (memorization, not learning).

Output: train/val/test splits as .npz files with sequences and labels.
"""

import numpy as np
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


def create_sliding_windows(poses: np.ndarray, seq_len: int,
                            stride: int) -> np.ndarray:
    """
    Create sliding window sequences from a pose array.

    Args:
        poses: (num_frames, 33, 4)
        seq_len: window size
        stride: step between windows

    Returns:
        windows: (num_windows, seq_len, 132)
    """
    num_frames = poses.shape[0]
    if num_frames < seq_len:
        pad = np.zeros((seq_len - num_frames, *poses.shape[1:]),
                       dtype=poses.dtype)
        poses = np.concatenate([poses, pad], axis=0)
        num_frames = seq_len

    flat = poses.reshape(num_frames, -1)

    windows = []
    for start in range(0, num_frames - seq_len + 1, stride):
        windows.append(flat[start:start + seq_len])

    return np.array(windows, dtype=np.float32)


def augment_mirror(poses: np.ndarray) -> np.ndarray:
    """Mirror pose by swapping left/right landmarks and flipping x."""
    mirrored = poses.copy()
    for left, right in MIRROR_PAIRS:
        mirrored[:, left], mirrored[:, right] = (
            poses[:, right].copy(), poses[:, left].copy()
        )
    mirrored[:, :, 0] = 1.0 - mirrored[:, :, 0]
    return mirrored


def augment_speed(poses: np.ndarray, factor: float) -> np.ndarray:
    """Resample poses to simulate speed change."""
    num_frames = poses.shape[0]
    new_len = int(num_frames / factor)
    if new_len < 2:
        return poses
    indices = np.linspace(0, num_frames - 1, new_len).astype(int)
    return poses[indices]


def augment_noise(poses: np.ndarray, std: float = 0.01) -> np.ndarray:
    """Add Gaussian noise to keypoint coordinates (not visibility)."""
    noisy = poses.copy()
    noise = np.random.randn(*poses.shape).astype(np.float32) * std
    noise[:, :, 3] = 0
    noisy += noise
    return noisy


def augment_dropout(poses: np.ndarray, drop_prob: float = 0.05) -> np.ndarray:
    """Randomly zero out entire keypoints to simulate occlusion."""
    dropped = poses.copy()
    mask = np.random.random((poses.shape[0], poses.shape[1])) < drop_prob
    dropped[mask] = 0
    return dropped


def get_windows_for_video(poses: np.ndarray, label: int,
                           seq_len: int, stride: int,
                           augment: bool = True,
                           aug_cfg: dict = None) -> tuple:
    """
    Get all windowed samples for a single video, with optional augmentation.
    Only augments training data.
    """
    all_windows = []
    all_labels = []

    # Collect variants
    variants = [poses]

    if augment and aug_cfg:
        if aug_cfg.get("mirror", True):
            variants.append(augment_mirror(poses))

        for factor in aug_cfg.get("speed_factors", [1.0]):
            if factor != 1.0:
                variants.append(augment_speed(poses, factor))
                if aug_cfg.get("mirror", True):
                    variants.append(augment_speed(augment_mirror(poses), factor))

    for variant in variants:
        # Base windows
        windows = create_sliding_windows(variant, seq_len, stride)
        all_windows.append(windows)
        all_labels.append(np.full(len(windows), label))

        if augment and aug_cfg:
            # Noisy version
            if aug_cfg.get("noise_std", 0) > 0:
                noisy = augment_noise(variant, aug_cfg["noise_std"])
                w = create_sliding_windows(noisy, seq_len, stride)
                all_windows.append(w)
                all_labels.append(np.full(len(w), label))

            # Dropout version
            if aug_cfg.get("random_drop_prob", 0) > 0:
                dropped = augment_dropout(variant, aug_cfg["random_drop_prob"])
                w = create_sliding_windows(dropped, seq_len, stride)
                all_windows.append(w)
                all_labels.append(np.full(len(w), label))

    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X, y


def prepare_dataset(poses_dir: str, output_dir: str,
                     config_path: str = "configs/config.yaml"):
    """
    Full dataset preparation with VIDEO-LEVEL splitting.

    Split strategy:
      - Group videos by label (fall vs ADL)
      - Within each group, assign 70% train / 15% val / 15% test
      - Only augment training videos
      - This ensures NO windows from the same video appear in multiple splits
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

    # Group by label
    fall_videos = [m for m in metadata if m["label"] == "fall"]
    adl_videos = [m for m in metadata if m["label"] == "adl"]

    print(f"Found {len(fall_videos)} fall videos, {len(adl_videos)} ADL videos")

    # Shuffle deterministically
    random.seed(42)
    random.shuffle(fall_videos)
    random.shuffle(adl_videos)

    # Split at video level: 70/15/15
    def split_videos(videos):
        n = len(videos)
        n_train = max(1, int(0.7 * n))
        n_val = max(1, int(0.15 * n))
        # rest goes to test
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

    # Process each split
    def process_split(video_metas, augment: bool, split_name: str):
        all_X, all_y = [], []

        for meta in video_metas:
            video_name = Path(meta["video"]).stem
            pose_file = poses_path / f"{video_name}_poses.npy"

            if not pose_file.exists():
                print(f"  Skipping {video_name} — not found")
                continue

            poses = np.load(pose_file)
            label = meta["label_id"]

            # Use larger stride for val/test (no augmentation needed)
            stride = lstm_cfg["stride"] if augment else lstm_cfg["sequence_length"]

            X, y = get_windows_for_video(
                poses, label,
                seq_len=lstm_cfg["sequence_length"],
                stride=stride,
                augment=augment,
                aug_cfg=aug_cfg if augment else None,
            )
            all_X.append(X)
            all_y.append(y)

        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)

        # Shuffle within split
        idx = np.random.permutation(len(X))
        X, y = X[idx], y[idx]

        print(f"  {split_name}: {len(X)} samples "
              f"(falls: {(y==1).sum()}, ADL: {(y==0).sum()})")
        return X, y

    print("\nProcessing splits...")
    X_train, y_train = process_split(train_videos, augment=True, split_name="Train")
    X_val, y_val = process_split(val_videos, augment=False, split_name="Val")
    X_test, y_test = process_split(test_videos, augment=False, split_name="Test")

    # Compute normalization stats from training set ONLY
    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)
    std[std < 1e-6] = 1.0

    # Save
    np.savez_compressed(output_path / "train.npz", X=X_train, y=y_train)
    np.savez_compressed(output_path / "val.npz", X=X_val, y=y_val)
    np.savez_compressed(output_path / "test.npz", X=X_test, y=y_test)
    np.savez(output_path / "norm_stats.npz", mean=mean, std=std)

    # Save split info for reproducibility
    split_info = {
        "train_videos": [m["video"] for m in train_videos],
        "val_videos": [m["video"] for m in val_videos],
        "test_videos": [m["video"] for m in test_videos],
    }
    with open(output_path / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"\nSaved to {output_path}")
    print(f"Split info saved to {output_path / 'split_info.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare LSTM dataset")
    parser.add_argument("--poses-dir", default="data/poses")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    prepare_dataset(args.poses_dir, args.output_dir, args.config)"""
Dataset Preparation (v2 — video-level split)
=============================================
Converts raw pose sequences into sliding-window samples for LSTM training.

CRITICAL FIX: Splits at the VIDEO level first, then windows within each split.
This prevents data leakage where windows from the same video appear in both
train and test sets — which causes 100% accuracy (memorization, not learning).

Output: train/val/test splits as .npz files with sequences and labels.
"""

import numpy as np
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


def create_sliding_windows(poses: np.ndarray, seq_len: int,
                            stride: int) -> np.ndarray:
    """
    Create sliding window sequences from a pose array.

    Args:
        poses: (num_frames, 33, 4)
        seq_len: window size
        stride: step between windows

    Returns:
        windows: (num_windows, seq_len, 132)
    """
    num_frames = poses.shape[0]
    if num_frames < seq_len:
        pad = np.zeros((seq_len - num_frames, *poses.shape[1:]),
                       dtype=poses.dtype)
        poses = np.concatenate([poses, pad], axis=0)
        num_frames = seq_len

    flat = poses.reshape(num_frames, -1)

    windows = []
    for start in range(0, num_frames - seq_len + 1, stride):
        windows.append(flat[start:start + seq_len])

    return np.array(windows, dtype=np.float32)


def augment_mirror(poses: np.ndarray) -> np.ndarray:
    """Mirror pose by swapping left/right landmarks and flipping x."""
    mirrored = poses.copy()
    for left, right in MIRROR_PAIRS:
        mirrored[:, left], mirrored[:, right] = (
            poses[:, right].copy(), poses[:, left].copy()
        )
    mirrored[:, :, 0] = 1.0 - mirrored[:, :, 0]
    return mirrored


def augment_speed(poses: np.ndarray, factor: float) -> np.ndarray:
    """Resample poses to simulate speed change."""
    num_frames = poses.shape[0]
    new_len = int(num_frames / factor)
    if new_len < 2:
        return poses
    indices = np.linspace(0, num_frames - 1, new_len).astype(int)
    return poses[indices]


def augment_noise(poses: np.ndarray, std: float = 0.01) -> np.ndarray:
    """Add Gaussian noise to keypoint coordinates (not visibility)."""
    noisy = poses.copy()
    noise = np.random.randn(*poses.shape).astype(np.float32) * std
    noise[:, :, 3] = 0
    noisy += noise
    return noisy


def augment_dropout(poses: np.ndarray, drop_prob: float = 0.05) -> np.ndarray:
    """Randomly zero out entire keypoints to simulate occlusion."""
    dropped = poses.copy()
    mask = np.random.random((poses.shape[0], poses.shape[1])) < drop_prob
    dropped[mask] = 0
    return dropped


def get_windows_for_video(poses: np.ndarray, label: int,
                           seq_len: int, stride: int,
                           augment: bool = True,
                           aug_cfg: dict = None) -> tuple:
    """
    Get all windowed samples for a single video, with optional augmentation.
    Only augments training data.
    """
    all_windows = []
    all_labels = []

    # Collect variants
    variants = [poses]

    if augment and aug_cfg:
        if aug_cfg.get("mirror", True):
            variants.append(augment_mirror(poses))

        for factor in aug_cfg.get("speed_factors", [1.0]):
            if factor != 1.0:
                variants.append(augment_speed(poses, factor))
                if aug_cfg.get("mirror", True):
                    variants.append(augment_speed(augment_mirror(poses), factor))

    for variant in variants:
        # Base windows
        windows = create_sliding_windows(variant, seq_len, stride)
        all_windows.append(windows)
        all_labels.append(np.full(len(windows), label))

        if augment and aug_cfg:
            # Noisy version
            if aug_cfg.get("noise_std", 0) > 0:
                noisy = augment_noise(variant, aug_cfg["noise_std"])
                w = create_sliding_windows(noisy, seq_len, stride)
                all_windows.append(w)
                all_labels.append(np.full(len(w), label))

            # Dropout version
            if aug_cfg.get("random_drop_prob", 0) > 0:
                dropped = augment_dropout(variant, aug_cfg["random_drop_prob"])
                w = create_sliding_windows(dropped, seq_len, stride)
                all_windows.append(w)
                all_labels.append(np.full(len(w), label))

    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X, y


def prepare_dataset(poses_dir: str, output_dir: str,
                     config_path: str = "configs/config.yaml"):
    """
    Full dataset preparation with VIDEO-LEVEL splitting.

    Split strategy:
      - Group videos by label (fall vs ADL)
      - Within each group, assign 70% train / 15% val / 15% test
      - Only augment training videos
      - This ensures NO windows from the same video appear in multiple splits
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

    # Group by label
    fall_videos = [m for m in metadata if m["label"] == "fall"]
    adl_videos = [m for m in metadata if m["label"] == "adl"]

    print(f"Found {len(fall_videos)} fall videos, {len(adl_videos)} ADL videos")

    # Shuffle deterministically
    random.seed(42)
    random.shuffle(fall_videos)
    random.shuffle(adl_videos)

    # Split at video level: 70/15/15
    def split_videos(videos):
        n = len(videos)
        n_train = max(1, int(0.7 * n))
        n_val = max(1, int(0.15 * n))
        # rest goes to test
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

    # Process each split
    def process_split(video_metas, augment: bool, split_name: str):
        all_X, all_y = [], []

        for meta in video_metas:
            video_name = Path(meta["video"]).stem
            pose_file = poses_path / f"{video_name}_poses.npy"

            if not pose_file.exists():
                print(f"  Skipping {video_name} — not found")
                continue

            poses = np.load(pose_file)
            label = meta["label_id"]

            # Use larger stride for val/test (no augmentation needed)
            stride = lstm_cfg["stride"] if augment else lstm_cfg["sequence_length"]

            X, y = get_windows_for_video(
                poses, label,
                seq_len=lstm_cfg["sequence_length"],
                stride=stride,
                augment=augment,
                aug_cfg=aug_cfg if augment else None,
            )
            all_X.append(X)
            all_y.append(y)

        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)

        # Shuffle within split
        idx = np.random.permutation(len(X))
        X, y = X[idx], y[idx]

        print(f"  {split_name}: {len(X)} samples "
              f"(falls: {(y==1).sum()}, ADL: {(y==0).sum()})")
        return X, y

    print("\nProcessing splits...")
    X_train, y_train = process_split(train_videos, augment=True, split_name="Train")
    X_val, y_val = process_split(val_videos, augment=False, split_name="Val")
    X_test, y_test = process_split(test_videos, augment=False, split_name="Test")

    # Compute normalization stats from training set ONLY
    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)
    std[std < 1e-6] = 1.0

    # Save
    np.savez_compressed(output_path / "train.npz", X=X_train, y=y_train)
    np.savez_compressed(output_path / "val.npz", X=X_val, y=y_val)
    np.savez_compressed(output_path / "test.npz", X=X_test, y=y_test)
    np.savez(output_path / "norm_stats.npz", mean=mean, std=std)

    # Save split info for reproducibility
    split_info = {
        "train_videos": [m["video"] for m in train_videos],
        "val_videos": [m["video"] for m in val_videos],
        "test_videos": [m["video"] for m in test_videos],
    }
    with open(output_path / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"\nSaved to {output_path}")
    print(f"Split info saved to {output_path / 'split_info.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare LSTM dataset")
    parser.add_argument("--poses-dir", default="data/poses")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    prepare_dataset(args.poses_dir, args.output_dir, args.config)