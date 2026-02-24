"""
data_preprocessing.py
=====================
ISL Recognition System — Data Preprocessing Module (Combined Dataset)

Loads hand-only keypoint sequences from NP_Data_Combined/ (126-dim),
encodes labels using label_map.json, and prepares train/test splits
for LSTM training.

Works with the unified dataset produced by video_to_keypoints.py.
"""

import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical  # type: ignore

# ─────────────────────────── Configuration ───────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "NP_Data_Combined")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "label_map.json")

SEQUENCE_LENGTH = 30        # Frames per sequence
NUM_FEATURES = 126          # 21 × 3 × 2 hands (hand-only)
TEST_SIZE = 0.10            # Fraction for testing
RANDOM_STATE = 42           # Reproducibility seed


# ─────────────────────────── Helper Functions ────────────────────────

def get_actions(data_path: str = DATA_PATH) -> np.ndarray:
    """
    Load sorted action names from the data directory or label_map.json.
    """
    # Prefer label_map.json for consistent ordering
    if os.path.isfile(LABEL_MAP_PATH):
        with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
            label_map = json.load(f)
        # Sort by index to get ordered list
        actions = sorted(label_map.keys(), key=lambda k: label_map[k])
        print(f"[INFO] Loaded {len(actions)} classes from label_map.json")
        return np.array(actions)

    # Fallback: scan directories
    if not os.path.isdir(data_path):
        raise FileNotFoundError(
            f"Data directory not found: {data_path}\n"
            "Run video_to_keypoints.py first to create it."
        )

    actions = sorted([
        d for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d))
    ])

    if len(actions) == 0:
        raise ValueError(f"No class subdirectories found in {data_path}")

    print(f"[INFO] Discovered {len(actions)} sign classes from directories")
    return np.array(actions)


def _find_sequence_dirs(action_dir: str) -> list[str]:
    """Locate numbered sequence folders inside an action directory."""
    children = sorted(os.listdir(action_dir))
    child_dirs = [
        c for c in children
        if os.path.isdir(os.path.join(action_dir, c))
    ]

    numeric_dirs = [c for c in child_dirs if c.isdigit()]
    if numeric_dirs:
        return [os.path.join(action_dir, d) for d in sorted(numeric_dirs, key=int)]

    # Check one level deeper for nested layout
    for sub in child_dirs:
        sub_path = os.path.join(action_dir, sub)
        grandchildren = [
            g for g in os.listdir(sub_path)
            if os.path.isdir(os.path.join(sub_path, g)) and g.isdigit()
        ]
        if grandchildren:
            print(f"[INFO] Nested layout: {action_dir}/{sub}/")
            return [os.path.join(sub_path, d) for d in sorted(grandchildren, key=int)]

    return []


def load_data(
    data_path: str = DATA_PATH,
    sequence_length: int = SEQUENCE_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load all keypoint sequences and their labels from disk.
    Expects 126-dim hand-only .npy files in NP_Data_Combined/.

    Returns
    -------
    sequences : np.ndarray, shape (N, sequence_length, 126)
    labels    : np.ndarray, shape (N,)
    """
    actions = get_actions(data_path)
    label_map = {action: idx for idx, action in enumerate(actions)}

    sequences: list[np.ndarray] = []
    labels: list[int] = []
    skipped = 0

    for action in actions:
        action_dir = os.path.join(data_path, action)
        if not os.path.isdir(action_dir):
            print(f"[WARN] Directory not found for '{action}' — skipping.")
            continue

        seq_dirs = _find_sequence_dirs(action_dir)

        if not seq_dirs:
            print(f"[WARN] No sequences found for '{action}' — skipping.")
            continue

        for seq_dir in seq_dirs:
            window: list[np.ndarray] = []

            for frame_num in range(sequence_length):
                frame_path = os.path.join(seq_dir, f"{frame_num}.npy")
                try:
                    kp = np.load(frame_path)
                    # Ensure correct dimension
                    if len(kp) == NUM_FEATURES:
                        window.append(kp)
                    elif len(kp) > NUM_FEATURES:
                        # Trim if needed
                        window.append(kp[:NUM_FEATURES])
                    else:
                        # Pad if needed
                        padded = np.zeros(NUM_FEATURES)
                        padded[:len(kp)] = kp
                        window.append(padded)
                except FileNotFoundError:
                    window.append(np.zeros(NUM_FEATURES))
                    skipped += 1

            sequences.append(np.array(window))
            labels.append(label_map[action])

    if skipped:
        print(f"[WARN] {skipped} missing frames were zero-padded.")

    sequences_arr = np.array(sequences, dtype=np.float32)
    labels_arr = np.array(labels, dtype=np.int32)

    print(f"[INFO] Loaded {sequences_arr.shape[0]} sequences  "
          f"| shape: {sequences_arr.shape}  | labels: {labels_arr.shape}")
    return sequences_arr, labels_arr


# ─────────────────────────── Main Entry Point ────────────────────────

def preprocess(
    data_path: str = DATA_PATH,
    sequence_length: int = SEQUENCE_LENGTH,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline:
      1. Load sequences & labels
      2. One-hot encode labels
      3. Split into train / test sets

    Returns
    -------
    X_train, X_test, y_train, y_test, actions
    """
    sequences, labels = load_data(data_path, sequence_length)

    num_classes = len(np.unique(labels))
    y_encoded = to_categorical(labels, num_classes=num_classes).astype(np.float32)

    # Ensure every class has at least 2 samples for stratified split
    from collections import Counter
    label_counts = Counter(labels.tolist())
    extras_X, extras_y, extras_labels = [], [], []
    for cls_idx, count in label_counts.items():
        if count < 2:
            mask = labels == cls_idx
            extras_X.append(sequences[mask])
            extras_y.append(y_encoded[mask])
            extras_labels.append(labels[mask])
            print(f"[INFO] Class {cls_idx} has only {count} sample(s) — duplicating.")
    if extras_X:
        sequences = np.concatenate([sequences] + extras_X)
        y_encoded = np.concatenate([y_encoded] + extras_y)
        labels = np.concatenate([labels] + extras_labels)

    X_train, X_test, y_train, y_test = train_test_split(
        sequences, y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    print(f"[INFO] Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")
    print(f"[INFO] X shape: {X_train.shape}  |  y shape: {y_train.shape}")

    actions = get_actions(data_path)
    return X_train, X_test, y_train, y_test, actions


# ─────────────────────────── Stand-alone Usage ───────────────────────

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, actions = preprocess()
    print(f"\n✅  Preprocessing complete.")
    print(f"   Classes       : {len(actions)}")
    print(f"   Features/frame: {NUM_FEATURES}")
    print(f"   X_train       : {X_train.shape}")
    print(f"   X_test        : {X_test.shape}")