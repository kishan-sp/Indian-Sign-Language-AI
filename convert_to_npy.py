"""
convert_to_npy.py
=================
ISL Recognition System — Unified Data Conversion Pipeline

Converts ALL data sources into a single NP_Data_Combined/ folder
of 126-dim hand keypoint .npy sequences, ready for LSTM training.

Supports three input types:
  1. Static images  (dataset - Gesture Speech → grouped into 30-frame sequences)
  2. Raw videos     (archive/1..11 → hand keypoints extracted per frame)
  3. Existing .npy  (NP_Data/ → sliced to hand-only 126-dim)

Usage
-----
    python convert_to_npy.py                  # Convert all sources
    python convert_to_npy.py --images-only    # Only process image dataset
    python convert_to_npy.py --videos-only    # Only process video dataset
    python convert_to_npy.py --npy-only       # Only migrate NP_Data

Output
------
    NP_Data_Combined/{sign_name}/{seq_num}/0.npy … 29.npy
    label_map.json
"""

import os
import sys
import json
import shutil
import argparse
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ─────────────────────────── Configuration ───────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARCHIVE_DIR = os.path.join(BASE_DIR, "archive")
IMAGE_DATASET_DIR = os.path.join(ARCHIVE_DIR, "dataset - Gesture Speech")
NP_Data_DIR = os.path.join(BASE_DIR, "NP_Data")
COMBINED_DIR = os.path.join(BASE_DIR, "NP_Data_Combined")
HAND_MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "label_map.json")

SEQUENCE_LENGTH = 30        # Frames per sequence
NUM_FEATURES = 126          # 21 × 3 × 2 hands
HAND_START_IDX = 1536       # Index range in original 1662-dim .npy
HAND_END_IDX = 1662

# Video class → sign name mapping
VIDEO_CLASS_MAP = {
    "1":  "hello",
    "2":  "bye",
    "3":  "morning",
    "4":  "good",
    "5":  "nice",
    "6":  "house",
    "7":  "thank you",
    "8":  "welcome",
    "9":  "yes",
    "10": "no",
    "11": "work",
}

# Image class → sign name mapping (rename special folder names)
# The `{` folder likely represents "nothing" / blank
IMAGE_CLASS_MAP = {
    "{": "nothing",  # Rename curly-brace folder
    # All a-z folders keep their lowercase name as-is
}


# ─────────────────────────── MediaPipe ───────────────────────────────

def create_hand_landmarker(mode="IMAGE"):
    """Create a HandLandmarker in IMAGE or VIDEO mode."""
    if not os.path.isfile(HAND_MODEL_PATH):
        print(f"[ERROR] hand_landmarker.task not found: {HAND_MODEL_PATH}")
        sys.exit(1)

    running_mode = (vision.RunningMode.IMAGE if mode == "IMAGE"
                    else vision.RunningMode.VIDEO)

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=running_mode,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def extract_hand_kp_from_image(hand_lm, image_rgb: np.ndarray) -> np.ndarray:
    """Extract 126-dim from a single image (IMAGE mode)."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = hand_lm.detect(mp_image)
    return _parse_hand_result(result)


def extract_hand_kp_from_frame(
    hand_lm, frame_rgb: np.ndarray, ts_ms: int
) -> np.ndarray:
    """Extract 126-dim from a video frame (VIDEO mode)."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = hand_lm.detect_for_video(mp_image, ts_ms)
    return _parse_hand_result(result)


def _parse_hand_result(result) -> np.ndarray:
    """Parse HandLandmarker result → 126-dim vector."""
    left_hand = np.zeros(63)
    right_hand = np.zeros(63)

    if result.hand_landmarks and result.handedness:
        for hand_lms, handedness_list in zip(
            result.hand_landmarks, result.handedness
        ):
            label = (handedness_list[0].category_name.lower()
                     if handedness_list else "")
            coords = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_lms[:21]]
            ).flatten()

            if label == "left":
                left_hand = coords
            elif label == "right":
                right_hand = coords

    return np.concatenate([left_hand, right_hand])


# ─────────────────────────── Helpers ─────────────────────────────────

def get_next_seq_index(sign_dir: str) -> int:
    """Find the next available sequence number."""
    if not os.path.isdir(sign_dir):
        return 0
    existing = [
        int(d) for d in os.listdir(sign_dir)
        if os.path.isdir(os.path.join(sign_dir, d)) and d.isdigit()
    ]
    return max(existing, default=-1) + 1


def save_sequence(out_dir: str, seq_idx: int, frames: list[np.ndarray]) -> str:
    """Save a list of keypoint arrays as a numbered sequence folder."""
    seq_dir = os.path.join(out_dir, str(seq_idx))
    os.makedirs(seq_dir, exist_ok=True)
    for i, kp in enumerate(frames):
        np.save(os.path.join(seq_dir, f"{i}.npy"), kp)
    return seq_dir


def resample(keypoints: list[np.ndarray], target: int) -> list[np.ndarray]:
    """Resample a keypoint list to exactly `target` frames."""
    n = len(keypoints)
    if n == 0:
        return [np.zeros(NUM_FEATURES) for _ in range(target)]
    if n == target:
        return keypoints
    if n < target:
        out = list(keypoints)
        while len(out) < target:
            out.append(keypoints[-1].copy())
        return out
    indices = np.linspace(0, n - 1, target, dtype=int)
    return [keypoints[i] for i in indices]


# ═══════════════════════ SOURCE 1: IMAGES ════════════════════════════

def process_images(hand_lm) -> dict[str, int]:
    """
    Process static image dataset (dataset - Gesture Speech).

    Strategy: For each class, extract hand keypoints from each image.
    Group every SEQUENCE_LENGTH images into one sequence.
    Since these are static sign poses (same gesture = same frame),
    each image produces a constant 30-frame sequence (replicated).

    Returns dict of {sign_name: num_sequences_created}.
    """
    stats: dict[str, int] = {}

    if not os.path.isdir(IMAGE_DATASET_DIR):
        print(f"[SKIP] Image dataset not found: {IMAGE_DATASET_DIR}")
        return stats

    class_folders = sorted(os.listdir(IMAGE_DATASET_DIR))
    print(f"\n  Found {len(class_folders)} image classes")

    for class_name in class_folders:
        class_path = os.path.join(IMAGE_DATASET_DIR, class_name)
        if not os.path.isdir(class_path):
            continue

        # Map class name
        sign_name = IMAGE_CLASS_MAP.get(class_name, class_name.lower())

        # Gather all image files
        images = sorted([
            f for f in os.listdir(class_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ])

        if not images:
            print(f"  [{sign_name}] No images — skipping.")
            continue

        # Output directory
        out_dir = os.path.join(COMBINED_DIR, sign_name)
        os.makedirs(out_dir, exist_ok=True)
        seq_idx = get_next_seq_index(out_dir)
        seq_count = 0

        print(f"  [{sign_name}] {len(images)} images → ", end="", flush=True)

        # Process each image → replicate into a constant 30-frame sequence
        for img_name in images:
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            kp = extract_hand_kp_from_image(hand_lm, rgb)

            # Replicate the same keypoints for all 30 frames
            # (static gesture → constant sequence)
            frames = [kp.copy() for _ in range(SEQUENCE_LENGTH)]
            save_sequence(out_dir, seq_idx, frames)

            seq_idx += 1
            seq_count += 1

        stats[sign_name] = seq_count
        print(f"{seq_count} sequences created")

    return stats


# ═══════════════════════ SOURCE 2: VIDEOS ════════════════════════════

def process_videos(hand_lm) -> dict[str, int]:
    """
    Process video files from archive/1..11 folders.

    Returns dict of {sign_name: num_sequences_created}.
    """
    stats: dict[str, int] = {}

    if not os.path.isdir(ARCHIVE_DIR):
        print(f"[SKIP] archive/ not found: {ARCHIVE_DIR}")
        return stats

    class_folders = sorted([
        d for d in os.listdir(ARCHIVE_DIR)
        if os.path.isdir(os.path.join(ARCHIVE_DIR, d)) and d.isdigit()
    ])

    if not class_folders:
        print("  [SKIP] No numbered video class folders found.")
        return stats

    running_ts = 0  # Monotonic timestamp for VIDEO mode

    for class_id in class_folders:
        class_path = os.path.join(ARCHIVE_DIR, class_id)
        sign_name = VIDEO_CLASS_MAP.get(class_id)
        if sign_name is None:
            print(f"  [WARN] Unknown class '{class_id}' — skipping.")
            continue

        out_dir = os.path.join(COMBINED_DIR, sign_name)
        os.makedirs(out_dir, exist_ok=True)
        seq_idx = get_next_seq_index(out_dir)
        video_count = 0

        print(f"  [{sign_name}] (folder {class_id}) → ", end="", flush=True)

        # Walk person subfolders
        for person in sorted(os.listdir(class_path)):
            person_path = os.path.join(class_path, person)
            if not os.path.isdir(person_path):
                continue

            videos = sorted([
                f for f in os.listdir(person_path)
                if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
            ])

            for video_file in videos:
                video_path = os.path.join(person_path, video_file)
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    continue

                raw_kps: list[np.ndarray] = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    running_ts += 33
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    kp = extract_hand_kp_from_frame(hand_lm, rgb, running_ts)
                    raw_kps.append(kp)
                cap.release()

                if not raw_kps:
                    continue

                frames = resample(raw_kps, SEQUENCE_LENGTH)
                save_sequence(out_dir, seq_idx, frames)
                seq_idx += 1
                video_count += 1

        stats[sign_name] = video_count
        print(f"{video_count} sequences")

    return stats


# ═══════════════════════ SOURCE 3: NP_Data ═══════════════════════════

def _find_seq_dirs(action_dir: str) -> list[str]:
    """Locate numbered sequence folders (handles nested layout)."""
    children = sorted(os.listdir(action_dir))
    child_dirs = [
        c for c in children
        if os.path.isdir(os.path.join(action_dir, c))
    ]

    numeric = [c for c in child_dirs if c.isdigit()]
    if numeric:
        return [os.path.join(action_dir, d) for d in sorted(numeric, key=int)]

    # Check one level deeper
    for sub in child_dirs:
        sub_path = os.path.join(action_dir, sub)
        grandchildren = [
            g for g in os.listdir(sub_path)
            if os.path.isdir(os.path.join(sub_path, g)) and g.isdigit()
        ]
        if grandchildren:
            return [
                os.path.join(sub_path, d)
                for d in sorted(grandchildren, key=int)
            ]
    return []


def process_NP_Data() -> list[str]:
    """
    Migrate NP_Data/ keypoints → NP_Data_Combined/ (hand-only slice).

    Returns list of migrated class names.
    """
    if not os.path.isdir(NP_Data_DIR):
        print(f"[SKIP] NP_Data/ not found: {NP_Data_DIR}")
        return []

    actions = sorted([
        d for d in os.listdir(NP_Data_DIR)
        if os.path.isdir(os.path.join(NP_Data_DIR, d))
    ])

    migrated: list[str] = []

    for action in actions:
        action_dir = os.path.join(NP_Data_DIR, action)
        out_dir = os.path.join(COMBINED_DIR, action)
        os.makedirs(out_dir, exist_ok=True)

        seq_idx = get_next_seq_index(out_dir)
        seq_dirs = _find_seq_dirs(action_dir)

        if not seq_dirs:
            continue

        for src_seq in seq_dirs:
            dst_seq = os.path.join(out_dir, str(seq_idx))
            os.makedirs(dst_seq, exist_ok=True)

            for f_num in range(SEQUENCE_LENGTH):
                src = os.path.join(src_seq, f"{f_num}.npy")
                dst = os.path.join(dst_seq, f"{f_num}.npy")
                try:
                    full_kp = np.load(src)
                    hand_kp = (full_kp[HAND_START_IDX:HAND_END_IDX]
                               if len(full_kp) >= HAND_END_IDX
                               else np.zeros(NUM_FEATURES))
                    np.save(dst, hand_kp)
                except FileNotFoundError:
                    np.save(dst, np.zeros(NUM_FEATURES))

            seq_idx += 1

        migrated.append(action)
        print(f"  [{action}] {len(seq_dirs)} sequences migrated")

    return migrated


# ─────────────────────────── Label Map ───────────────────────────────

def build_label_map() -> dict[str, int]:
    """Build and save label_map.json from NP_Data_Combined/."""
    classes = sorted([
        d for d in os.listdir(COMBINED_DIR)
        if os.path.isdir(os.path.join(COMBINED_DIR, d))
    ])
    label_map = {name: idx for idx, name in enumerate(classes)}

    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)

    return label_map


# ─────────────────────────── Main ────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert images, videos, and .npy data to unified keypoint format."
    )
    parser.add_argument("--images-only", action="store_true",
                        help="Only process the image dataset")
    parser.add_argument("--videos-only", action="store_true",
                        help="Only process the video dataset")
    parser.add_argument("--npy-only", action="store_true",
                        help="Only migrate NP_Data")
    parser.add_argument("--no-clean", action="store_true",
                        help="Don't delete existing NP_Data_Combined/")
    args = parser.parse_args()

    do_all = not (args.images_only or args.videos_only or args.npy_only)

    print("=" * 60)
    print("  ISL — Unified Data Conversion Pipeline")
    print("=" * 60)

    # Clean output directory (unless --no-clean)
    if not args.no_clean and do_all:
        if os.path.isdir(COMBINED_DIR):
            print(f"\n[INFO] Removing existing {COMBINED_DIR} …")
            shutil.rmtree(COMBINED_DIR)
    os.makedirs(COMBINED_DIR, exist_ok=True)

    total_seqs = 0

    # ── Source 1: Images ──
    if do_all or args.images_only:
        print("\n" + "─" * 55)
        print("  SOURCE 1: Static Images (dataset - Gesture Speech)")
        print("─" * 55)
        hand_lm_img = create_hand_landmarker(mode="IMAGE")
        img_stats = process_images(hand_lm_img)
        hand_lm_img.close()
        img_total = sum(img_stats.values())
        total_seqs += img_total
        print(f"\n  → {img_total} sequences from {len(img_stats)} image classes")

    # ── Source 2: Videos ──
    if do_all or args.videos_only:
        print("\n" + "─" * 55)
        print("  SOURCE 2: Video Files (archive/1..11)")
        print("─" * 55)
        hand_lm_vid = create_hand_landmarker(mode="VIDEO")
        vid_stats = process_videos(hand_lm_vid)
        hand_lm_vid.close()
        vid_total = sum(vid_stats.values())
        total_seqs += vid_total
        print(f"\n  → {vid_total} sequences from {len(vid_stats)} video classes")

    # ── Source 3: NP_Data ──
    if do_all or args.npy_only:
        print("\n" + "─" * 55)
        print("  SOURCE 3: NP_Data/ (existing .npy keypoints)")
        print("─" * 55)
        migrated = process_NP_Data()
        # Count what was migrated
        npy_total = 0
        for cls_name in migrated:
            cls_dir = os.path.join(COMBINED_DIR, cls_name)
            if os.path.isdir(cls_dir):
                npy_total += sum(
                    1 for d in os.listdir(cls_dir)
                    if os.path.isdir(os.path.join(cls_dir, d)) and d.isdigit()
                )
        total_seqs += npy_total
        print(f"\n  → {len(migrated)} classes migrated from NP_Data/")

    # ── Build label map ──
    print("\n" + "─" * 55)
    print("  Building label map")
    print("─" * 55)
    label_map = build_label_map()

    # ── Count final totals ──
    final_seqs = 0
    class_summary = []
    for cls_name in sorted(label_map.keys()):
        cls_dir = os.path.join(COMBINED_DIR, cls_name)
        if os.path.isdir(cls_dir):
            count = sum(
                1 for d in os.listdir(cls_dir)
                if os.path.isdir(os.path.join(cls_dir, d)) and d.isdigit()
            )
            final_seqs += count
            class_summary.append((cls_name, count))

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Output        : {COMBINED_DIR}")
    print(f"  Label map     : {LABEL_MAP_PATH}")
    print(f"  Total classes : {len(label_map)}")
    print(f"  Total sequences: {final_seqs}")
    print(f"  Seq length    : {SEQUENCE_LENGTH}")
    print(f"  Features/frame: {NUM_FEATURES}")

    # Class breakdown
    print(f"\n  {'Class':<20} {'Sequences':>10}")
    print(f"  {'─'*20} {'─'*10}")
    for name, count in class_summary:
        print(f"  {name:<20} {count:>10}")

    print(f"\n✅  Done. Run `python model_training.py` to train.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Interrupted by user.")
        sys.exit(0)
