"""
collect_data.py
===============
ISL Recognition System — Interactive Data Collection Tool

Captures hand keypoint sequences from webcam and saves them as .npy
files into MP_Data_Combined/ for LSTM training.

Usage
-----
    python collect_data.py

Controls
--------
    Type sign name + Enter  — Set the current sign label
    s                       — Start recording a 30-frame sequence
    n                       — Change to a new sign name
    q                       — Quit

Requirements
------------
    - hand_landmarker.task (in project root)
    - opencv-python, mediapipe, numpy
"""

import os
import sys
import json
import time
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ─────────────────────────── Configuration ───────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMBINED_DIR = os.path.join(BASE_DIR, "MP_Data_Combined")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "label_map.json")
HAND_MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")

SEQUENCE_LENGTH = 30    # Frames per sequence (must match training)
NUM_FEATURES = 126      # 21 landmarks × 3 coords × 2 hands
CAMERA_INDEX = 0
COUNTDOWN_SECS = 3      # Countdown before recording starts

# Display
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Hand skeleton connections
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# Colors
GREEN = (0, 255, 128)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
GRAY = (180, 180, 180)
DARK_BG = (30, 30, 30)


# ─────────────────────────── MediaPipe Setup ─────────────────────────

def create_hand_landmarker():
    """Create HandLandmarker in VIDEO mode."""
    if not os.path.isfile(HAND_MODEL_PATH):
        print(f"[ERROR] hand_landmarker.task not found: {HAND_MODEL_PATH}")
        sys.exit(1)

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


# ─────────────────────────── Keypoint Extraction ─────────────────────

def extract_hand_keypoints(hand_result) -> np.ndarray:
    """Extract 126-dim hand keypoints from HandLandmarker result."""
    left_hand = np.zeros(63)
    right_hand = np.zeros(63)

    if hand_result.hand_landmarks and hand_result.handedness:
        for hand_lms, handedness_list in zip(
            hand_result.hand_landmarks, hand_result.handedness
        ):
            label = handedness_list[0].category_name.lower() if handedness_list else ""
            raw = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms[:21]])

            # ✅ Normalize: Relative to wrist and scale-invariant
            wrist = raw[0]
            relative = (raw - wrist)
            max_dist = np.max(np.linalg.norm(relative, axis=1))
            if max_dist > 0:
                relative /= max_dist

            coords = relative.flatten()
            if label == "left":
                left_hand = coords
            elif label == "right":
                right_hand = coords

    return np.concatenate([left_hand, right_hand])

 

# ─────────────────────────── Drawing Helpers ─────────────────────────

def draw_hands(image, hand_result):
    """Draw hand landmarks on the frame."""
    h, w, _ = image.shape
    if not hand_result.hand_landmarks:
        return image

    colors = {
        "left":  ((245, 117, 66), (245, 66, 230)),
        "right": ((121, 22, 76),  (121, 44, 250)),
    }

    for hand_lms, handedness_list in zip(
        hand_result.hand_landmarks, hand_result.handedness
    ):
        label = handedness_list[0].category_name.lower() if handedness_list else "left"
        pt_color, ln_color = colors.get(label, colors["left"])
        pts = [(w - 1 - int(lm.x * w), int(lm.y * h)) for lm in hand_lms]

        for start, end in HAND_CONNECTIONS:
            if start < len(pts) and end < len(pts):
                cv2.line(image, pts[start], pts[end], ln_color, 2)
        for px, py in pts:
            cv2.circle(image, (px, py), 4, pt_color, -1)
            cv2.circle(image, (px, py), 5, WHITE, 1)

    return image


def draw_status_bar(image, sign_name, seq_count, state, frame_num=0):
    """Draw a status overlay at the top of the frame."""
    h, w, _ = image.shape

    # Dark overlay
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), DARK_BG, -1)
    image = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)

    # Sign name
    if sign_name:
        cv2.putText(image, f"Sign: {sign_name}", (15, 30),
                    FONT, 0.8, GREEN, 2, cv2.LINE_AA)
        cv2.putText(image, f"Sequences: {seq_count}", (15, 58),
                    FONT, 0.6, WHITE, 1, cv2.LINE_AA)
    else:
        cv2.putText(image, "No sign selected", (15, 30),
                    FONT, 0.8, YELLOW, 2, cv2.LINE_AA)
        cv2.putText(image, "Press 'n' to set sign name", (15, 58),
                    FONT, 0.6, GRAY, 1, cv2.LINE_AA)

    # State indicator
    if state == "idle":
        cv2.putText(image, "[S] Record  [N] New sign  [Q] Quit",
                    (15, 88), FONT, 0.5, GRAY, 1, cv2.LINE_AA)
    elif state == "countdown":
        pass  # Countdown drawn separately
    elif state == "recording":
        # Recording indicator + progress bar
        cv2.circle(image, (w - 25, 25), 10, RED, -1)
        cv2.putText(image, "REC", (w - 65, 32),
                    FONT, 0.6, RED, 2, cv2.LINE_AA)

        # Progress bar
        bar_x, bar_y, bar_w, bar_h = 15, 80, w - 30, 12
        progress = frame_num / SEQUENCE_LENGTH
        cv2.rectangle(image, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        cv2.rectangle(image, (bar_x, bar_y),
                      (bar_x + int(bar_w * progress), bar_y + bar_h), GREEN, -1)
        cv2.putText(image, f"{frame_num}/{SEQUENCE_LENGTH}",
                    (bar_x + bar_w + 5, bar_y + 11),
                    FONT, 0.4, WHITE, 1, cv2.LINE_AA)

    return image


def draw_countdown(image, seconds_left):
    """Draw a large countdown number in the center."""
    h, w, _ = image.shape
    text = str(seconds_left)
    text_size = cv2.getTextSize(text, FONT, 4, 6)[0]
    x = (w - text_size[0]) // 2
    y = (h + text_size[1]) // 2

    # Shadow
    cv2.putText(image, text, (x + 3, y + 3), FONT, 4, (0, 0, 0), 8, cv2.LINE_AA)
    # Text
    cv2.putText(image, text, (x, y), FONT, 4, YELLOW, 6, cv2.LINE_AA)

    return image


def draw_saved_flash(image):
    """Flash a green 'SAVED' message."""
    h, w, _ = image.shape
    text = "SAVED!"
    text_size = cv2.getTextSize(text, FONT, 2, 4)[0]
    x = (w - text_size[0]) // 2
    y = (h + text_size[1]) // 2
    cv2.putText(image, text, (x, y), FONT, 2, GREEN, 4, cv2.LINE_AA)
    return image


# ─────────────────────────── Label Map ───────────────────────────────

def update_label_map():
    """Rebuild label_map.json from MP_Data_Combined/ directory."""
    os.makedirs(COMBINED_DIR, exist_ok=True)
    classes = sorted([
        d for d in os.listdir(COMBINED_DIR)
        if os.path.isdir(os.path.join(COMBINED_DIR, d))
    ])

    label_map = {name: idx for idx, name in enumerate(classes)}

    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)

    return label_map


def get_next_seq_index(sign_dir: str) -> int:
    """Find the next available sequence number for a sign."""
    if not os.path.isdir(sign_dir):
        return 0
    existing = [
        int(d) for d in os.listdir(sign_dir)
        if os.path.isdir(os.path.join(sign_dir, d)) and d.isdigit()
    ]
    return max(existing, default=-1) + 1


def count_sequences(sign_dir: str) -> int:
    """Count existing sequences for a sign."""
    if not os.path.isdir(sign_dir):
        return 0
    return sum(
        1 for d in os.listdir(sign_dir)
        if os.path.isdir(os.path.join(sign_dir, d)) and d.isdigit()
    )


# ─────────────────────────── Input Helper ────────────────────────────

def get_sign_name_from_console() -> str:
    """Prompt user for sign name in the console."""
    print("\n" + "─" * 40)
    name = input("  Enter sign name (e.g., 'water'): ").strip().lower()
    if name:
        print(f"  ✓ Sign set to: '{name}'")
        print(f"  Press 's' in the webcam window to start recording.")
    else:
        print("  ✗ Empty name — try again.")
    print("─" * 40)
    return name


# ─────────────────────────── Main Loop ───────────────────────────────

def main() -> None:
    print("=" * 55)
    print("  ISL Data Collection Tool")
    print("=" * 55)
    print()
    print("  Controls:")
    print("    s  — Start recording a sequence")
    print("    n  — Set/change sign name")
    print("    q  — Quit")
    print()

    # ── MediaPipe ──
    hand_lm = create_hand_landmarker()
    print("[INFO] HandLandmarker ready.")

    # ── Webcam ──
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera (index {CAMERA_INDEX}).")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("[INFO] Webcam started.\n")

    # ── State ──
    sign_name = ""
    state = "idle"          # idle | countdown | recording | saved
    frame_ts_ms = 0
    recording_buffer: list[np.ndarray] = []
    countdown_start = 0.0
    saved_flash_start = 0.0
    total_collected = 0

    # Prompt for first sign name
    sign_name = get_sign_name_from_console()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            # ── Detect hands (on raw frame) ──
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            frame_ts_ms += 33
            hand_result = hand_lm.detect_for_video(mp_image, frame_ts_ms)

            # ── Flip for display ──
            frame = cv2.flip(frame, 1)

            # ── Draw hands ──
            frame = draw_hands(frame, hand_result)

            # ── State machine ──
            sign_dir = os.path.join(COMBINED_DIR, sign_name) if sign_name else ""
            seq_count = count_sequences(sign_dir) if sign_dir else 0

            if state == "countdown":
                elapsed = time.time() - countdown_start
                remaining = COUNTDOWN_SECS - int(elapsed)

                if remaining > 0:
                    frame = draw_countdown(frame, remaining)
                else:
                    # Start recording
                    state = "recording"
                    recording_buffer = []
                    print(f"  Recording sequence {seq_count}...", end=" ")

            elif state == "recording":
                # Capture keypoints
                kp = extract_hand_keypoints(hand_result)
                recording_buffer.append(kp)

                frame = draw_status_bar(
                    frame, sign_name, seq_count, "recording",
                    frame_num=len(recording_buffer),
                )

                if len(recording_buffer) >= SEQUENCE_LENGTH:
                    # Save the sequence
                    seq_idx = get_next_seq_index(sign_dir)
                    seq_dir = os.path.join(sign_dir, str(seq_idx))
                    os.makedirs(seq_dir, exist_ok=True)

                    for i, kp_frame in enumerate(recording_buffer[:SEQUENCE_LENGTH]):
                        np.save(os.path.join(seq_dir, f"{i}.npy"), kp_frame)

                    total_collected += 1
                    new_count = count_sequences(sign_dir)
                    print(f"SAVED (seq {seq_idx}, total: {new_count})")

                    # Update label map
                    update_label_map()

                    # Flash "SAVED"
                    state = "saved"
                    saved_flash_start = time.time()
                    recording_buffer = []

            elif state == "saved":
                frame = draw_saved_flash(frame)
                if time.time() - saved_flash_start > 0.5:
                    state = "idle"

            else:  # idle
                frame = draw_status_bar(frame, sign_name, seq_count, "idle")

            # ── Show ──
            cv2.imshow("ISL Data Collection - Press Q to Quit", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("s") and state == "idle" and sign_name:
                # Start countdown
                state = "countdown"
                countdown_start = time.time()
                os.makedirs(sign_dir, exist_ok=True)
                print(f"\n  [{sign_name}] Starting in {COUNTDOWN_SECS}s...")
            elif key == ord("n") and state == "idle":
                sign_name = get_sign_name_from_console()

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hand_lm.close()

        # Final label map update
        label_map = update_label_map()

        print(f"\n{'=' * 55}")
        print(f"  Session Summary")
        print(f"{'=' * 55}")
        print(f"  Sequences recorded : {total_collected}")
        print(f"  Total classes      : {len(label_map)}")
        print(f"  Data directory     : {COMBINED_DIR}")
        print(f"  Label map          : {LABEL_MAP_PATH}")
        print(f"\n  To retrain: python model_training.py")
        print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
