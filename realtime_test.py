"""
realtime_test.py
================
ISL Recognition System — Real-Time Webcam Inference (Hand-Only)

Uses OpenCV + MediaPipe HandLandmarker to detect hands and extract
keypoints, then feeds a 30-frame sliding window into a trained LSTM
model to predict ISL signs in real time.

Usage
-----
    python realtime_test.py

Controls
--------
    q  — Quit the application

Requirements
------------
    - isl_lstm_model.h5  (run model_training.py first)
    - hand_landmarker.task  (in project root)
    - NP_Data/  (for class label discovery)
"""

import os
import sys
import time
import json
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model  # type: ignore

# ─────────────────────────── Configuration ───────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "isl_lstm_model.h5")
DATA_PATH = os.path.join(BASE_DIR, "NP_Data_Combined")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "label_map.json")
HAND_MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")

SEQUENCE_LENGTH = 30            # Must match training
NUM_FEATURES = 126              # 21 landmarks × 3 coords × 2 hands
CONFIDENCE_THRESHOLD = 0.60     # Min confidence to display prediction
CAMERA_INDEX = 0                # Webcam device index

# Display
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.0
FONT_THICKNESS = 2
BAR_WIDTH = 300
BAR_HEIGHT = 25

# Hand connections for drawing
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),         # Thumb
    (0,5),(5,6),(6,7),(7,8),         # Index
    (0,9),(9,10),(10,11),(11,12),    # Middle
    (0,13),(13,14),(14,15),(15,16),  # Ring
    (0,17),(17,18),(18,19),(19,20),  # Pinky
    (5,9),(9,13),(13,17),            # Palm
]


# ─────────────────────────── Helper Functions ────────────────────────

def get_actions(data_path: str = DATA_PATH) -> np.ndarray:
    """Load class names from label_map.json or directory scan."""
    # Prefer label_map.json for consistent ordering
    if os.path.isfile(LABEL_MAP_PATH):
        with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
            label_map = json.load(f)
        actions = sorted(label_map.keys(), key=lambda k: label_map[k])
        return np.array(actions)

    # Fallback: scan directories
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    actions = sorted([
        d for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d))
    ])
    if not actions:
        raise ValueError(f"No class subdirectories found in {data_path}")
    return np.array(actions)


def create_hand_landmarker():
    """Create a HandLandmarker in VIDEO mode."""
    if not os.path.isfile(HAND_MODEL_PATH):
        print(f"[ERROR] hand_landmarker.task not found at: {HAND_MODEL_PATH}")
        sys.exit(1)

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)

def extract_hand_keypoints(hand_result) -> np.ndarray:
    """
    Extract a flat 126-dim keypoint vector from HandLandmarker result.
    Matches the normalized training data format.
    """
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


def draw_hands(image, hand_result):
    """Draw hand landmarks and connections on the frame."""
    h, w, _ = image.shape

    if not hand_result.hand_landmarks:
        return image

    colors = {
        "left":  ((245, 117, 66), (245, 66, 230)),   # Orange points, pink lines
        "right": ((121, 22, 76),  (121, 44, 250)),    # Purple points, blue lines
    }

    for hand_lms, handedness_list in zip(
        hand_result.hand_landmarks, hand_result.handedness
    ):
        label = handedness_list[0].category_name.lower() if handedness_list else "left"
        pt_color, ln_color = colors.get(label, colors["left"])

        # Convert to pixel coordinates (mirror x for flipped display)
        pts = [(w - 1 - int(lm.x * w), int(lm.y * h)) for lm in hand_lms]

        # Draw connections
        for start, end in HAND_CONNECTIONS:
            if start < len(pts) and end < len(pts):
                cv2.line(image, pts[start], pts[end], ln_color, 2)

        # Draw landmarks
        for px, py in pts:
            cv2.circle(image, (px, py), 4, pt_color, -1)
            cv2.circle(image, (px, py), 5, (255, 255, 255), 1)

    return image


def draw_hud(image, prediction, confidence, fps, state, capture_progress=0):
    """Draw prediction overlay, confidence bar, FPS, and capture status."""
    h, w, _ = image.shape

    # Semi-transparent top bar
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (30, 30, 30), -1)
    image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)

    # Prediction text
    cv2.putText(image, f"Sign: {prediction}", (15, 35), FONT, FONT_SCALE,
                (0, 255, 128), FONT_THICKNESS, cv2.LINE_AA)

    # Confidence bar
    bar_x, bar_y = 15, 52
    filled = int(BAR_WIDTH * confidence)
    cv2.rectangle(image, (bar_x, bar_y),
                  (bar_x + BAR_WIDTH, bar_y + BAR_HEIGHT), (60, 60, 60), -1)
    color = (int(max(0, 255 * (1 - confidence))),
             int(min(255, 255 * confidence)), 0)
    cv2.rectangle(image, (bar_x, bar_y),
                  (bar_x + filled, bar_y + BAR_HEIGHT), color, -1)
    cv2.putText(image, f"{int(confidence * 100)}%",
                (bar_x + BAR_WIDTH + 10, bar_y + 20),
                FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # FPS
    cv2.putText(image, f"FPS: {fps:.0f}", (w - 130, 30),
                FONT, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

    # State-dependent bottom bar
    if state == "idle":
        cv2.putText(image, "[C] Capture   [Q] Quit", (15, h - 20),
                    FONT, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
    elif state == "capturing":
        # Recording progress bar
        prog_y = h - 35
        prog_w = w - 30
        progress = capture_progress / CAPTURE_FRAMES
        cv2.rectangle(image, (15, prog_y), (15 + prog_w, prog_y + 18),
                      (60, 60, 60), -1)
        cv2.rectangle(image, (15, prog_y),
                      (15 + int(prog_w * progress), prog_y + 18),
                      (0, 0, 255), -1)

        # REC indicator
        cv2.circle(image, (w - 25, 70), 8, (0, 0, 255), -1)
        cv2.putText(image, "REC", (w - 60, 76),
                    FONT, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

        pct = int(progress * 100)
        cv2.putText(image, f"Capturing... {pct}%", (15, h - 10),
                    FONT, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    elif state == "predicting":
        cv2.putText(image, "Predicting...", (15, h - 20),
                    FONT, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

    return image


# ─────────────────────────── Configuration ───────────────────────────

CAPTURE_DURATION = 2.0                      # Seconds to capture
CAPTURE_FRAMES = int(CAPTURE_DURATION * 30) # ~60 frames at 30fps


# ─────────────────────────── Main Loop ───────────────────────────────

def main() -> None:
    """ISL recognition — capture-then-predict mode."""
    # ── Load LSTM model ──
    print("[INFO] Loading LSTM model …")
    if not os.path.isfile(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("       Run 'python model_training.py' first.")
        sys.exit(1)

    model = load_model(MODEL_PATH)
    actions = get_actions()
    print(f"[INFO] Model loaded. Classes: {len(actions)}, Features: {NUM_FEATURES}")

    # ── Warm-up inference ──
    dummy = np.zeros((1, SEQUENCE_LENGTH, NUM_FEATURES), dtype=np.float32)
    _ = model.predict(dummy, verbose=0)
    print("[INFO] Model warm-up complete.")

    # ── MediaPipe hand landmarker ──
    hand_lm = create_hand_landmarker()
    print("[INFO] HandLandmarker ready.")

    # ── Webcam ──
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera (index {CAMERA_INDEX}).")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("[INFO] Webcam started.")
    print("[INFO] Press 'c' to capture, 'q' to quit.\n")

    # ── State ──
    state = "idle"               # idle | capturing | predicting
    capture_buffer: list[np.ndarray] = []
    current_prediction = "Press 'C' to capture"
    current_confidence = 0.0
    prev_time = time.time()
    frame_ts_ms = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            # FPS
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            # ── Detect hands on RAW (unflipped) frame ──
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            frame_ts_ms += 33  # ~30 fps monotonic
            hand_result = hand_lm.detect_for_video(mp_image, frame_ts_ms)

            # ── Flip frame for natural mirror display ──
            frame = cv2.flip(frame, 1)

            # ── Draw hand landmarks ──
            frame = draw_hands(frame, hand_result)

            # ── State machine ──
            if state == "capturing":
                # Collect keypoints during capture window
                keypoints = extract_hand_keypoints(hand_result)
                capture_buffer.append(keypoints)

                if len(capture_buffer) >= CAPTURE_FRAMES:
                    state = "predicting"

            elif state == "predicting":
                # Downsample captured frames to SEQUENCE_LENGTH
                indices = np.linspace(
                    0, len(capture_buffer) - 1, SEQUENCE_LENGTH, dtype=int
                )
                sequence = [capture_buffer[i] for i in indices]

                input_data = np.expand_dims(
                    np.array(sequence, dtype=np.float32), axis=0
                )
                prediction = model.predict(input_data, verbose=0)[0]
                class_idx = int(np.argmax(prediction))
                confidence = float(prediction[class_idx])

                if confidence >= CONFIDENCE_THRESHOLD:
                    current_prediction = actions[class_idx]
                    current_confidence = confidence
                else:
                    current_prediction = f"Uncertain ({actions[class_idx]})"
                    current_confidence = confidence

                print(f"  → Predicted: {current_prediction}  "
                      f"({current_confidence*100:.1f}%)")

                capture_buffer = []
                state = "idle"

            # ── Draw HUD ──
            frame = draw_hud(
                frame, current_prediction, current_confidence,
                fps, state, len(capture_buffer),
            )

            cv2.imshow("ISL Recognition - Press C to Capture, Q to Quit", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("c") and state == "idle":
                state = "capturing"
                capture_buffer = []
                current_prediction = "Capturing..."
                current_confidence = 0.0
                print("[CAPTURE] Recording for 2 seconds...")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hand_lm.close()
        print("[INFO] Resources released. Goodbye!")


if __name__ == "__main__":
    main()
