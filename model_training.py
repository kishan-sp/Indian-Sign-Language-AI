"""
model_training.py
=================
ISL Recognition System — LSTM Model Builder & Trainer (Hand-Only)

Imports preprocessed hand-only data from data_preprocessing.py,
constructs a multi-layer LSTM network, trains it with callbacks,
and saves the best model as isl_lstm_model.h5.

Input: (30 frames, 126 hand features) per sequence

Usage
-----
    python model_training.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential          # type: ignore
from tensorflow.keras.layers import (                   # type: ignore
    LSTM, Dense, Dropout, BatchNormalization,
)
from tensorflow.keras.callbacks import (                # type: ignore
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard,
)
from tensorflow.keras.optimizers import Adam            # type: ignore

from data_preprocessing import preprocess, SEQUENCE_LENGTH, NUM_FEATURES

# ─────────────────────────── Configuration ───────────────────────────

MODEL_SAVE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "isl_lstm_model.h5"
)
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

EPOCHS = 150
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EARLY_STOP_PATIENCE = 20
LR_REDUCE_PATIENCE = 8
LR_REDUCE_FACTOR = 0.5


# ─────────────────────────── Model Builder ───────────────────────────

def build_model(
    sequence_length: int,
    num_features: int,
    num_classes: int,
) -> Sequential:
    """
    Build a 3-layer LSTM model for hand-keypoint sequence classification.

    Architecture (optimised for 126-dim hand input)
    ------------------------------------------------
    LSTM(64)  → Dropout → BatchNorm
    LSTM(128) → Dropout → BatchNorm
    LSTM(64)  → Dropout → BatchNorm
    Dense(64, relu) → Dropout
    Dense(32, relu)
    Dense(num_classes, softmax)
    """
    model = Sequential([
        # --- Layer 1 ---
        LSTM(64, return_sequences=True,
             input_shape=(sequence_length, num_features),
             name="lstm_1"),
        Dropout(0.3, name="drop_1"),
        BatchNormalization(name="bn_1"),

        # --- Layer 2 ---
        LSTM(128, return_sequences=True, name="lstm_2"),
        Dropout(0.3, name="drop_2"),
        BatchNormalization(name="bn_2"),

        # --- Layer 3 ---
        LSTM(64, return_sequences=False, name="lstm_3"),
        Dropout(0.3, name="drop_3"),
        BatchNormalization(name="bn_3"),

        # --- Classifier Head ---
        Dense(64, activation="relu", name="fc_1"),
        Dropout(0.3, name="drop_4"),
        Dense(32, activation="relu", name="fc_2"),
        Dense(num_classes, activation="softmax", name="output"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ─────────────────────────── Training ────────────────────────────────

def train_model(model, X_train, X_test, y_train, y_test):
    """Train with EarlyStopping, LR reduction, and checkpointing."""
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=LR_REDUCE_FACTOR,
            patience=LR_REDUCE_PATIENCE,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        TensorBoard(log_dir=LOG_DIR),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    return history


# ─────────────────────────── Evaluation ──────────────────────────────

def evaluate_model(model, X_test, y_test, actions):
    """Print classification report and save confusion matrix plot."""
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(
        y_true_classes, y_pred_classes,
        target_names=actions, zero_division=0,
    ))

    cm = confusion_matrix(y_true_classes, y_pred_classes)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix", fontsize=16)
    fig.colorbar(im, ax=ax)
    tick_marks = np.arange(len(actions))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(actions, rotation=90, fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(actions, fontsize=7)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Confusion matrix saved → {cm_path}")


def save_training_plots(history):
    """Save accuracy and loss curves."""
    base_dir = os.path.dirname(os.path.abspath(__file__))

    fig, ax = plt.subplots()
    ax.plot(history.history["accuracy"], label="Train Accuracy")
    ax.plot(history.history["val_accuracy"], label="Val Accuracy")
    ax.set_title("Model Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(base_dir, "accuracy_plot.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(history.history["loss"], label="Train Loss")
    ax.plot(history.history["val_loss"], label="Val Loss")
    ax.set_title("Model Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(base_dir, "loss_plot.png"), dpi=120)
    plt.close(fig)

    print("[INFO] Training plots saved → accuracy_plot.png, loss_plot.png")


# ─────────────────────────── Main ────────────────────────────────────

def main() -> None:
    """End-to-end: preprocess → build → train → evaluate → save."""
    print("=" * 60)
    print("  ISL LSTM Model — Training Pipeline (Hand-Only)")
    print("=" * 60)

    # 1. Preprocess
    print("\n[STEP 1] Loading & preprocessing data (hand features only) …")
    X_train, X_test, y_train, y_test, actions = preprocess()

    num_classes = y_train.shape[1]
    print(f"[INFO] Classes: {num_classes}  |  Features/frame: {NUM_FEATURES}")

    # 2. Build model
    print("\n[STEP 2] Building LSTM model …")
    model = build_model(SEQUENCE_LENGTH, NUM_FEATURES, num_classes)
    model.summary()

    # 3. Train
    print("\n[STEP 3] Training …")
    history = train_model(model, X_train, X_test, y_train, y_test)

    # 4. Save plots
    save_training_plots(history)

    # 5. Evaluate
    print("\n[STEP 4] Evaluating on test set …")
    evaluate_model(model, X_test, y_test, actions)

    # 6. Final save confirmation
    if os.path.isfile(MODEL_SAVE_PATH):
        size_mb = os.path.getsize(MODEL_SAVE_PATH) / (1024 * 1024)
        print(f"\n✅  Model saved → {MODEL_SAVE_PATH}  ({size_mb:.1f} MB)")
    else:
        print("\n⚠️  Model file not found — check training logs.")

    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Training interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise
