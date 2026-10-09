#!/usr/bin/env python3
"""Train one integrated CNN-embedding plus linear-SVM classifier."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from acoustic_model import preprocessing


CLASS_NAMES = ["normal", "overhang", "underhang"]
LABELS = {name: index for index, name in enumerate(CLASS_NAMES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("train images"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/hybrid_multiclass"))
    parser.add_argument("--source-sample-rate", type=int, default=16_000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def stratified_split(files: list[Path], labels: list[int], test_size: float, seed: int):
    return train_test_split(
        files,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )


def collect_dataset(root: Path, source_sr: int, seed: int):
    files: list[Path] = []
    file_labels: list[int] = []
    for class_name, label in LABELS.items():
        class_dir = root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        class_files = sorted(class_dir.rglob("*.csv"))
        print(f"{class_name}: {len(class_files)} CSV files", flush=True)
        files.extend(class_files)
        file_labels.extend([label] * len(class_files))

    if not files:
        raise FileNotFoundError(f"No CSV recordings found under {root}")

    remaining_files, test_files, remaining_y, test_y = stratified_split(
        files, file_labels, test_size=0.20, seed=seed
    )
    train_files, validation_files, train_y, validation_y = stratified_split(
        remaining_files, remaining_y, test_size=0.15, seed=seed + 1
    )
    split_paths = [
        {path.resolve() for path in split}
        for split in (train_files, validation_files, test_files)
    ]
    if (
        split_paths[0] & split_paths[1]
        or split_paths[0] & split_paths[2]
        or split_paths[1] & split_paths[2]
    ):
        raise RuntimeError("Recording-level train, validation, and test splits overlap.")
    print(
        "Recording splits — "
        f"train: {len(train_files)}, validation: {len(validation_files)}, "
        f"test: {len(test_files)} (disjoint)",
        flush=True,
    )

    def build_split(split_files: list[Path], split_labels: list[int], name: str):
        X: list[np.ndarray] = []
        y: list[int] = []
        for index, (path, label) in enumerate(zip(split_files, split_labels), start=1):
            signal = preprocessing.load_signal(path)
            signal = preprocessing.resample_signal(signal, source_sr)
            for window in preprocessing.make_windows(signal):
                spec = preprocessing.spectrogram_from_window(window)
                X.append(spec[..., np.newaxis])
                y.append(label)
            if index % 50 == 0 or index == len(split_files):
                print(f"{name}: processed {index}/{len(split_files)} recordings", flush=True)
        if not X:
            raise ValueError(f"No usable windows were produced for {name}.")
        return np.stack(X), np.asarray(y, dtype=np.int64)

    X_train, y_train = build_split(train_files, train_y, "train")
    X_validation, y_validation = build_split(validation_files, validation_y, "validation")
    X_test, y_test = build_split(test_files, test_y, "test")
    print(
        f"Windows — train: {len(y_train)}, validation: {len(y_validation)}, test: {len(y_test)}",
        flush=True,
    )
    recording_split_counts = {
        "train": len(train_files),
        "validation": len(validation_files),
        "test": len(test_files),
    }
    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        recording_split_counts,
    )


def build_cnn() -> tf.keras.Model:
    inputs = tf.keras.layers.Input(
        shape=(preprocessing.N_MELS, preprocessing.SPEC_FRAMES, 1), name="mel_spectrogram"
    )
    x = tf.keras.layers.Conv2D(16, 3, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    embedding = tf.keras.layers.Dense(32, activation="relu", name="cnn_embedding")(x)
    logits = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax", name="cnn_classifier")(embedding)
    model = tf.keras.Model(inputs, logits)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def balance_training_windows(X: np.ndarray, y: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(y == label) for label in range(len(CLASS_NAMES))]
    if any(len(indices) == 0 for indices in class_indices):
        raise ValueError("Every class must have training windows for the hybrid model.")
    per_class = max(map(len, class_indices))
    sampled = np.concatenate([
        rng.choice(indices, size=per_class, replace=len(indices) < per_class)
        for indices in class_indices
    ])
    rng.shuffle(sampled)
    return X[sampled], y[sampled]


def main() -> None:
    args = parse_args()
    if args.source_sample_rate <= 0:
        raise ValueError("Source sample rate must be positive.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output folder is not empty; refusing to overwrite: {args.output_dir}")

    tf.keras.utils.set_random_seed(args.seed)
    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        recording_split_counts,
    ) = collect_dataset(args.root, args.source_sample_rate, args.seed)
    balanced_X, balanced_y = balance_training_windows(X_train, y_train, args.seed)
    print("Balanced CNN training class counts:", np.bincount(balanced_y).tolist(), flush=True)

    cnn = build_cnn()
    cnn.fit(
        balanced_X,
        balanced_y,
        validation_data=(X_validation, y_validation),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        )],
        verbose=2,
    )

    encoder = tf.keras.Model(cnn.input, cnn.get_layer("cnn_embedding").output)
    train_embeddings = encoder.predict(X_train, batch_size=128, verbose=0)
    test_embeddings = encoder.predict(X_test, batch_size=128, verbose=0)
    svm = LinearSVC(C=1.0, class_weight="balanced", random_state=args.seed, dual="auto", max_iter=10_000)
    svm.fit(train_embeddings, y_train)
    svm_scores = svm.decision_function(test_embeddings)
    predictions = svm.classes_[np.argmax(svm_scores, axis=1)]

    print("Hybrid held-out accuracy:", accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions, target_names=CLASS_NAMES, digits=4, zero_division=0))
    print("Hybrid confusion matrix:\n", confusion_matrix(y_test, predictions))

    # LinearSVC is one-vs-rest: its decision function is exactly a dense affine
    # layer. Embed those learned SVM coefficients after the CNN embedding so the
    # Pi receives and loads one integrated model file.
    hybrid_scores = tf.keras.layers.Dense(len(CLASS_NAMES), activation=None, name="svm_decision")(
        encoder.output
    )
    hybrid = tf.keras.Model(encoder.input, hybrid_scores, name="cnn_linear_svm_hybrid")
    decision_layer = hybrid.get_layer("svm_decision")
    decision_layer.set_weights([svm.coef_.T.astype(np.float32), svm.intercept_.astype(np.float32)])

    fused_scores = test_embeddings @ svm.coef_.T + svm.intercept_
    fused_predictions = svm.classes_[np.argmax(fused_scores, axis=1)]
    if not np.array_equal(fused_predictions, predictions):
        raise RuntimeError("Fused SVM weights do not reproduce LinearSVC decisions.")

    test_accuracy = accuracy_score(y_test, predictions)
    test_report = classification_report(
        y_test,
        predictions,
        target_names=CLASS_NAMES,
        output_dict=True,
        digits=4,
        zero_division=0,
    )
    test_confusion = confusion_matrix(y_test, predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "hybrid_model.h5"
    hybrid.save(model_path)
    metadata = {
        "mode": "multiclass",
        "class_names": CLASS_NAMES,
        "architecture": "CNN mel-spectrogram embedding followed by fused one-vs-rest LinearSVC decision layer",
        "input": {
            "signal_column": 0,
            "source_sample_rate": args.source_sample_rate,
            "sample_rate": preprocessing.TARGET_SR,
            "window_seconds": preprocessing.WINDOW_SECONDS,
            "n_mels": preprocessing.N_MELS,
            "n_fft": preprocessing.N_FFT,
            "hop_length": preprocessing.HOP_LENGTH,
            "input_shape": [preprocessing.N_MELS, preprocessing.SPEC_FRAMES, 1],
        },
        "hybrid_model": model_path.name,
        "test_windows": int(len(y_test)),
        "recording_split_counts": recording_split_counts,
        "held_out_metrics": {
            "unit": "windows",
            "accuracy": float(test_accuracy),
            "classification_report": test_report,
            "confusion_matrix": test_confusion.tolist(),
        },
        "split_method": "stratified recording-level split; windows from the same CSV remain in one split",
        "decision_outputs": "raw one-vs-rest SVM scores; choose the class with the highest score",
        "training_elapsed_seconds": round(time.monotonic() - start_time, 2),
    }
    (args.output_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print("Integrated hybrid model saved to:", model_path, flush=True)


if __name__ == "__main__":
    start_time = time.monotonic()
    main()
