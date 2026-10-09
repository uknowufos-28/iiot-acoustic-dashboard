#!/usr/bin/env python3
"""Train small acoustic fault classifiers from the raw CSV recordings in train images.

This script treats each CSV as a raw acoustic time series and turns it into:
- a 2D mel-spectrogram image for a compact CNN
- a 66-D audio feature vector for an RBF SVM

Supported label modes:
- multiclass: normal / overhang / underhang
- binary: normal / fault (overhang + underhang combined)

The dataset split is done per file path so windows from one recording do not leak
between train/test sets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import acoustic_preprocessing as preprocessing
import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path("train images")
TARGET_SR = preprocessing.TARGET_SR
WINDOW_SECONDS = preprocessing.WINDOW_SECONDS
N_MELS = preprocessing.N_MELS
N_FFT = preprocessing.N_FFT
HOP_LENGTH = preprocessing.HOP_LENGTH
WINDOW_SAMPLES = preprocessing.WINDOW_SAMPLES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train compact acoustic fault models from raw CSV audio recordings.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Root directory with class folders.")
    parser.add_argument("--mode", choices=["multiclass", "binary"], default="multiclass",
                        help="Train on three classes (normal/overhang/underhang) or binary (normal/fault).")
    parser.add_argument("--max-files-per-class", type=int, default=None,
                        help="Optional per-class file limit for quick runs; defaults to all recordings.")
    parser.add_argument("--max-windows-per-file", type=int, default=None,
                        help="Optional per-recording window cap for quick runs; defaults to all full windows.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction used for testing.")
    parser.add_argument("--validation-size", type=float, default=0.15,
                        help="Fraction of the remaining recordings used for validation.")
    parser.add_argument("--source-sample-rate", type=int, default=TARGET_SR,
                        help="Sample rate of each source CSV; signals are resampled to 16 kHz.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of CNN training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="CNN batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--train-svm", action="store_true", help="Also train and evaluate the optional SVM.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory to save model artifacts (default: models/<mode>).")
    return parser.parse_args()


def file_label(file_path: Path, mode: str) -> int:
    rel_parts = file_path.relative_to(file_path.parents[0].parent if False else file_path).parts if False else []
    # Use the actual dataset path structure: root/<class>/.../file.csv
    parts = file_path.parts
    try:
        class_name = next(part for part in parts if part in {"normal", "overhang", "underhang"})
    except StopIteration as exc:  # pragma: no cover
        raise ValueError(f"Could not infer label from path: {file_path}") from exc

    if class_name == "normal":
        return 0
    if mode == "binary":
        return 1
    if class_name == "overhang":
        return 1
    if class_name == "underhang":
        return 2
    raise ValueError(f"Unsupported class name: {class_name}")


def collect_files(root: Path, max_files_per_class: int | None) -> Dict[int, List[Path]]:
    labels = {"normal": 0, "overhang": 1, "underhang": 2}
    by_label: Dict[int, List[Path]] = {0: [], 1: [], 2: []}

    for class_name, label in labels.items():
        class_dir = root / class_name
        if not class_dir.exists():
            continue
        csv_files = sorted(class_dir.rglob("*.csv"))
        if max_files_per_class is not None:
            csv_files = csv_files[:max_files_per_class]
        by_label[label] = csv_files

    return by_label


def load_signal(csv_path: Path) -> np.ndarray:
    return preprocessing.load_signal(csv_path)


def make_windows(signal: np.ndarray, sr: int = TARGET_SR) -> List[np.ndarray]:
    return preprocessing.make_windows(signal)


def spectrogram_from_window(window: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    return preprocessing.spectrogram_from_window(window)


def features_from_window(window: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    return preprocessing.features_from_window(window)


def build_cnn(num_classes: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(N_MELS, preprocessing.SPEC_FRAMES, 1)),
        tf.keras.layers.Conv2D(16, 3, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(32, 3, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(64, 3, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def split_recordings(files: List[Path], labels: List[int], test_size: float, seed: int):
    label_counts = np.bincount(np.asarray(labels, dtype=int))
    stratify = labels if len(label_counts) > 1 and label_counts.min() >= 2 else None
    try:
        return train_test_split(
            files,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        return train_test_split(files, test_size=test_size, random_state=seed)


def make_dataset(
    root: Path,
    mode: str,
    max_files_per_class: int | None,
    seed: int,
    test_size: float,
    validation_size: float,
    source_sample_rate: int,
    max_windows_per_file: int | None = None,
) -> Tuple[np.ndarray, ...]:
    by_label = collect_files(root, max_files_per_class)
    files: List[Path] = []
    file_labels: List[int] = []

    for label, class_files in by_label.items():
        if mode == "binary" and label == 0:
            # normal remains 0, all faults become 1
            class_files = class_files
        for file_path in class_files:
            if mode == "binary":
                label_out = 0 if label == 0 else 1
            else:
                label_out = label
            files.append(file_path)
            file_labels.append(label_out)

    if not files:
        raise FileNotFoundError(f"No CSV files found under {root}")

    train_validation_files, test_files = split_recordings(files, file_labels, test_size, seed)
    train_validation_labels = [file_label(path, mode) for path in train_validation_files]
    train_validation_labels = [0 if mode == "binary" and label == 0 else (1 if mode == "binary" else label)
                               for label in train_validation_labels]
    train_files, validation_files = split_recordings(
        train_validation_files,
        train_validation_labels,
        validation_size,
        seed + 1,
    )

    X_train, y_train, svm_X_train = [], [], []
    X_validation, y_validation, svm_X_validation = [], [], []
    X_test, y_test, svm_X_test = [], [], []

    def add_examples(file_list: List[Path], spectrograms: list, features: list, labels_out: list):
        for file_path in file_list:
            label = file_label(file_path, mode)
            if mode == "binary":
                label = 0 if label == 0 else 1
            signal = preprocessing.resample_signal(load_signal(file_path), source_sample_rate)
            windows = make_windows(signal)
            selected_windows = windows if max_windows_per_file is None else windows[:max_windows_per_file]
            for window in selected_windows:
                spec = spectrogram_from_window(window)
                if spec.shape[0] != N_MELS:
                    continue
                labels_out.append(label)
                spectrograms.append(spec[..., np.newaxis])
                features.append(features_from_window(window))

    add_examples(train_files, X_train, svm_X_train, y_train)
    add_examples(validation_files, X_validation, svm_X_validation, y_validation)
    add_examples(test_files, X_test, svm_X_test, y_test)

    if not X_train or not X_validation or not X_test:
        raise ValueError("No usable windows were produced for the train/validation/test split.")

    return (
        np.stack(X_train), np.array(y_train),
        np.stack(X_validation), np.array(y_validation),
        np.stack(X_test), np.array(y_test),
        np.stack(svm_X_train), np.stack(svm_X_validation), np.stack(svm_X_test),
    )


def report_metrics(name: str, y_true: np.ndarray, probabilities: np.ndarray, class_names: List[str]) -> None:
    predictions = np.argmax(probabilities, axis=1)
    print(f"{name} accuracy:", accuracy_score(y_true, predictions))
    print(classification_report(y_true, predictions, target_names=class_names, digits=4, zero_division=0))
    print(f"{name} confusion matrix:\n", confusion_matrix(y_true, predictions))


def train_svm(
    svm_X_train: np.ndarray,
    y_train: np.ndarray,
    svm_X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    output_dir: Path,
) -> SVC:
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", probability=True, random_state=42),
    )
    model.fit(svm_X_train, y_train)
    report_metrics("SVM", y_test, model.predict_proba(svm_X_test), class_names)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "svm_model.joblib")
    return model


def evaluate_tflite(model_content: bytes, X_test: np.ndarray) -> np.ndarray:
    interpreter = tf.lite.Interpreter(model_content=model_content)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    predictions = []

    for sample in X_test:
        input_value = sample[np.newaxis, ...].astype(np.float32)
        input_dtype = input_details["dtype"]
        if np.issubdtype(input_dtype, np.integer):
            scale, zero_point = input_details["quantization"]
            if scale <= 0:
                raise ValueError("TFLite input tensor has an invalid quantization scale.")
            limits = np.iinfo(input_dtype)
            input_value = np.clip(np.rint(input_value / scale + zero_point), limits.min, limits.max)
            input_value = input_value.astype(input_dtype)
        interpreter.set_tensor(input_details["index"], input_value)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]
        if np.issubdtype(output.dtype, np.integer):
            scale, zero_point = output_details["quantization"]
            output = (output.astype(np.float32) - zero_point) * scale
        predictions.append(output.astype(np.float32))
    return np.stack(predictions)


def train_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    class_names: List[str],
    seed: int,
    source_sample_rate: int,
) -> tf.keras.Model:
    tf.keras.utils.set_random_seed(seed)
    model = build_cnn(len(class_names))
    class_counts = np.bincount(y_train, minlength=len(class_names))
    class_weights = {
        class_id: len(y_train) / (len(class_names) * max(int(count), 1))
        for class_id, count in enumerate(class_counts)
    }
    print("Training class counts:", class_counts.tolist(), "weights:", class_weights)
    model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )],
        verbose=2,
    )
    report_metrics("CNN float", y_test, model.predict(X_test, verbose=0), class_names)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "cnn_model.keras")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset():
        for sample in X_train[:min(100, len(X_train))]:
            yield [sample[np.newaxis, ...].astype(np.float32)]

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    (output_dir / "cnn_model_int8.tflite").write_bytes(tflite_model)
    report_metrics("CNN INT8 TFLite", y_test, evaluate_tflite(tflite_model, X_test), class_names)

    metadata = {
        "mode": "binary" if len(class_names) == 2 else "multiclass",
        "class_names": class_names,
        "sample_rate": TARGET_SR,
        "source_sample_rate": source_sample_rate,
        "window_seconds": WINDOW_SECONDS,
        "n_mels": N_MELS,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "spectrogram_frames": preprocessing.SPEC_FRAMES,
        "input_shape": [N_MELS, preprocessing.SPEC_FRAMES, 1],
        "quantization": "full_int8",
    }
    (output_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return model


def main() -> None:
    args = parse_args()
    root = args.root
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    print(f"Scanning {root} for class folders: normal, overhang, underhang")
    print(f"Mode: {args.mode}, max files per class: {args.max_files_per_class}")

    by_label = collect_files(root, args.max_files_per_class)
    print({label: len(files) for label, files in by_label.items()})

    X_train, y_train, X_validation, y_validation, X_test, y_test, svm_X_train, svm_X_validation, svm_X_test = make_dataset(
        root,
        args.mode,
        args.max_files_per_class,
        args.seed,
        args.test_size,
        args.validation_size,
        args.source_sample_rate,
        max_windows_per_file=args.max_windows_per_file,
    )
    print(
        f"Training examples: {len(X_train)} | Validation examples: {len(X_validation)} "
        f"| Test examples: {len(X_test)}"
    )

    class_names = ["normal", "fault"] if args.mode == "binary" else ["normal", "overhang", "underhang"]
    output_dir = args.output_dir or Path("models") / args.mode

    if args.train_svm:
        print("\nTraining optional SVM model...")
        train_svm(svm_X_train, y_train, svm_X_test, y_test, class_names, output_dir)

    print("\nTraining CNN model...")
    train_cnn(
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        output_dir,
        args.epochs,
        args.batch_size,
        class_names,
        args.seed,
        args.source_sample_rate,
    )
    print("\nDone. Models saved in:", output_dir)


if __name__ == "__main__":
    main()
