#!/usr/bin/env python3
"""Predict a bearing class from the first three seconds of a numeric CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

import acoustic_preprocessing as preprocessing


CLASS_NAMES = {0: "normal", 1: "overhang", 2: "underhang"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Numeric CSV recording; first column is used")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/pi_full_multiclass/svm_model.joblib"),
        help="Path to the trained SVM pipeline",
    )
    parser.add_argument(
        "--source-sample-rate",
        type=int,
        default=16_000,
        help="Measured CSV acquisition rate in Hz (default: 16000)",
    )
    args = parser.parse_args()

    if not args.csv.is_file():
        parser.error(f"CSV file not found: {args.csv}")
    if not args.model.is_file():
        parser.error(f"SVM model not found: {args.model}")

    signal = preprocessing.load_signal(args.csv)
    signal = preprocessing.resample_signal(signal, args.source_sample_rate)
    windows = preprocessing.make_windows(signal)
    features = preprocessing.features_from_window(windows[0])[np.newaxis, :]

    model = joblib.load(args.model)
    probabilities = model.predict_proba(features)[0]
    class_ids = np.asarray(model.classes_, dtype=int)
    if len(class_ids) != len(probabilities) or any(class_id not in CLASS_NAMES for class_id in class_ids):
        raise ValueError("The SVM classes do not match the expected three-class label mapping.")

    best = int(np.argmax(probabilities))
    result = {
        "prediction": CLASS_NAMES[int(class_ids[best])],
        "confidence": float(probabilities[best]),
        "class_probabilities": {
            CLASS_NAMES[int(class_id)]: float(probability)
            for class_id, probability in zip(class_ids, probabilities)
        },
        "analyzed_seconds": preprocessing.WINDOW_SECONDS,
        "source_sample_rate_hz": args.source_sample_rate,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
