#!/usr/bin/env python3
"""Run the integrated CNN+SVM model on a numeric CSV recording."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

import acoustic_preprocessing as preprocessing


CLASS_NAMES = ["normal", "overhang", "underhang"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Numeric CSV; the first column is used")
    parser.add_argument("--model", type=Path, default=Path("models/hybrid_multiclass/hybrid_model.h5"))
    parser.add_argument("--source-sample-rate", type=int, default=16_000)
    args = parser.parse_args()

    if not args.csv.is_file():
        parser.error(f"CSV file not found: {args.csv}")
    if not args.model.is_file():
        parser.error(f"Hybrid model not found: {args.model}")

    signal = preprocessing.load_signal(args.csv)
    signal = preprocessing.resample_signal(signal, args.source_sample_rate)
    window = preprocessing.make_windows(signal)[0]
    spec = preprocessing.spectrogram_from_window(window)[np.newaxis, ..., np.newaxis]

    model = tf.keras.models.load_model(args.model, compile=False)
    scores = np.asarray(model.predict(spec, verbose=0))[0]
    if scores.shape != (len(CLASS_NAMES),) or not np.isfinite(scores).all():
        raise ValueError(f"Unexpected hybrid model output: {scores}")
    winner = int(np.argmax(scores))
    print(json.dumps({
        "prediction": CLASS_NAMES[winner],
        "svm_decision_scores": {
            class_name: float(score) for class_name, score in zip(CLASS_NAMES, scores)
        },
        "analyzed_seconds": preprocessing.WINDOW_SECONDS,
        "source_sample_rate_hz": args.source_sample_rate,
    }, indent=2))


if __name__ == "__main__":
    main()
