"""Model loading and CSV inference used by the operator dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import acoustic_preprocessing as preprocessing

try:
    import tensorflow as tf
except Exception:
    tf = None


CLASS_NAMES = {0: "Normal", 1: "Overhang fault", 2: "Underhang fault"}


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    class_scores: dict[str, float]
    sample_count: int
    analyzed_samples: int


@lru_cache(maxsize=4)
def load_models(model_dir: str):
    folder = Path(model_dir)
    svm_path = folder / "svm_model.joblib"
    cnn_path = folder / "cnn_model.keras"
    svm = joblib.load(svm_path) if svm_path.exists() else None
    cnn = tf.keras.models.load_model(cnn_path) if tf is not None and cnn_path.exists() else None
    return svm, cnn


def read_csv_signal(data: bytes) -> np.ndarray:
    """Read numeric CSV samples, using the first column as the mono signal."""
    frame = pd.read_csv(BytesIO(data), header=None, comment="#")
    frame = frame.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if frame.empty:
        raise ValueError("No numeric samples were found in this CSV.")
    signal = frame.iloc[:, 0].dropna().to_numpy(dtype=np.float32)
    if signal.size == 0:
        raise ValueError("The first CSV column does not contain numeric samples.")
    return signal


def predict_signal(signal: np.ndarray, model_name: str, svm, cnn) -> tuple[Prediction, np.ndarray]:
    windows = preprocessing.make_windows(signal)
    window = windows[0]

    if model_name == "CNN":
        if cnn is None:
            raise RuntimeError("CNN inference needs TensorFlow and a saved CNN model.")
        spec = preprocessing.spectrogram_from_window(window)[np.newaxis, ..., np.newaxis]
        probabilities = np.asarray(cnn.predict(spec, verbose=0))[0]
        class_ids = np.arange(len(probabilities), dtype=int)
    else:
        if svm is None:
            raise RuntimeError("The saved SVM model was not found.")
        features = preprocessing.features_from_window(window)[np.newaxis, :]
        probabilities = np.asarray(svm.predict_proba(features))[0]
        class_ids = np.asarray(getattr(svm, "classes_", np.arange(len(probabilities))), dtype=int)
        if len(class_ids) != len(probabilities):
            class_ids = np.arange(len(probabilities), dtype=int)

    label_by_id = {0: "Normal", 1: "Anomaly"} if len(probabilities) == 2 else CLASS_NAMES
    winner = int(np.argmax(probabilities))
    class_id = int(class_ids[winner])
    scores = {
        label_by_id.get(int(class_id), f"Class {class_id}"): float(score)
        for class_id, score in zip(class_ids, probabilities)
    }
    result = Prediction(
        label=label_by_id.get(class_id, f"Class {class_id}"),
        confidence=float(probabilities[winner]),
        class_scores=scores,
        sample_count=int(len(signal)),
        analyzed_samples=int(len(window)),
    )
    return result, window


def available_models(svm, cnn) -> list[str]:
    return [name for name, model in (("CNN", cnn), ("SVM", svm)) if model is not None]
