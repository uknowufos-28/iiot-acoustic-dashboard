"""Inference helpers for the integrated CNN+SVM model."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import threading

import numpy as np

from . import preprocessing


CLASS_NAMES = ("Normal", "Overhang fault", "Underhang fault")


@dataclass(frozen=True)
class WindowResult:
    window_number: int
    start_seconds: float
    end_seconds: float
    label: str
    decision_scores: dict[str, float]


class TFLiteHybrid:
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.input_detail = interpreter.get_input_details()[0]
        self.output_detail = interpreter.get_output_details()[0]
        self._lock = threading.Lock()

        input_shape = tuple(int(value) for value in self.input_detail["shape"])
        output_shape = tuple(int(value) for value in self.output_detail["shape"])
        expected_input = (1, preprocessing.N_MELS, preprocessing.SPEC_FRAMES, 1)
        if input_shape != expected_input:
            raise ValueError(f"Unexpected hybrid model input shape: {input_shape}")
        if output_shape != (1, len(CLASS_NAMES)):
            raise ValueError(f"Unexpected hybrid model output shape: {output_shape}")

    def predict(self, spectrogram: np.ndarray) -> np.ndarray:
        values = np.asarray(spectrogram, dtype=np.float32)
        if values.shape != (1, preprocessing.N_MELS, preprocessing.SPEC_FRAMES, 1):
            raise ValueError(f"Unexpected spectrogram shape: {values.shape}")

        with self._lock:
            self.interpreter.set_tensor(self.input_detail["index"], values)
            self.interpreter.invoke()
            scores = np.asarray(
                self.interpreter.get_tensor(self.output_detail["index"])[0],
                dtype=np.float32,
            )
        if scores.shape != (len(CLASS_NAMES),) or not np.isfinite(scores).all():
            raise ValueError(f"Hybrid model returned invalid decision scores: {scores}")
        return scores


@lru_cache(maxsize=2)
def load_hybrid_model(model_path: str) -> TFLiteHybrid:
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Hybrid TFLite model not found: {path}")

    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError(
                "Hybrid inference needs tflite-runtime or TensorFlow Lite in this Python environment."
            ) from exc
        Interpreter = tf.lite.Interpreter

    interpreter = Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    return TFLiteHybrid(interpreter)


def analyze_signal(
    signal: np.ndarray,
    model: TFLiteHybrid,
    source_sample_rate: int,
) -> tuple[list[WindowResult], np.ndarray]:
    if source_sample_rate <= 0:
        raise ValueError("Source sample rate must be a positive integer.")
    values = np.asarray(signal, dtype=np.float32).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Sensor signal must contain finite numeric samples.")

    resampled = preprocessing.resample_signal(values, source_sample_rate)
    windows = preprocessing.make_windows(resampled)
    results = []
    for index, window in enumerate(windows):
        spectrogram = preprocessing.spectrogram_from_window(window)[None, ..., None]
        scores = model.predict(spectrogram)
        winner = int(np.argmax(scores))
        start_seconds = index * preprocessing.WINDOW_SECONDS
        results.append(WindowResult(
            window_number=index + 1,
            start_seconds=start_seconds,
            end_seconds=start_seconds + preprocessing.WINDOW_SECONDS,
            label=CLASS_NAMES[winner],
            decision_scores={
                name: float(score) for name, score in zip(CLASS_NAMES, scores)
            },
        ))
    return results, resampled
