"""Audio preprocessing shared by model training and Raspberry Pi inference."""

from pathlib import Path

import librosa
import numpy as np

TARGET_SR = 16_000
WINDOW_SECONDS = 3.0
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
WINDOW_SAMPLES = int(TARGET_SR * WINDOW_SECONDS)
SPEC_FRAMES = 1 + WINDOW_SAMPLES // HOP_LENGTH


def load_signal(csv_path: Path) -> np.ndarray:
    signal = np.genfromtxt(csv_path, delimiter=",", dtype=np.float32)
    if signal.size == 0:
        raise ValueError(f"CSV is empty: {csv_path}")
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    return signal[:, 0].astype(np.float32)


def make_windows(signal: np.ndarray) -> list[np.ndarray]:
    if len(signal) < WINDOW_SAMPLES:
        signal = np.pad(signal, (0, WINDOW_SAMPLES - len(signal)))
    return [
        signal[start:start + WINDOW_SAMPLES]
        for start in range(0, len(signal) - WINDOW_SAMPLES + 1, WINDOW_SAMPLES)
    ]


def spectrogram_from_window(window: np.ndarray) -> np.ndarray:
    spec = librosa.feature.melspectrogram(
        y=window.astype(np.float32),
        sr=TARGET_SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    spec_db = librosa.power_to_db(spec, ref=np.max)
    spec_db = (spec_db - np.min(spec_db)) / (np.max(spec_db) - np.min(spec_db) + 1e-8)
    return spec_db.astype(np.float32)


def features_from_window(window: np.ndarray) -> np.ndarray:
    values = window.astype(np.float32)
    mfcc = librosa.feature.mfcc(y=values, sr=TARGET_SR, n_mfcc=20, n_fft=N_FFT, hop_length=HOP_LENGTH)
    zcr = librosa.feature.zero_crossing_rate(values, frame_length=N_FFT, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=values, frame_length=N_FFT, hop_length=HOP_LENGTH)

    mfcc_stats = []
    for band in mfcc:
        mean = np.mean(band)
        mfcc_stats.extend([float(mean), float(np.var(band)), float(np.mean(np.abs(band - mean)))])
    zcr_stats = [float(np.mean(zcr)), float(np.var(zcr)), float(np.mean(np.abs(zcr - np.mean(zcr))))]
    rms_stats = [float(np.mean(rms)), float(np.var(rms)), float(np.mean(np.abs(rms - np.mean(rms))))]
    return np.asarray(mfcc_stats + zcr_stats + rms_stats, dtype=np.float32)