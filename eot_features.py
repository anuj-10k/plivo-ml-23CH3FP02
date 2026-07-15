"""Causal acoustic features for end-of-turn detection.

Every public extraction function slices the waveform at ``pause_start`` before
doing any analysis.  Neither ``pause_end`` nor any following samples are
accepted by this module.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf


EPS = 1e-10
FRAME_MS = 25
HOP_MS = 10
WINDOWS = (0.30, 0.70, 1.50, 3.00)


def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return np.asarray(x, dtype=np.float32), int(sr)


def _frames(x: np.ndarray, sr: int, frame_ms: int = FRAME_MS) -> np.ndarray:
    frame_len = max(1, round(sr * frame_ms / 1000))
    hop_len = max(1, round(sr * HOP_MS / 1000))
    if x.size < frame_len:
        x = np.pad(x, (frame_len - x.size, 0))
    n = 1 + (x.size - frame_len) // hop_len
    indices = np.arange(frame_len)[None, :] + hop_len * np.arange(n)[:, None]
    return x[indices]


def _slope(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    t = np.linspace(-1.0, 1.0, values.size)
    return float(np.dot(t, values - values.mean()) / (np.dot(t, t) + EPS))


def _safe_stats(values: np.ndarray) -> list[float]:
    if values.size == 0:
        return [0.0] * 7
    return [
        float(values.mean()), float(values.std()), float(values.min()),
        float(values.max()), float(np.median(values)),
        float(np.quantile(values, 0.25)), float(np.quantile(values, 0.75)),
    ]


def _pitch(frames: np.ndarray, sr: int, energy_db: np.ndarray) -> np.ndarray:
    """Small CPU autocorrelation pitch tracker; zero denotes unvoiced."""
    result = np.zeros(len(frames), dtype=np.float32)
    lo = max(1, int(sr / 400.0))
    hi = min(frames.shape[1] - 1, int(sr / 60.0))
    if hi <= lo:
        return result
    voiced_energy_floor = max(-55.0, float(np.max(energy_db)) - 32.0)
    window = np.hanning(frames.shape[1]).astype(np.float32)
    for i, frame in enumerate(frames):
        if energy_db[i] < voiced_energy_floor:
            continue
        centered = (frame - frame.mean()) * window
        ac = np.correlate(centered, centered, mode="full")[len(centered) - 1:]
        if ac[0] <= EPS:
            continue
        region = ac[lo:hi + 1]
        lag = lo + int(np.argmax(region))
        strength = ac[lag] / (ac[0] + EPS)
        if strength >= 0.32:
            result[i] = sr / lag
    return result


def _window_features(segment: np.ndarray, sr: int) -> list[float]:
    fr = _frames(segment, sr)
    rms = np.sqrt(np.mean(fr * fr, axis=1) + EPS)
    energy = 20.0 * np.log10(rms + EPS)
    peak = float(np.max(energy))
    active = energy > max(-55.0, peak - 28.0)
    signs = fr >= 0
    zcr = np.mean(signs[:, 1:] != signs[:, :-1], axis=1)

    feats = _safe_stats(energy)
    feats += [
        float(energy[-1]), float(np.mean(energy[-3:])), float(np.mean(energy[-8:])),
        _slope(energy),
        float(np.mean(energy[len(energy) // 2:]) - np.mean(energy[:max(1, len(energy) // 2)])),
        float(active.mean()), float(np.mean(active[-5:])),
    ]
    feats += _safe_stats(zcr)[:2] + [_slope(zcr), float(np.mean(zcr[-5:]))]

    f0 = _pitch(fr, sr, energy)
    voiced_mask = f0 > 0
    voiced = np.log2(f0[voiced_mask]) if voiced_mask.any() else np.empty(0)
    feats += _safe_stats(voiced)
    feats += [float(voiced_mask.mean()), float(np.mean(voiced_mask[-5:]))]
    if voiced.size:
        last_values = voiced[-min(8, voiced.size):]
        feats += [float(voiced[-1]), _slope(last_values), float(voiced[-1] - np.median(voiced))]
    else:
        feats += [0.0, 0.0, 0.0]

    # Duration of the final contiguous voiced island and gap from it to pause.
    voiced_indices = np.flatnonzero(voiced_mask)
    if voiced_indices.size:
        final = int(voiced_indices[-1])
        begin = final
        while begin > 0 and voiced_mask[begin - 1]:
            begin -= 1
        feats += [(final - begin + 1) * HOP_MS / 1000.0,
                  (len(voiced_mask) - 1 - final) * HOP_MS / 1000.0]
    else:
        feats += [0.0, len(voiced_mask) * HOP_MS / 1000.0]

    # A rough causal syllable/rhythm proxy: prominent local energy maxima.
    if len(energy) >= 3:
        peaks = (energy[1:-1] > energy[:-2]) & (energy[1:-1] >= energy[2:])
        peaks &= energy[1:-1] > max(-50.0, peak - 18.0)
        peak_rate = float(peaks.sum() / max(segment.size / sr, 0.05))
    else:
        peak_rate = 0.0
    feats.append(peak_rate)
    return feats


def extract_features(x: np.ndarray, sr: int, pause_start: float,
                     pause_index: int) -> np.ndarray:
    """Return features using only samples in ``x[0:pause_start]``."""
    end = min(len(x), max(0, int(math.floor(float(pause_start) * sr))))
    past = x[:end]  # The single causality boundary for all acoustic features.
    feats: list[float] = [
        float(pause_start), math.log1p(max(0.0, float(pause_start))),
        float(pause_index), math.log1p(max(0, int(pause_index))),
    ]
    for seconds in WINDOWS:
        n = max(1, int(seconds * sr))
        segment = past[max(0, len(past) - n):]
        feats.extend(_window_features(segment, sr))
    return np.nan_to_num(np.asarray(feats, dtype=np.float32), nan=0.0,
                         posinf=0.0, neginf=0.0)
