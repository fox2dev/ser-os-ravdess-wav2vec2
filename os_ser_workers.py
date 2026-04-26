from __future__ import annotations

import os
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any

import numpy as np
import soundfile as sf
import librosa
from filelock import FileLock

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

def parse_ravdess_filename(path: str | Path) -> Dict[str, Any]:
    """Parse RAVDESS filename: modality-vocal_channel-emotion-intensity-statement-repetition-actor.wav"""
    p = Path(path)
    parts = p.stem.split("-")
    if len(parts) != 7:
        raise ValueError(f"Invalid RAVDESS filename format: {p.name}")

    modality, vocal_channel, emotion_code, intensity, statement, repetition, actor = parts
    if emotion_code not in EMOTION_MAP:
        raise ValueError(f"Unknown emotion code {emotion_code} in {p.name}")

    return {
        "path": str(p),
        "filename": p.name,
        "modality": modality,
        "vocal_channel": vocal_channel,
        "emotion_code": emotion_code,
        "emotion": EMOTION_MAP[emotion_code],
        "intensity": intensity,
        "statement": statement,
        "repetition": repetition,
        "actor": int(actor),
    }

def _cache_name_for_file(path: str | Path, target_sr: int) -> str:
    p = Path(path)
    h = hashlib.sha1(str(p.resolve()).encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"{p.stem}_{target_sr}hz_{h}.npy"

def _safe_load_cache(cache_path: Path):
    try:
        arr = np.load(cache_path, mmap_mode="r")
        if arr.ndim != 1 or len(arr) == 0:
            raise ValueError("Invalid cached waveform shape")
        return arr
    except Exception:
        try:
            cache_path.unlink(missing_ok=True)
        except Exception:
            pass
        return None

def _atomic_save_npy(cache_path: Path, array: np.ndarray) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(cache_path.name + f".tmp.{os.getpid()}.npy")

    np.save(str(tmp_path), array.astype(np.float32), allow_pickle=False)

    # fsync the temporary file for durability before atomic replacement.
    with open(tmp_path, "ab") as f:
        f.flush()
        os.fsync(f.fileno())

    os.replace(str(tmp_path), str(cache_path))

def load_audio_mono_16k(path: str | Path, target_sr: int = 16000) -> np.ndarray:
    """Load audio as mono float32, resample to target_sr, peak-normalize safely."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if len(audio) == 0:
        raise ValueError(f"Empty audio file: {path}")

    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    audio = np.asarray(audio, dtype=np.float32)

    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 0:
        audio = audio / peak

    return audio.astype(np.float32)

def preprocess_one(
    path: str | Path,
    cache_dir: str | Path,
    target_sr: int = 16000,
    force_reprocess: bool = False,
) -> Dict[str, Any]:
    """Preprocess one file with synchronization and optional cache bypass."""
    t0 = time.perf_counter()
    p = Path(path)
    cache_dir = Path(cache_dir)
    meta = parse_ravdess_filename(p)

    cache_path = cache_dir / _cache_name_for_file(p, target_sr)
    lock_path = str(cache_path) + ".lock"

    file_stat = os.stat(str(p))

    with FileLock(lock_path):
        cached = None
        if not force_reprocess and cache_path.exists():
            cached = _safe_load_cache(cache_path)
        if cached is not None:
            duration_sec = len(cached) / float(target_sr)
            meta.update({
                "cache_path": str(cache_path),
                "duration_sec": duration_sec,
                "source_size_bytes": file_stat.st_size,
                "cache_hit": True,
                "preprocess_time_sec": time.perf_counter() - t0,
                "worker_pid": os.getpid(),
            })
            return meta

        audio = load_audio_mono_16k(p, target_sr=target_sr)
        _atomic_save_npy(cache_path, audio)

    meta.update({
        "cache_path": str(cache_path),
        "duration_sec": len(audio) / float(target_sr),
        "source_size_bytes": file_stat.st_size,
        "cache_hit": False,
        "preprocess_time_sec": time.perf_counter() - t0,
        "worker_pid": os.getpid(),
    })
    return meta
