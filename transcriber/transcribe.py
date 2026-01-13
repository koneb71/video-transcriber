from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from faster_whisper import WhisperModel


@dataclass(frozen=True)
class TranscriptionSegment:
    start: float
    end: float
    text: str

    # Optional diagnostics (may be None depending on backend/version)
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None


@dataclass(frozen=True)
class TranscriptionResult:
    segments: list[TranscriptionSegment]
    info: dict[str, Any]


def _default_compute_type_for_device(device: str) -> str:
    # Reasonable defaults:
    # - CPU: int8 is typically fastest and memory-efficient.
    # - GPU (CUDA/Metal): float16 is a common performant default.
    d = (device or "").lower()
    if d in {"cuda", "metal"}:
        return "float16"
    return "int8"


def _init_model_with_fallback(
    *,
    model_name: str,
    device: str,
    compute_type: str | None,
) -> tuple[WhisperModel, str, str]:
    """
    Initialize faster-whisper model, supporting device='auto' which prefers GPU if present.
    Returns: (model, device_used, compute_type_used)
    """
    requested_device = (device or "auto").lower()

    def _try_init(d: str) -> tuple[WhisperModel, str, str]:
        ct = compute_type if compute_type is not None else _default_compute_type_for_device(d)
        return WhisperModel(model_name, device=d, compute_type=ct), d, ct

    if requested_device != "auto":
        return _try_init(requested_device)

    # Prefer GPU if available, then CPU.
    # - Windows/Linux: try CUDA first.
    # - macOS: try Metal first.
    last_err: Exception | None = None
    if sys.platform == "darwin":
        candidates = ("metal", "cpu")
    else:
        # win32, linux, etc.
        candidates = ("cuda", "cpu")

    for candidate in candidates:
        try:
            return _try_init(candidate)
        except Exception as e:  # noqa: BLE001 - best-effort backend probing
            last_err = e

    # Should be unreachable since cpu should generally work, but be explicit.
    raise RuntimeError("Failed to initialize WhisperModel on any backend") from last_err


def transcribe_wav(
    *,
    wav_path: Path,
    model_name: str = "small",
    language: str = "en",
    device: str = "auto",
    compute_type: str | None = None,
    beam_size: int = 5,
    vad_filter: bool = True,
) -> TranscriptionResult:
    """
    Run local transcription using faster-whisper and return timestamped segments.
    """
    wav_path = wav_path.expanduser().resolve()

    model, device_used, compute_type_used = _init_model_with_fallback(
        model_name=model_name,
        device=device,
        compute_type=compute_type,
    )
    segments_iter, info = model.transcribe(
        str(wav_path),
        language=language,
        beam_size=beam_size,
        vad_filter=vad_filter,
    )

    segments: list[TranscriptionSegment] = []
    for seg in _iter_segments(segments_iter):
        segments.append(
            TranscriptionSegment(
                start=float(seg.start),
                end=float(seg.end),
                text=(seg.text or "").strip(),
                avg_logprob=getattr(seg, "avg_logprob", None),
                no_speech_prob=getattr(seg, "no_speech_prob", None),
                compression_ratio=getattr(seg, "compression_ratio", None),
            )
        )

    info_dict: dict[str, Any] = {}
    # `info` is a dataclass-like object in faster-whisper; keep it JSON-safe.
    for key in (
        "language",
        "language_probability",
        "duration",
        "duration_after_vad",
    ):
        if hasattr(info, key):
            info_dict[key] = getattr(info, key)

    info_dict["model_name"] = model_name
    info_dict["device"] = device_used
    info_dict["compute_type"] = compute_type_used
    info_dict["beam_size"] = beam_size
    info_dict["vad_filter"] = vad_filter

    return TranscriptionResult(segments=segments, info=info_dict)


def _iter_segments(segments_iter: Iterable[Any]) -> Iterable[Any]:
    for s in segments_iter:
        yield s


def format_timestamp(seconds: float) -> str:
    """
    Format seconds to HH:MM:SS.mmm (no days).
    """
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000.0))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


