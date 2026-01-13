from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FfmpegResult:
    wav_path: Path


class FfmpegNotFoundError(RuntimeError):
    pass


class FfmpegFailedError(RuntimeError):
    pass


def ensure_ffmpeg_available() -> str:
    """
    Returns the ffmpeg executable path if available, otherwise raises.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FfmpegNotFoundError(
            "ffmpeg not found on PATH. Install ffmpeg and ensure it's available as `ffmpeg` "
            "("
            "macOS: `brew install ffmpeg`, "
            "Windows: `winget install Gyan.FFmpeg` or `choco install ffmpeg`, "
            "Linux: `sudo apt-get install ffmpeg`"
            ")."
        )
    return ffmpeg


def extract_audio_to_wav(
    *,
    input_path: Path,
    output_wav_path: Path,
    sample_rate_hz: int = 16_000,
    channels: int = 1,
) -> FfmpegResult:
    """
    Extract audio from a video file to a PCM WAV suitable for ASR (mono, 16kHz).
    """
    ffmpeg = ensure_ffmpeg_available()

    input_path = input_path.expanduser().resolve()
    output_wav_path = output_wav_path.expanduser().resolve()
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg,
        "-y",  # overwrite output
        "-i",
        str(input_path),
        "-vn",  # no video
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate_hz),
        "-c:a",
        "pcm_s16le",
        str(output_wav_path),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FfmpegFailedError(
            "ffmpeg failed to extract audio.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stderr:\n{proc.stderr}"
        )

    return FfmpegResult(wav_path=output_wav_path)


