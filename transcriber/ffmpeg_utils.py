from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from transcriber.errors import CancelledError


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
    cancel_event: object | None = None,
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
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
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

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def _cancelled() -> bool:
        return bool(cancel_event) and bool(getattr(cancel_event, "is_set", lambda: False)())

    while proc.poll() is None:
        if _cancelled():
            # Best-effort termination.
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:  # noqa: BLE001
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass
            raise CancelledError("Cancelled while extracting audio.")
        time.sleep(0.1)

    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise FfmpegFailedError(
            "ffmpeg failed to extract audio.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stderr:\n{stderr}"
        )

    return FfmpegResult(wav_path=output_wav_path)


