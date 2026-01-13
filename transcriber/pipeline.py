from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from transcriber.errors import CancelledError
from transcriber.ffmpeg_utils import extract_audio_to_wav
from transcriber.utils import sanitize_filename_component


@dataclass(frozen=True)
class TranscriptionOutputs:
    base_name: str
    outdir: Path
    segments_json_path: Path
    timestamps_txt_path: Path
    wav_path: Path | None


def transcribe_media_to_outputs(
    *,
    input_path: Path,
    outdir: Path,
    model: str = "small",
    language: str = "en",
    device: str = "auto",
    compute_type: str | None = None,
    beam_size: int = 5,
    vad_filter: bool = True,
    keep_wav: bool = False,
    on_log: Callable[[str], None] | None = None,
    cancel_event: object | None = None,
) -> TranscriptionOutputs:
    """
    End-to-end pipeline:
    - Extract audio with ffmpeg
    - Transcribe with faster-whisper
    - Write `*.segments.json` and `*.timestamps.txt`
    """
    input_path = input_path.expanduser().resolve()
    outdir = outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    base = sanitize_filename_component(input_path.stem)
    segments_json_path = outdir / f"{base}.segments.json"
    timestamps_txt_path = outdir / f"{base}.timestamps.txt"
    kept_wav_path = outdir / f"{base}.wav"

    def log(msg: str) -> None:
        if on_log:
            on_log(msg)

    def cancelled() -> bool:
        return bool(cancel_event) and bool(getattr(cancel_event, "is_set", lambda: False)())

    log(f"Input: {input_path.name}")
    log("Extracting audio with ffmpeg…")

    wav_path: Path | None = None
    if keep_wav:
        if cancelled():
            raise CancelledError("Cancelled.")
        wav_path = kept_wav_path
        extract_audio_to_wav(input_path=input_path, output_wav_path=wav_path, cancel_event=cancel_event)
        _transcribe_wav_and_write_outputs(
            wav_path=wav_path,
            model=model,
            language=language,
            device=device,
            compute_type=compute_type,
            beam_size=beam_size,
            vad_filter=vad_filter,
            segments_json_path=segments_json_path,
            timestamps_txt_path=timestamps_txt_path,
            on_log=on_log,
            cancel_event=cancel_event,
        )
        return TranscriptionOutputs(
            base_name=base,
            outdir=outdir,
            segments_json_path=segments_json_path,
            timestamps_txt_path=timestamps_txt_path,
            wav_path=wav_path,
        )

    with tempfile.TemporaryDirectory(prefix="transcriber-") as td:
        if cancelled():
            raise CancelledError("Cancelled.")
        wav_path = Path(td) / f"{base}.wav"
        extract_audio_to_wav(input_path=input_path, output_wav_path=wav_path, cancel_event=cancel_event)
        _transcribe_wav_and_write_outputs(
            wav_path=wav_path,
            model=model,
            language=language,
            device=device,
            compute_type=compute_type,
            beam_size=beam_size,
            vad_filter=vad_filter,
            segments_json_path=segments_json_path,
            timestamps_txt_path=timestamps_txt_path,
            on_log=on_log,
            cancel_event=cancel_event,
        )

    return TranscriptionOutputs(
        base_name=base,
        outdir=outdir,
        segments_json_path=segments_json_path,
        timestamps_txt_path=timestamps_txt_path,
        wav_path=(kept_wav_path if keep_wav else None),
    )


def _transcribe_wav_and_write_outputs(
    *,
    wav_path: Path,
    model: str,
    language: str,
    device: str,
    compute_type: str | None,
    beam_size: int,
    vad_filter: bool,
    segments_json_path: Path,
    timestamps_txt_path: Path,
    on_log: Callable[[str], None] | None,
    cancel_event: object | None,
) -> None:
    # Lazy import so simply importing the package/GUI is lightweight.
    from transcriber.transcribe import format_timestamp, transcribe_wav

    if cancel_event and getattr(cancel_event, "is_set", lambda: False)():
        raise CancelledError("Cancelled.")

    if on_log:
        on_log("Transcribing… (this may take a while on first run)")

    result = transcribe_wav(
        wav_path=wav_path,
        model_name=model,
        language=language,
        device=device,
        compute_type=compute_type,
        beam_size=beam_size,
        vad_filter=vad_filter,
        cancel_event=cancel_event,
    )

    if cancel_event and getattr(cancel_event, "is_set", lambda: False)():
        raise CancelledError("Cancelled.")

    if on_log:
        on_log("Writing outputs…")

    if cancel_event and getattr(cancel_event, "is_set", lambda: False)():
        raise CancelledError("Cancelled.")

    payload = {
        "info": result.info,
        "segments": [
            {
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "avg_logprob": s.avg_logprob,
                "no_speech_prob": s.no_speech_prob,
                "compression_ratio": s.compression_ratio,
            }
            for s in result.segments
        ],
    }

    segments_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines: list[str] = []
    for s in result.segments:
        if not s.text:
            continue
        start = format_timestamp(s.start)
        end = format_timestamp(s.end)
        lines.append(f"[{start} --> {end}] {s.text}")

    timestamps_txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    if on_log:
        on_log("Done.")

