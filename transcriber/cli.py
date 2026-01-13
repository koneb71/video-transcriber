from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from transcriber.ffmpeg_utils import extract_audio_to_wav
from transcriber.transcribe import format_timestamp, transcribe_wav


_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def sanitize_filename_component(name: str) -> str:
    """
    Make a string safe as a single filename component across Windows/macOS/Linux.
    - Windows forbids: <>:"/\\|?* and NUL, plus trailing dots/spaces and reserved names.
    """
    # Replace Windows-forbidden characters with underscores.
    cleaned = "".join("_" if c in '<>:"/\\|?*' else c for c in name)
    cleaned = cleaned.replace("\x00", "_")

    # Windows doesn't allow trailing spaces/dots.
    cleaned = cleaned.rstrip(" .")

    # Avoid empty names.
    if not cleaned:
        cleaned = "output"

    # Avoid reserved device names on Windows (case-insensitive).
    if cleaned.upper() in _WINDOWS_RESERVED_NAMES:
        cleaned = f"_{cleaned}"

    return cleaned


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Local MP4 → timestamped text transcriber")
    p.add_argument("--input", required=True, help="Path to input .mp4 (or any ffmpeg-supported media)")
    p.add_argument("--outdir", default="output", help="Output directory (default: ./output)")
    p.add_argument("--model", default="small", help="Whisper model name (default: small)")
    p.add_argument("--language", default="en", help="Language code (default: en)")
    p.add_argument(
        "--device",
        default="auto",
        help="Device for faster-whisper: auto|cpu|cuda|metal (default: auto)",
    )
    p.add_argument(
        "--compute-type",
        default=None,
        help="Compute type (e.g. int8, float16). If omitted, picked based on device.",
    )
    p.add_argument("--beam-size", type=int, default=5, help="Beam size (default: 5)")
    p.add_argument("--no-vad", action="store_true", help="Disable VAD filter")
    p.add_argument("--keep-wav", action="store_true", help="Keep extracted wav next to outputs")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    input_path = Path(args.input).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # On Windows, some characters are invalid in filenames. Sanitize outputs so
    # inputs like "Lecture: Part 1.mp4" don't crash on write.
    base = sanitize_filename_component(input_path.stem)
    segments_json_path = outdir / f"{base}.segments.json"
    timestamps_txt_path = outdir / f"{base}.timestamps.txt"
    kept_wav_path = outdir / f"{base}.wav"

    # Extract audio to a temp wav (or keep it if requested)
    if args.keep_wav:
        wav_path = kept_wav_path
        extract_audio_to_wav(input_path=input_path, output_wav_path=wav_path)
    else:
        with tempfile.TemporaryDirectory(prefix="transcriber-") as td:
            wav_path = Path(td) / f"{base}.wav"
            extract_audio_to_wav(input_path=input_path, output_wav_path=wav_path)
            _transcribe_and_write(
                wav_path=wav_path,
                model=args.model,
                language=args.language,
                device=args.device,
                compute_type=args.compute_type,
                beam_size=args.beam_size,
                vad_filter=(not args.no_vad),
                segments_json_path=segments_json_path,
                timestamps_txt_path=timestamps_txt_path,
            )
            return 0

    _transcribe_and_write(
        wav_path=wav_path,
        model=args.model,
        language=args.language,
        device=args.device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        vad_filter=(not args.no_vad),
        segments_json_path=segments_json_path,
        timestamps_txt_path=timestamps_txt_path,
    )
    return 0


def _transcribe_and_write(
    *,
    wav_path: Path,
    model: str,
    language: str,
    device: str,
    compute_type: str,
    beam_size: int,
    vad_filter: bool,
    segments_json_path: Path,
    timestamps_txt_path: Path,
) -> None:
    result = transcribe_wav(
        wav_path=wav_path,
        model_name=model,
        language=language,
        device=device,
        compute_type=compute_type,
        beam_size=beam_size,
        vad_filter=vad_filter,
    )

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


if __name__ == "__main__":
    raise SystemExit(main())


