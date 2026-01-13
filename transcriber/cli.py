from __future__ import annotations

import argparse
from pathlib import Path

from transcriber.pipeline import transcribe_media_to_outputs


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Local media → timestamped text transcriber (offline)")
    p.add_argument(
        "--input",
        required=True,
        help=(
            "Path to an input media file (any ffmpeg-supported video/audio), e.g. "
            ".mp4 .mov .mkv .webm .avi .mp3 .m4a"
        ),
    )
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

    transcribe_media_to_outputs(
        input_path=input_path,
        outdir=outdir,
        model=args.model,
        language=args.language,
        device=args.device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        vad_filter=(not args.no_vad),
        keep_wav=args.keep_wav,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


