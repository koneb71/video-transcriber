# Local Media → Text Transcriber (offline)

Transcribe **video or audio files** (anything `ffmpeg` can decode) to **timestamped text** locally (no cloud APIs).

## Requirements
- macOS, Windows, or Linux
- Python 3.10+ (3.11 recommended)
- `ffmpeg` installed and available on your PATH

Install ffmpeg (macOS):

```bash
brew install ffmpeg
```

Install ffmpeg (Windows):

- `winget install Gyan.FFmpeg`
- or `choco install ffmpeg`

Install ffmpeg (Linux, Debian/Ubuntu):

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

## Setup
From this repo folder:

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
python -m pip install -U pip
pip install -r requirements.txt
```

On Windows (PowerShell):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

On Windows (cmd.exe):

```bat
py -m venv .venv
.\.venv\Scripts\activate.bat
python -m pip install -U pip
pip install -r requirements.txt
```

## Usage
Transcribe a media file to timestamped text + JSON segments:

```bash
python -m transcriber.cli --input /path/to/video.mkv --model small --outdir ./output
```

Outputs:
- `output/<video>.timestamps.txt`
- `output/<video>.segments.json`

## GUI (desktop app)
Launch the GUI:

```bash
python -m transcriber.gui
```

## Notes (local-only)
- The first run will download the selected Whisper model weights into your local cache (still running on-device).
- All transcription happens locally on your machine.

## Options
- `--model`: Whisper model name (e.g. `tiny`, `base`, `small`, `medium`, `large-v3`)
- `--language`: defaults to `en`
- `--device`: defaults to `auto` (tries GPU backends first, falls back to CPU)
- `--compute-type`: if omitted, defaults based on device (GPU: `float16`, CPU: `int8`)


