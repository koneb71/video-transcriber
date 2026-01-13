# Contributing

Thanks for helping improve this project!

## Ways to contribute
- Report bugs and request features via GitHub Issues.
- Improve docs (README, examples, troubleshooting).
- Submit pull requests for fixes and enhancements.

## Development setup
### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

### Windows (PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

## Running locally
```bash
python -m transcriber.cli --input /path/to/media --outdir ./output
```

## Pull request guidelines
- Keep PRs focused (one fix/feature per PR).
- Update `README.md` if you change CLI flags, defaults, or platform behavior.
- Prefer cross-platform changes (macOS/Windows/Linux).
- Avoid committing large media files and generated outputs.
- If adding dependencies, explain why and pin versions in `requirements.txt`.

## Reporting bugs
Please include:
- OS (macOS/Windows/Linux) + Python version
- `ffmpeg -version` output
- The exact command you ran
- The full error/traceback

## License
By contributing, you agree that your contributions will be licensed under the project’s MIT License.
