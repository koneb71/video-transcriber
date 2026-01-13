from __future__ import annotations


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

