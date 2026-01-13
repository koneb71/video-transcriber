from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from transcriber.errors import CancelledError
from transcriber.pipeline import TranscriptionOutputs, transcribe_media_to_outputs


@dataclass(frozen=True)
class _UiResult:
    outputs: TranscriptionOutputs | None
    error: str | None


def main() -> int:
    app = _TranscriberApp()
    app.run()
    return 0


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _make_icon_photoimage(*, size: int = 128) -> tk.PhotoImage:
    """
    Generate an app icon in-code (no SVG/PNG dependencies).
    This is used for:
    - window icon (title bar / dock)
    - header logo mark
    """
    # Brand colors (match UI palette)
    c0 = _hex_to_rgb("#8B5CF6")  # violet-500
    c1 = _hex_to_rgb("#4F46E5")  # indigo-600
    wave = "#ede9fe"  # violet-50
    white = "#ffffff"
    shadow = "#0b1020"

    img = tk.PhotoImage(width=size, height=size)

    # Diagonal-ish gradient background
    for y in range(size):
        row: list[str] = []
        for x in range(size):
            # t in [0,1]
            t = (0.65 * x + 1.0 * y) / (1.65 * (size - 1))
            t = 0.0 if t < 0 else (1.0 if t > 1 else t)
            r = int(_lerp(c0[0], c1[0], t))
            g = int(_lerp(c0[1], c1[1], t))
            b = int(_lerp(c0[2], c1[2], t))
            row.append(_rgb_to_hex((r, g, b)))
        img.put("{" + " ".join(row) + "}", to=(0, y))

    # Helper to fill rectangles quickly
    def fill_rect(x0: int, y0: int, x1: int, y1: int, color: str) -> None:
        x0 = max(0, min(size, x0))
        y0 = max(0, min(size, y0))
        x1 = max(0, min(size, x1))
        y1 = max(0, min(size, y1))
        if x1 <= x0 or y1 <= y0:
            return
        img.put(color, to=(x0, y0, x1, y1))

    # Waveform bars
    cx = size // 2
    base_y0 = int(size * 0.30)
    base_y1 = int(size * 0.80)
    bar_w = max(2, int(size * 0.055))
    gap = max(2, int(size * 0.035))
    heights = [0.50, 0.72, 0.92, 1.00, 0.92, 0.72, 0.50]
    total_w = len(heights) * bar_w + (len(heights) - 1) * gap
    start_x = cx - total_w // 2
    for i, h in enumerate(heights):
        x0 = start_x + i * (bar_w + gap)
        x1 = x0 + bar_w
        mid = (base_y0 + base_y1) // 2
        half = int((base_y1 - base_y0) * 0.5 * h)
        y0 = mid - half
        y1 = mid + half
        fill_rect(x0, y0, x1, y1, wave)

    # "T" monogram with subtle shadow
    t_thick = max(3, int(size * 0.075))
    top_y = int(size * 0.27)
    bot_y = int(size * 0.82)
    left_x = int(size * 0.28)
    right_x = int(size * 0.72)
    stem_x0 = cx - t_thick // 2
    stem_x1 = stem_x0 + t_thick
    bar_y0 = top_y - t_thick // 2
    bar_y1 = bar_y0 + t_thick

    # Shadow offset
    s = max(1, size // 64)
    fill_rect(left_x + s, bar_y0 + s, right_x + s, bar_y1 + s, shadow)
    fill_rect(stem_x0 + s, top_y + s, stem_x1 + s, bot_y + s, shadow)

    # Foreground
    fill_rect(left_x, bar_y0, right_x, bar_y1, white)
    fill_rect(stem_x0, top_y, stem_x1, bot_y, white)

    return img


class _TranscriberApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Transcriber")
        self.root.minsize(980, 640)

        self._q: queue.Queue[tuple[str, object]] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._cancel_event = threading.Event()

        self._vars()
        self._style()
        self._layout()

        self._poll_queue()

    def _vars(self) -> None:
        self.input_path = tk.StringVar(value="")
        self.outdir = tk.StringVar(value=str(Path.cwd() / "output"))

        self.model = tk.StringVar(value="small")
        self.language = tk.StringVar(value="en")
        self.device = tk.StringVar(value="auto")
        self.compute_type = tk.StringVar(value="")  # empty = auto default
        self.beam_size = tk.IntVar(value=5)
        self.vad_filter = tk.BooleanVar(value=True)
        self.keep_wav = tk.BooleanVar(value=False)

        self.status = tk.StringVar(value="Ready")

    def _style(self) -> None:
        style = ttk.Style(self.root)
        # "clam" is consistently themeable across platforms.
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        # Subtle, modern palette (works with ttk widgets).
        self._c_bg = "#0f172a"  # slate-900
        self._c_panel = "#111c33"
        self._c_text = "#e5e7eb"  # gray-200
        self._c_muted = "#9ca3af"  # gray-400
        self._c_accent = "#7c3aed"  # violet-600
        self._c_border = "#223255"
        self._c_good = "#22c55e"
        self._c_bad = "#ef4444"

        self.root.configure(bg=self._c_bg)

        # App icon (keep a reference so Tk doesn't GC it)
        self._icon_img = _make_icon_photoimage(size=128)
        try:
            self.root.iconphoto(True, self._icon_img)
        except Exception:  # noqa: BLE001 - platform/theme differences
            pass

        default_font = ("SF Pro Display", 12) if sys.platform == "darwin" else ("Segoe UI", 11)
        heading_font = ("SF Pro Display", 20, "bold") if sys.platform == "darwin" else ("Segoe UI", 18, "bold")
        subheading_font = ("SF Pro Display", 11) if sys.platform == "darwin" else ("Segoe UI", 10)

        self._font_default = default_font
        self._font_heading = heading_font
        self._font_subheading = subheading_font

        style.configure(".", background=self._c_bg, foreground=self._c_text, font=self._font_default)
        style.configure("TFrame", background=self._c_bg)
        style.configure("Card.TFrame", background=self._c_panel, bordercolor=self._c_border, relief="solid")
        style.configure("TLabel", background=self._c_bg, foreground=self._c_text)
        style.configure("Muted.TLabel", background=self._c_bg, foreground=self._c_muted)
        style.configure("Card.TLabel", background=self._c_panel, foreground=self._c_text)
        style.configure("CardMuted.TLabel", background=self._c_panel, foreground=self._c_muted)

        style.configure("TEntry", fieldbackground="#0b1220", foreground=self._c_text, bordercolor=self._c_border)
        style.configure("TCombobox", fieldbackground="#0b1220", foreground=self._c_text, bordercolor=self._c_border)
        style.map("TCombobox", fieldbackground=[("readonly", "#0b1220")])

        style.configure("Primary.TButton", background=self._c_accent, foreground="white", bordercolor=self._c_accent)
        style.map(
            "Primary.TButton",
            background=[("active", "#6d28d9"), ("disabled", "#2b2d3b")],
            foreground=[("disabled", "#9ca3af")],
        )
        style.configure("TButton", background="#15213c", foreground=self._c_text, bordercolor=self._c_border)
        style.map("TButton", background=[("active", "#1b2b4e")])

        style.configure("Danger.TButton", background="#b91c1c", foreground="white", bordercolor="#b91c1c")
        style.map(
            "Danger.TButton",
            background=[("active", "#991b1b"), ("disabled", "#2b2d3b")],
            foreground=[("disabled", "#9ca3af")],
        )

        # Progressbar accent (best-effort; some platforms ignore).
        style.configure("Accent.Horizontal.TProgressbar", troughcolor="#0b1220", background=self._c_accent)

    def _layout(self) -> None:
        root = self.root

        container = ttk.Frame(root, padding=20)
        container.pack(fill="both", expand=True)

        header = ttk.Frame(container)
        header.pack(fill="x")

        # Logo lockup (icon + wordmark)
        brand = ttk.Frame(header)
        brand.pack(anchor="w")
        # Smaller mark for header
        self._logo_mark_img = _make_icon_photoimage(size=56)
        ttk.Label(brand, image=self._logo_mark_img).grid(row=0, column=0, rowspan=2, sticky="w", padx=(0, 14))
        ttk.Label(brand, text="Transcriber", font=self._font_heading).grid(row=0, column=1, sticky="w")
        ttk.Label(
            brand,
            text="Offline media transcription • ffmpeg → faster-whisper → timestamped text",
            style="Muted.TLabel",
            font=self._font_subheading,
        ).grid(row=1, column=1, sticky="w", pady=(4, 0))

        body = ttk.Frame(container)
        body.pack(fill="both", expand=True, pady=(16, 0))
        body.columnconfigure(0, weight=1, uniform="col")
        body.columnconfigure(1, weight=1, uniform="col")
        body.rowconfigure(0, weight=1)

        # Left: inputs/options
        left = ttk.Frame(body, style="Card.TFrame", padding=16)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        # Right: logs/results
        right = ttk.Frame(body, style="Card.TFrame", padding=16)
        right.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        right.rowconfigure(1, weight=1)

        # --- Left card content ---
        ttk.Label(left, text="Inputs", style="Card.TLabel", font=(self._font_default[0], self._font_default[1], "bold")).pack(
            anchor="w"
        )
        ttk.Label(left, text="Pick a media file and where outputs should go.", style="CardMuted.TLabel").pack(
            anchor="w", pady=(4, 10)
        )

        self._path_row(
            parent=left,
            label="Media file",
            var=self.input_path,
            browse_text="Choose…",
            browse=self._choose_input,
        )
        self._path_row(
            parent=left,
            label="Output folder",
            var=self.outdir,
            browse_text="Choose…",
            browse=self._choose_outdir,
        )

        ttk.Separator(left).pack(fill="x", pady=14)

        ttk.Label(left, text="Transcription", style="Card.TLabel", font=(self._font_default[0], self._font_default[1], "bold")).pack(
            anchor="w"
        )
        ttk.Label(left, text="Quality vs speed is mostly driven by model size.", style="CardMuted.TLabel").pack(
            anchor="w", pady=(4, 10)
        )

        form = ttk.Frame(left, style="Card.TFrame")
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)

        self._field_combo(form, row=0, label="Model", var=self.model, values=["tiny", "base", "small", "medium", "large-v3"])
        self._field_entry(form, row=1, label="Language", var=self.language)
        self._field_combo(form, row=2, label="Device", var=self.device, values=["auto", "cpu", "cuda", "metal"])
        self._field_entry(form, row=3, label="Compute type", var=self.compute_type)
        self._field_spin(form, row=4, label="Beam size", var=self.beam_size, from_=1, to=10)

        toggles = ttk.Frame(left, style="Card.TFrame")
        toggles.pack(fill="x", pady=(12, 0))
        ttk.Checkbutton(toggles, text="VAD filter (recommended)", variable=self.vad_filter).pack(anchor="w")
        ttk.Checkbutton(toggles, text="Keep extracted WAV next to outputs", variable=self.keep_wav).pack(anchor="w", pady=(4, 0))

        buttons = ttk.Frame(left, style="Card.TFrame")
        buttons.pack(fill="x", pady=(16, 0))
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)
        buttons.columnconfigure(2, weight=1)
        self.start_btn = ttk.Button(buttons, text="Start transcription", style="Primary.TButton", command=self._start)
        self.start_btn.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.stop_btn = ttk.Button(buttons, text="Stop", style="Danger.TButton", command=self._stop, state="disabled")
        self.stop_btn.grid(row=0, column=1, sticky="ew", padx=(8, 8))
        self.clear_btn = ttk.Button(buttons, text="Clear log", command=self._clear_log)
        self.clear_btn.grid(row=0, column=2, sticky="ew", padx=(8, 0))

        # --- Right card content ---
        ttk.Label(right, text="Progress", style="Card.TLabel", font=(self._font_default[0], self._font_default[1], "bold")).pack(
            anchor="w"
        )
        self.status_label = ttk.Label(right, textvariable=self.status, style="CardMuted.TLabel")
        self.status_label.pack(anchor="w", pady=(4, 8))

        self.progress = ttk.Progressbar(right, mode="indeterminate", style="Accent.Horizontal.TProgressbar")
        self.progress.pack(fill="x", pady=(0, 10))

        self.log = ScrolledText(
            right,
            height=14,
            wrap="word",
            bg="#0b1220",
            fg=self._c_text,
            insertbackground=self._c_text,
            selectbackground="#1f2a44",
            relief="flat",
            padx=12,
            pady=10,
        )
        self.log.pack(fill="both", expand=True)
        self.log.configure(state="disabled")

        out_actions = ttk.Frame(right, style="Card.TFrame")
        out_actions.pack(fill="x", pady=(12, 0))
        out_actions.columnconfigure(0, weight=1)
        out_actions.columnconfigure(1, weight=1)
        self.open_outdir_btn = ttk.Button(out_actions, text="Open output folder", command=self._open_outdir, state="disabled")
        self.open_outdir_btn.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.open_timestamps_btn = ttk.Button(out_actions, text="Open timestamps", command=self._open_timestamps, state="disabled")
        self.open_timestamps_btn.grid(row=0, column=1, sticky="ew", padx=(8, 0))

        self._last_outputs: TranscriptionOutputs | None = None

    def _path_row(self, *, parent: ttk.Frame, label: str, var: tk.StringVar, browse_text: str, browse: callable) -> None:
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x", pady=(0, 10))
        ttk.Label(row, text=label, style="CardMuted.TLabel").pack(anchor="w")
        inner = ttk.Frame(row, style="Card.TFrame")
        inner.pack(fill="x", pady=(6, 0))
        inner.columnconfigure(0, weight=1)
        entry = ttk.Entry(inner, textvariable=var)
        entry.grid(row=0, column=0, sticky="ew")
        ttk.Button(inner, text=browse_text, command=browse).grid(row=0, column=1, padx=(10, 0))

    def _field_entry(self, parent: ttk.Frame, *, row: int, label: str, var: tk.StringVar) -> None:
        ttk.Label(parent, text=label, style="CardMuted.TLabel").grid(row=row, column=0, sticky="w", pady=6, padx=(0, 10))
        e = ttk.Entry(parent, textvariable=var)
        e.grid(row=row, column=1, sticky="ew", pady=6)

    def _field_combo(self, parent: ttk.Frame, *, row: int, label: str, var: tk.StringVar, values: list[str]) -> None:
        ttk.Label(parent, text=label, style="CardMuted.TLabel").grid(row=row, column=0, sticky="w", pady=6, padx=(0, 10))
        cb = ttk.Combobox(parent, textvariable=var, values=values, state="readonly")
        cb.grid(row=row, column=1, sticky="ew", pady=6)

    def _field_spin(self, parent: ttk.Frame, *, row: int, label: str, var: tk.IntVar, from_: int, to: int) -> None:
        ttk.Label(parent, text=label, style="CardMuted.TLabel").grid(row=row, column=0, sticky="w", pady=6, padx=(0, 10))
        sp = ttk.Spinbox(parent, textvariable=var, from_=from_, to=to, width=6)
        sp.grid(row=row, column=1, sticky="w", pady=6)

    def _choose_input(self) -> None:
        filetypes = [
            ("Media files", "*.mp4 *.mov *.mkv *.webm *.avi *.mp3 *.m4a *.wav *.flac *.aac *.ogg"),
            ("All files", "*.*"),
        ]
        path = filedialog.askopenfilename(title="Choose a media file", filetypes=filetypes)
        if path:
            self.input_path.set(path)

    def _choose_outdir(self) -> None:
        path = filedialog.askdirectory(title="Choose output folder")
        if path:
            self.outdir.set(path)

    def _set_running(self, running: bool) -> None:
        if running:
            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
            self.clear_btn.configure(state="disabled")
            self.open_outdir_btn.configure(state="disabled")
            self.open_timestamps_btn.configure(state="disabled")
            self.progress.start(12)
        else:
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            self.clear_btn.configure(state="normal")
            self.progress.stop()

    def _stop(self) -> None:
        if self._worker and self._worker.is_alive():
            self._cancel_event.set()
            self.stop_btn.configure(state="disabled")
            self.status.set("Cancelling…")
            self._append_log("Cancellation requested…")

    def _start(self) -> None:
        if self._worker and self._worker.is_alive():
            return

        in_path = Path(self.input_path.get()).expanduser()
        if not in_path.exists():
            messagebox.showerror("Missing file", "Please choose an existing media file.")
            return

        outdir = Path(self.outdir.get()).expanduser()
        model = (self.model.get() or "small").strip()
        language = (self.language.get() or "en").strip()
        device = (self.device.get() or "auto").strip()
        compute_type_raw = (self.compute_type.get() or "").strip()
        compute_type = None if compute_type_raw.lower() in {"", "auto", "(auto)"} else compute_type_raw
        beam_size = int(self.beam_size.get())
        vad_filter = bool(self.vad_filter.get())
        keep_wav = bool(self.keep_wav.get())

        self._last_outputs = None
        self._cancel_event.clear()
        self._clear_log()
        self.status.set("Working…")
        self._set_running(True)

        def _log(msg: str) -> None:
            self._q.put(("log", msg))

        def _run() -> None:
            try:
                outputs = transcribe_media_to_outputs(
                    input_path=in_path,
                    outdir=outdir,
                    model=model,
                    language=language,
                    device=device,
                    compute_type=compute_type,
                    beam_size=beam_size,
                    vad_filter=vad_filter,
                    keep_wav=keep_wav,
                    on_log=_log,
                    cancel_event=self._cancel_event,
                )
                self._q.put(("done", _UiResult(outputs=outputs, error=None)))
            except CancelledError:
                self._q.put(("cancelled", None))
            except Exception:
                self._q.put(("done", _UiResult(outputs=None, error=traceback.format_exc())))

        self._worker = threading.Thread(target=_run, daemon=True)
        self._worker.start()

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self._q.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                elif kind == "done":
                    res: _UiResult = payload  # type: ignore[assignment]
                    self._on_done(res)
                elif kind == "cancelled":
                    self._on_cancelled()
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _on_cancelled(self) -> None:
        self._set_running(False)
        self.status.set("Cancelled")
        self._append_log("Cancelled.")

    def _on_done(self, res: _UiResult) -> None:
        self._set_running(False)
        if res.error:
            self.status.set("Failed")
            self._append_log("")
            self._append_log("Error:")
            self._append_log(res.error)
            messagebox.showerror("Transcription failed", "Something went wrong. Check the log for details.")
            return

        self._last_outputs = res.outputs
        self.status.set("Finished")
        if self._last_outputs:
            self.open_outdir_btn.configure(state="normal")
            self.open_timestamps_btn.configure(state="normal")
            self._append_log("")
            self._append_log(f"Outputs saved to: {self._last_outputs.outdir}")

    def _open_outdir(self) -> None:
        if not self._last_outputs:
            return
        self._open_path(self._last_outputs.outdir)

    def _open_timestamps(self) -> None:
        if not self._last_outputs:
            return
        self._open_path(self._last_outputs.timestamps_txt_path)

    def _open_path(self, path: Path) -> None:
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", str(path)], check=False)
            elif os.name == "nt":
                os.startfile(str(path))  # type: ignore[attr-defined]
            else:
                subprocess.run(["xdg-open", str(path)], check=False)
        except Exception as e:
            messagebox.showerror("Could not open", str(e))

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    raise SystemExit(main())

