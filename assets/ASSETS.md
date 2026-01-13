## Transcriber brand assets

- `assets/icon.svg`: square app icon (works well at 1024×1024)
- `assets/logo.svg`: logo lockup (icon + wordmark)

### Exporting for app packaging

PyInstaller prefers:
- Windows: `.ico`
- macOS: `.icns`

Suggested conversion tools:
- Inkscape (cross-platform)
- ImageMagick

Example (ImageMagick) exports:

```bash
# Windows ICO (multi-size)
magick assets/icon.svg -define icon:auto-resize=256,128,64,48,32,16 dist/icon.ico

# macOS ICNS (requires iconutil workflow; easiest via a GUI exporter)
```

