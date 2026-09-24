# iRacing AI Paint Generator

Generate ready-to-use iRacing paint files (`car_<ID>.tga` and optional `car_spec_<ID>.tga`) from text prompts and reference photos. The app uses official iRacing UV templates, paintable masks, and wireframe guides so AI output maps correctly onto the car.

**Live docs:** [focustherage.github.io/iracing-ai-paint-generator](https://focustherage.github.io/iracing-ai-paint-generator/)

**Download:** [main.zip](https://github.com/FocusTheRage/iracing-ai-paint-generator/archive/refs/heads/main.zip) · [Releases](https://github.com/FocusTheRage/iracing-ai-paint-generator/releases)

## Features

- **183 iRacing cars** — catalog scraped from Trading Paints, with per-car UV wireframes
- **Constrained generation** — paints only on paintable UV islands; wireframe artifacts are stripped automatically
- **Regional overrides** — prompt keywords like `hood`, `doors`, `rear quarter` target specific panels (where atlas data exists)
- **Reference photo analysis** — optional vision pass extracts colors and style cues from a reference image
- **Spec map export** — optional PBR spec TGA (metallic / roughness / clearcoat)
- **One-click install** — auto-copy or manual **Copy to iRacing Folder** into `Documents\iRacing\paint\`
- **Dual-folder install (Windows)** — writes to both nested (`stockcars2\arcaford25`) and legacy flat (`stockcars2 arcaford25`) paths so paints show up regardless of how your iRacing install is laid out
- **iRacing 3D preview** — installs paint files and launches the iRacing UI viewer for an in-sim-quality preview
- **Demo mode** — procedural fallback when no API key is configured

## Requirements

- Windows 10/11 (primary target; Python stack is cross-platform)
- Python 3.10+
- An API key for at least one cloud backend (optional — Pollinations is free with no key, and demo mode needs no keys)

## Quick start

```bash
git clone https://github.com/FocusTheRage/iracing-ai-paint-generator.git
cd iracing-ai-paint-generator
python -m pip install -r requirements.txt
copy .env.example .env
# Edit .env and add your XAI_API_KEY (recommended)
python app.py
```

Open **http://127.0.0.1:7860** in your browser.

1. Pick a car, enter your **Customer ID**, and describe the livery.
2. Click **Generate Paint** — download TGAs or use **Copy to iRacing Folder**.
3. Optional: enable **Auto-install to iRacing folder** on generate, or **Preview in iRacing 3D Viewer** after export.

### Windows launchers

| File | Purpose |
|------|---------|
| `run.bat` | Opens the app in a new console window and launches the browser |
| `run.ps1` | PowerShell launcher (same behavior) |
| `Launch.vbs` | Detached launcher with a popup reminder |
| `stop.bat` | Stops processes listening on ports 7860–7869 |

## API backends

Set keys in `.env` (see `.env.example`):

| Variable | Backend |
|----------|---------|
| `XAI_API_KEY` | xAI Grok Imagine + vision (recommended) |
| `OPENAI_API_KEY` | OpenAI DALL-E 3 |
| `STABILITY_API_KEY` | Stability AI SD3 |
| *(none)* | Pollinations Flux — free, no key required |

**Auto** mode tries backends in that order, then Pollinations (free), then falls back to demo mode.

## First-run template cache

Wireframe PNGs ship with the repo. Per-car mask/guide caches are built on first use from iRacing’s official PSD templates (downloaded from iRacing CDN).

To pre-build every car and refresh wireframes:

```bash
python scripts/bootstrap_all_cars.py
```

To refresh the car catalog from Trading Paints:

```bash
python scripts/scrape_trading_paints_catalog.py
```

## Project layout

```
iracing-ai-paint-generator/
├── app.py                 # Gradio web UI
├── ai_backend.py          # Cloud AI + demo generation
├── iracing_preview.py     # Install paints + launch iRacing 3D viewer
├── paint_processor.py     # TGA export, spec maps, session output
├── template_manager.py    # iRacing PSD download & cache
├── cars_config.py         # Car list + iRacing paint install paths
├── data/                  # iracing_car_catalog.json
├── templates/
│   ├── wireframes/        # Per-car UV wireframe PNGs + manifest
│   ├── atlas/             # Regional UV atlas JSON (select cars)
│   └── cache/             # Generated masks/guides (gitignored)
├── scripts/               # Bootstrap & maintenance tools
└── output/                # Generated paints (gitignored)
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GRADIO_PORT` | `7860` | Web UI port |
| `GRADIO_HOST` | `127.0.0.1` | Bind address |
| `XAI_VISION_MODEL` | `grok-4.3` | Vision model for reference photos |

## iRacing install paths

Paints install under:

```
%USERPROFILE%\Documents\iRacing\paint\
```

Official iRacing paths use nested folders (e.g. `stockcars2\arcaford25`). Many Windows installs also keep a **legacy flat folder** with spaces (e.g. `stockcars2 arcaford25`). The app copies to **both** layouts and clears `.mip` cache files so iRacing reloads fresh TGAs.

After **Generate Paint** or **Copy to iRacing Folder**, the status panel lists every destination folder written, with checkmarks on each file.

| Control | What it does |
|---------|----------------|
| **Auto-install to iRacing folder** | Copies TGAs on generate |
| **Copy to iRacing Folder** | Copies the last generated paint for that car + Customer ID |
| **Preview in iRacing 3D Viewer** | Installs paints and opens iRacing UI — use Paint Shop / Car Model to view |

If the sim still shows an old design, restart iRacing after copying.

## Development scripts

| Script | Purpose |
|--------|---------|
| `scripts/bootstrap_all_cars.py` | Cache all car templates + export wireframes |
| `scripts/scrape_trading_paints_catalog.py` | Update `data/iracing_car_catalog.json` |
| `scripts/import_atlas_from_png.py` | Import regional UV atlas from labeled PNG |
| `scripts/setup_cup_camry_atlas.py` | Cup Camry atlas setup helper |
| `scripts/publish_to_github.ps1` | Create repo, push, and enable GitHub Pages |

## Publish to GitHub

```powershell
gh auth login
powershell -ExecutionPolicy Bypass -File scripts/publish_to_github.ps1
```

## License

MIT — see [LICENSE](LICENSE).

## Disclaimer

This project is not affiliated with iRacing.com Motorsport Simulations. iRacing templates are downloaded from iRacing’s public member CDN for personal paint creation, consistent with iRacing’s paint system.