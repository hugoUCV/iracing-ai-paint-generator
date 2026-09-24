"""
iRacing AI Paint Generator
==========================
Gradio web app that generates ready-to-use iRacing paint TGA files
from text prompts and optional reference photos.

Usage:
    pip install -r requirements.txt
    set XAI_API_KEY=your_key_here   # optional — demo mode works without it
    python app.py
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from PIL import Image

from ai_backend import generate_livery, infer_material_hints, parse_prompt_constraints
from cars_config import (
    CAR_BY_NAME,
    CAR_CHOICES,
    IRacingCar,
    build_install_instructions,
    build_trading_paints_instructions,
    get_output_filenames,
    get_paint_install_path,
    get_paint_install_paths,
)
from paint_processor import (
    OUTPUT_DIR,
    clip_paint_to_mask,
    create_template_preview,
    export_paint_files,
    fill_unpainted_mask_areas,
    generate_spec_map,
    get_session_output_dir,
    post_process_paint,
    strip_template_artifacts,
    strip_guide_overlays,
)
from iracing_preview import install_paint_for_preview, open_iracing_3d_preview
from regional_paint import apply_regional_overrides
from template_manager import get_car_template

load_dotenv()

# Log to stdout (not stderr) so PowerShell doesn't report false exit code 1 with 2>&1.
import sys

logging.basicConfig(level=logging.WARNING, stream=sys.stdout)
for noisy in ("httpx", "httpcore", "gradio", "urllib3", "filelock"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

APP_TITLE = "iRacing AI Paint Generator"
APP_DIR = Path(__file__).parent


def validate_customer_id(customer_id: str) -> tuple[bool, str]:
    """iRacing Customer IDs are numeric."""
    cid = str(customer_id).strip()
    if not cid:
        return False, "Customer ID is required."
    if not re.fullmatch(r"\d{4,8}", cid):
        return False, "Customer ID must be 4–8 digits (your iRacing account number)."
    return True, cid


def _empty_result(error_msg: str) -> tuple:
    """Return outputs matching all Gradio output components."""
    return None, None, None, f"**Error:** {error_msg}", "", None


def _last_gen_record(car_name: str, customer_id: str) -> dict[str, str]:
    return {"car_name": car_name, "customer_id": str(customer_id).strip()}


def generate_paint(
    car_name: str,
    customer_id: str,
    prompt: str,
    no_text: bool,
    reference_image: Image.Image | None,
    creativity: float,
    generate_spec: bool,
    install_to_iracing: bool,
    backend: str,
    progress=gr.Progress(),
) -> tuple:
    """
    Main generation handler called by the Gradio UI.
    Returns preview, downloads, status text, and instructions.
    """
    try:
        progress(0.05, desc="Validating inputs…")

        ok, cid_or_err = validate_customer_id(customer_id)
        if not ok:
            return _empty_result(cid_or_err)

        if not prompt or not prompt.strip():
            return _empty_result("Please enter a livery description.")

        car: IRacingCar = CAR_BY_NAME[car_name]
        customer_id = cid_or_err

        progress(0.10, desc="Loading official iRacing UV template…")
        template = get_car_template(car)

        progress(0.20, desc="Analyzing reference image…")
        gen = generate_livery(
            user_prompt=prompt.strip(),
            car=car,
            customer_id=customer_id,
            reference_image=reference_image,
            no_text=no_text,
            creativity=creativity,
            backend_preference=backend,
            template=template,
        )

        progress(0.55, desc="Post-processing paint…")
        # gen.image was already stripped + masked in generate_livery — avoid a second
        # full color-heuristic pass that can erase blue/cyan livery (e.g. lightning).
        paint = post_process_paint(gen.image)
        paint = fill_unpainted_mask_areas(paint, template)
        progress(0.62, desc="Applying regional paint instructions…")
        paint = apply_regional_overrides(paint, prompt.strip(), template, no_text=no_text)
        progress(0.66, desc="Removing labels, zone boxes, and wireframe…")
        paint = strip_guide_overlays(paint, template, passes=3)
        paint = fill_unpainted_mask_areas(paint, template)
        paint = clip_paint_to_mask(paint, template)

        progress(0.70, desc="Generating spec map…")
        materials = infer_material_hints(prompt, gen.reference_analysis)
        spec = (
            generate_spec_map(paint, materials, template=template)
            if generate_spec
            else None
        )

        progress(0.85, desc="Exporting TGA files…")
        paths = export_paint_files(
            paint=paint,
            spec=spec if spec else generate_spec_map(paint, materials, template=template),
            car=car,
            customer_id=customer_id,
            install_to_iracing=install_to_iracing,
            template=template,
        )

        template_preview = create_template_preview(paint, template)

        constraints = parse_prompt_constraints(prompt.strip(), no_text_option=no_text)
        status_lines = [
            "**Paint generated successfully!**",
            f"- Backend: `{gen.backend}`",
            f"- Template: `{template.source_zip}` (official iRacing UV)",
            f"- Car: **{car.display_name}**",
            f"- Files: `car_{customer_id}.tga`"
            + (f" + `car_spec_{customer_id}.tga`" if generate_spec else ""),
            f"- Output folder: `{paths['session_dir']}`",
        ]
        if constraints.no_text:
            status_lines.append(
                "- **No text/logos mode** — AI instructed for graphics-only livery"
                + (" (car number still allowed)" if constraints.allow_car_number else "")
            )
        if template.uv_atlas is not None:
            from regional_paint import regional_override_summary

            status_lines.append(
                f"- UV atlas: **{len(template.uv_atlas.regions)} body regions** "
                f"(unlabeled wireframe sent to AI; labels stripped from output)"
            )
            override_note = regional_override_summary(prompt.strip(), template.uv_atlas)
            if override_note:
                status_lines.append("\n" + override_note)
        if gen.reference_analysis:
            status_lines.append("\n**Reference analysis:**\n" + gen.reference_analysis[:800])
        if install_to_iracing:
            paint_name, spec_name = get_output_filenames(customer_id)
            verified_dirs = []
            for install_dir in get_paint_install_paths(car, customer_id):
                installed_paint = install_dir / paint_name
                if installed_paint.exists():
                    verified_dirs.append(install_dir)
            if verified_dirs:
                status_lines.append("\n**Installed to iRacing:**")
                for install_dir in verified_dirs:
                    status_lines.append(f"- `{install_dir}`")
            elif "install_paint" in paths:
                status_lines.append(
                    f"\n**Installed to iRacing:** `{paths['install_paint'].parent}`"
                )
            else:
                primary = get_paint_install_path(car, customer_id)
                status_lines.append(
                    f"\n**Warning:** Auto-install was enabled but files were not "
                    f"found under `{primary}`. Use **Copy to iRacing Folder** or "
                    f"check folder permissions."
                )

        instructions = (
            build_install_instructions(car, customer_id)
            + "\n"
            + build_trading_paints_instructions(car, customer_id)
        )

        progress(1.0, desc="Done!")
        paint_file = str(paths["paint"])
        spec_file = str(paths["spec"]) if generate_spec else None
        return (
            template_preview,
            paint_file,
            spec_file,
            "\n".join(status_lines),
            instructions,
            _last_gen_record(car_name, customer_id),
        )
    except Exception as exc:
        logger.exception("Paint generation failed")
        return _empty_result(str(exc))


def copy_to_iracing(
    car_name: str,
    customer_id: str,
    last_generated: dict[str, str] | None,
) -> str:
    """Copy the latest generated TGAs into the iRacing paint directory."""
    if last_generated and last_generated.get("car_name"):
        car_name = last_generated["car_name"]
        customer_id = last_generated.get("customer_id") or customer_id

    ok, cid_or_err = validate_customer_id(customer_id)
    if not ok:
        return f"**Error:** {cid_or_err}"

    if car_name not in CAR_BY_NAME:
        return f"**Error:** Unknown car `{car_name}` — generate a paint first."

    car = CAR_BY_NAME[car_name]
    session_dir = get_session_output_dir(car, cid_or_err)
    try:
        dest_dirs = install_paint_for_preview(car, cid_or_err)
    except FileNotFoundError as exc:
        return (
            f"**Error:** {exc}\n\n"
            f"Session output folder: `{session_dir}`"
        )
    except OSError as exc:
        return f"**Error:** {exc}"

    paint_name, spec_name = get_output_filenames(cid_or_err)
    lines = [
        "**Copied to iRacing paint folder**",
        f"- Car: **{car.display_name}** (Customer ID **{cid_or_err}**)",
        f"- Source: `{session_dir}`",
        "",
        "**Destinations:**",
    ]
    for dest_dir in dest_dirs:
        paint_dest = dest_dir / paint_name
        spec_dest = dest_dir / spec_name
        lines.append(f"- `{dest_dir}`")
        lines.append(
            f"  - `{paint_dest}`"
            + (" ✓" if paint_dest.exists() else " (missing)")
        )
        if spec_dest.exists():
            lines.append(f"  - `{spec_dest}` ✓")
    lines.append(
        "\nIf iRacing still shows the old paint, restart the sim — .mip cache was cleared."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
.gradio-container { max-width: 1400px !important; }
.sidebar-panel { background: #1a1d23; border-radius: 12px; padding: 8px; }
.main-panel { background: #12151a; border-radius: 12px; }
#generate-btn { background: linear-gradient(135deg, #e10600, #ff4444) !important; font-weight: 700 !important; }
.status-box { font-size: 0.95em; }
"""

with gr.Blocks(title=APP_TITLE) as demo:
    gr.Markdown(
        f"""
# {APP_TITLE}
Generate **ready-to-use iRacing paint TGA files** from a text prompt and optional reference photo.
Outputs `car_<CustomerID>.tga` and `car_spec_<CustomerID>.tga` for direct use in
`Documents\\iRacing\\paint\\<car-folder>\\` or Trading Paints upload.
        """
    )

    last_generated_state = gr.State(value=None)

    with gr.Row(equal_height=False):
        # ---- Sidebar inputs ----
        with gr.Column(scale=1, elem_classes=["sidebar-panel"]):
            gr.Markdown("### Inputs")

            default_car = next(
                (n for n in CAR_CHOICES if "Toyota Camry" in n and "Next Gen" in n),
                CAR_CHOICES[0],
            )
            car_dropdown = gr.Dropdown(
                choices=CAR_CHOICES,
                value=default_car,
                label="iRacing Car",
                info=f"{len(CAR_CHOICES)} cars — type to search.",
                filterable=True,
            )

            customer_id_input = gr.Textbox(
                label="iRacing Customer ID",
                placeholder="e.g. 414913",
                info="Used only for TGA filenames (car_<ID>.tga) — not painted on the car.",
            )

            prompt_input = gr.Textbox(
                label="Livery Prompt",
                placeholder='e.g. "thunderstorm theme, dark gray base, purple lightning bolts, no text"',
                lines=4,
            )

            no_text_cb = gr.Checkbox(
                value=False,
                label="No text or logos",
                info="Graphics only — blocks words, sponsor logos, and numbers unless you also specify a car number.",
            )

            reference_upload = gr.Image(
                label="Reference Photo (optional)",
                type="pil",
                sources=["upload", "clipboard"],
                height=200,
            )

            creativity_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.7,
                step=0.05,
                label="Creativity / Randomness",
                info="Higher = more artistic variation in the AI output.",
            )

            backend_dropdown = gr.Dropdown(
                choices=["auto", "pollinations", "xai", "openai", "stability"],
                value="auto",
                label="AI Backend",
                info="Auto: XAI → OpenAI → Stability → Pollinations (free) → Demo.",
            )

            generate_spec_cb = gr.Checkbox(
                value=True,
                label="Generate Spec Map (car_spec_<ID>.tga)",
                info="PBR spec: R=metallic, G=roughness, B=clearcoat, A=mask.",
            )

            install_cb = gr.Checkbox(
                value=False,
                label="Auto-install to iRacing folder",
                info="Copy TGAs directly into Documents\\iRacing\\paint\\ on generate.",
            )

            generate_btn = gr.Button(
                "Generate Paint",
                variant="primary",
                elem_id="generate-btn",
                size="lg",
            )

            gr.Markdown(
                f"""
---
**AI Backends** (set keys as environment variables in `.env`):
- `XAI_API_KEY` — Grok Imagine + vision (highest quality)
- `OPENAI_API_KEY` — DALL-E 3 + vision
- `STABILITY_API_KEY` — Stable Diffusion 3
- **Pollinations** — free, no key needed (Flux model)

With no API keys, **Auto** uses Pollinations (free), then Demo Mode.
                """
            )

        # ---- Main output area ----
        with gr.Column(scale=2, elem_classes=["main-panel"]):
            gr.Markdown("### Generated Output")

            preview_main = gr.Image(
                label="Paint Preview (exported TGA layout — no wireframe overlay)",
                type="pil",
                height=520,
            )

            status_output = gr.Markdown(label="Status", elem_classes=["status-box"])

            with gr.Row():
                download_paint = gr.File(label="Download car_<ID>.tga", interactive=False)
                download_spec = gr.File(label="Download car_spec_<ID>.tga", interactive=False)

            with gr.Row():
                preview_iracing_btn = gr.Button(
                    "Preview in iRacing 3D Viewer",
                    variant="primary",
                )
                copy_btn = gr.Button("Copy to iRacing Folder", variant="secondary")

            preview_iracing_status = gr.Markdown()
            copy_status = gr.Markdown()

            with gr.Accordion("Install & Trading Paints Instructions", open=True):
                instructions_output = gr.Markdown()

    # ---- Event wiring ----
    generate_btn.click(
        fn=generate_paint,
        inputs=[
            car_dropdown,
            customer_id_input,
            prompt_input,
            no_text_cb,
            reference_upload,
            creativity_slider,
            generate_spec_cb,
            install_cb,
            backend_dropdown,
        ],
        outputs=[
            preview_main,
            download_paint,
            download_spec,
            status_output,
            instructions_output,
            last_generated_state,
        ],
    )

    copy_btn.click(
        fn=copy_to_iracing,
        inputs=[car_dropdown, customer_id_input, last_generated_state],
        outputs=[copy_status],
    )

    def _preview_in_iracing(car_name, customer_id, last_generated):
        if last_generated and last_generated.get("car_name"):
            car_name = last_generated["car_name"]
            customer_id = last_generated.get("customer_id") or customer_id
        ok, cid_or_err = validate_customer_id(customer_id)
        if not ok:
            return f"**Error:** {cid_or_err}"
        if car_name not in CAR_BY_NAME:
            return "**Error:** Generate a paint first."
        car = CAR_BY_NAME[car_name]
        return open_iracing_3d_preview(car, cid_or_err)

    preview_iracing_btn.click(
        fn=_preview_in_iracing,
        inputs=[car_dropdown, customer_id_input, last_generated_state],
        outputs=[preview_iracing_status],
    )

    gr.Markdown(
        """
---
*Paint files are 2048×2048 RGBA TGA. Spec maps follow iRacing PBR channel layout.
For best results, download the official UV template from
[Trading Paints Car Templates](https://www.tradingpaints.com/cartemplates)
and refine the AI output in Photoshop/GIMP before racing.*
        """
    )

def _cleanup_stale_ports(ports: range = range(7860, 7870)) -> None:
    """Stop leftover python.exe servers blocking the default Gradio port range."""
    if os.getenv("CLEANUP_PORTS", "true").lower() in ("0", "false", "no"):
        return
    if os.name != "nt":
        return

    import subprocess

    try:
        netstat = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return

    current_pid = os.getpid()
    target_ports = {str(p) for p in ports}
    pids: set[int] = set()

    for line in netstat.stdout.splitlines():
        if "LISTENING" not in line:
            continue
        for port in target_ports:
            if f":{port} " in line:
                parts = line.split()
                try:
                    pids.add(int(parts[-1]))
                except (ValueError, IndexError):
                    pass

    for pid in pids:
        if pid in (0, current_pid):
            continue
        try:
            tasks = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if "python" not in tasks.stdout.lower():
                continue
            subprocess.run(
                ["taskkill", "/F", "/PID", str(pid)],
                capture_output=True,
                timeout=5,
                check=False,
            )
            print(f"Stopped stale python server (PID {pid})", flush=True)
        except Exception:
            pass


def _preferred_ports() -> list[int]:
    """Build an ordered list of ports to try when launching Gradio."""
    ports: list[int] = []
    for key in ("GRADIO_SERVER_PORT", "GRADIO_PORT"):
        raw = os.getenv(key, "").strip()
        if raw:
            preferred = int(raw)
            if preferred not in ports:
                ports.append(preferred)
    for candidate in range(7860, 7900):
        if candidate not in ports:
            ports.append(candidate)
    ports.append(0)  # 0 lets the OS assign any free port
    return ports


def _launch_demo(host: str) -> None:
    """Launch Gradio, retrying on the next port if the preferred one is taken."""
    launch_kwargs = dict(
        server_name=host,
        share=os.getenv("GRADIO_SHARE", "").lower() in ("1", "true", "yes"),
        show_error=True,
        inbrowser=os.getenv("GRADIO_INBROWSER", "false").lower() in ("1", "true", "yes"),
        allowed_paths=[str(OUTPUT_DIR.resolve())],
        theme=gr.themes.Base(
            primary_hue="red",
            secondary_hue="gray",
            neutral_hue="gray",
        ),
        css=CUSTOM_CSS,
    )

    last_error: Exception | None = None
    for port in _preferred_ports():
        label = "auto" if port == 0 else str(port)
        try:
            print(f"Starting {APP_TITLE} on port {label}…", flush=True)
            demo.launch(server_port=port, **launch_kwargs)
            return
        except OSError as exc:
            last_error = exc
            if "empty port" in str(exc).lower() or "address already in use" in str(exc).lower():
                print(f"Port {label} is busy, trying next…", flush=True)
                continue
            raise

    raise RuntimeError(f"Could not bind a free port. Last error: {last_error}")


if __name__ == "__main__":
    host = os.getenv("GRADIO_HOST", "127.0.0.1")
    _cleanup_stale_ports()
    print(f"Launching {APP_TITLE} — keep this window open while using the app.", flush=True)
    if not sys.stdin.isatty():
        print(
            "NOTE: Detected piped/non-interactive launch. If this exits unexpectedly, "
            "double-click Launch.vbs or run.bat from File Explorer instead.",
            flush=True,
        )
    _launch_demo(host)