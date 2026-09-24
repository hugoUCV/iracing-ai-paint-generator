"""
AI image generation and vision analysis backends.

Supports (in priority order):
  1. xAI Grok Imagine  — set XAI_API_KEY
  2. OpenAI DALL-E 3   — set OPENAI_API_KEY
  3. Stability AI      — set STABILITY_API_KEY
  4. Pollinations Flux — free, no API key required
  5. Local procedural fallback (no API key required)
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import requests
from PIL import Image, ImageChops, ImageDraw, ImageFont

from cars_config import IRacingCar
from template_manager import CarTemplate, get_car_template
from regional_paint import apply_regional_overrides, regional_override_summary
from uv_atlas import format_atlas_reference, resolve_regions_from_prompt

logger = logging.getLogger(__name__)

# Vision model for reference-photo analysis (xAI image understanding).
XAI_VISION_MODEL = os.getenv("XAI_VISION_MODEL", "grok-4.3")

_PLACEHOLDER_KEY_MARKERS = (
    "your-key-here",
    "your_key_here",
    "changeme",
    "placeholder",
    "example",
    "xxx",
)


def _is_configured_api_key(key: str, *, prefix: str = "") -> bool:
    """Ignore empty values and .env.example placeholders."""
    value = (key or "").strip()
    if not value or len(value) < 12:
        return False
    if prefix and not value.startswith(prefix):
        return False
    low = value.lower()
    return not any(marker in low for marker in _PLACEHOLDER_KEY_MARKERS)


def _extract_response_text(response: object) -> str:
    """Normalize text from OpenAI/xAI response objects."""
    for attr in ("output_text", "text"):
        value = getattr(response, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()

    output = getattr(response, "output", None)
    if isinstance(output, list):
        chunks: list[str] = []
        for item in output:
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for part in content:
                    text = getattr(part, "text", None)
                    if isinstance(text, str) and text.strip():
                        chunks.append(text.strip())
            elif isinstance(content, str) and content.strip():
                chunks.append(content.strip())
        if chunks:
            return "\n".join(chunks)

    choices = getattr(response, "choices", None)
    if choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None) if message else None
        if isinstance(content, str) and content.strip():
            return content.strip()
    return ""


# Keywords used to infer materials for spec-map generation.
MATTE_KEYWORDS = ("matte", "flat", "satin", "primer")
GLOSS_KEYWORDS = ("gloss", "glossy", "wet", "clear", "lacquer")
METALLIC_KEYWORDS = ("metallic", "chrome", "silver", "gold", "pearl", "foil")
CHROME_KEYWORDS = ("chrome", "mirror", "polished")


@dataclass
class GenerationResult:
    """Output from any generation backend."""

    image: Image.Image
    backend: str
    prompt_used: str
    reference_analysis: str = ""


@dataclass
class PromptConstraints:
    """User intent flags parsed from the prompt and UI options."""

    no_text: bool = False
    no_logos: bool = False
    allow_car_number: bool = False


_NO_TEXT_RE = re.compile(
    r"\b(?:"
    r"no\s+text|without\s+text|no\s+words?|no\s+lettering|no\s+writing|"
    r"no\s+typography|textless|zero\s+text|"
    r"leave\s+out\s+(?:the\s+)?text|leave\s+text\s+out|"
    r"don'?t\s+(?:include|add|put|use)\s+(?:any\s+)?text|"
    r"do\s+not\s+(?:include|add|put|use)\s+(?:any\s+)?text|"
    r"avoid\s+(?:any\s+)?text|"
    r"no\s+(?:sponsor\s+)?(?:logos?|branding|lettering|typography)"
    r")\b",
    re.I,
)
_NO_LOGO_RE = re.compile(
    r"\b(?:"
    r"no\s+(?:sponsor\s+)?logos?|without\s+logos?|no\s+sponsors?|"
    r"no\s+branding|no\s+decals?"
    r")\b",
    re.I,
)
_NO_NUMBER_RE = re.compile(r"\bno\s+(?:car\s+)?numbers?\b", re.I)


def parse_prompt_constraints(
    user_prompt: str,
    *,
    no_text_option: bool = False,
) -> PromptConstraints:
    """Detect negative instructions the user typed (and optional UI toggle)."""
    text = user_prompt or ""
    no_text = no_text_option or bool(_NO_TEXT_RE.search(text))
    no_logos = bool(_NO_LOGO_RE.search(text))
    has_number = extract_car_number(text) is not None
    allow_car_number = has_number and not bool(_NO_NUMBER_RE.search(text))
    if no_text:
        no_logos = True
    return PromptConstraints(
        no_text=no_text,
        no_logos=no_logos,
        allow_car_number=allow_car_number,
    )


def _image_to_base64_uri(image: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def _download_image(url: str) -> Image.Image:
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGBA")


def _decode_b64_image(data: str) -> Image.Image:
    raw = base64.b64decode(data)
    return Image.open(io.BytesIO(raw)).convert("RGBA")


def build_livery_prompt(
    user_prompt: str,
    car: IRacingCar,
    template: Optional[CarTemplate] = None,
    reference_analysis: str = "",
    creativity: float = 0.7,
    uses_official_template: bool = True,
    constraints: Optional[PromptConstraints] = None,
) -> str:
    """
    Craft a generation prompt optimized for the official iRacing UV template.
    """
    style_intensity = "bold and creative" if creativity > 0.6 else "clean and professional"
    ref_block = ""
    if reference_analysis.strip():
        ref_block = reference_analysis

    targeted_block = ""
    atlas_block = ""
    if template is not None and template.uv_atlas is not None:
        targeted = resolve_regions_from_prompt(user_prompt, template.uv_atlas)
        atlas_block = "\n" + format_atlas_reference(template.uv_atlas, targeted) + "\n"
        override_block = regional_override_summary(user_prompt, template.uv_atlas)
        if override_block:
            atlas_block += "\n" + override_block + "\n"
        if targeted:
            names = ", ".join(r.label for r in targeted)
            targeted_block = (
                f"IMPORTANT: The user specifically mentioned these UV regions: {names}. "
                f"Apply their color/design instructions to THOSE body areas only, "
                f"using the UV REGION MAP coordinates below. "
            )

    template_rules = ""
    if uses_official_template:
        template_rules = (
            "CRITICAL: The reference image is the official FLAT 2D UV UNWRAP (2048x2048) — "
            "NOT a side-view or 3D render of a car. "
            "Cyan lines = panel edges on gray paintable UV islands. "
            "Use the UV REGION MAP text below to know which panel is which "
            "(e.g. HOOD, LEFT DOOR, REAR BUMPER). Never render region coordinates "
            "as visible rectangles or frames. "
            "You MUST keep the exact same panel positions as the reference — do NOT redraw as a side-view car. "
            "Put each graphic on the correct UV panel using the region map coordinates. "
            "Paint one continuous seamless livery — do NOT draw borders, frames, rectangles, "
            "or outlines around UV regions, even in matching colors. "
            "Cover every gray paintable panel. Leave non-panel areas solid black. "
            "OUTPUT RULES: Deliver clean livery paint ONLY — no cyan wireframe, no labels, "
            "no boxes, no panel frames, no gray template fill, no guide text. Same flat layout, paint only. "
        )

    if constraints is None:
        constraints = parse_prompt_constraints(user_prompt)

    number_hint = ""
    if constraints.allow_car_number:
        car_number = extract_car_number(user_prompt)
        if car_number:
            number_hint = (
                f"Car number {car_number} prominently displayed on the door "
                f"and other appropriate body panels. "
            )

    if constraints.no_text:
        graphics_hint = (
            "GRAPHICS ONLY: Use abstract patterns, stripes, gradients, weather effects, "
            "and icon-style shapes on each UV panel — absolutely NO readable text of any kind. "
        )
        text_block = (
            "CRITICAL — NO TEXT: Do NOT include any words, letters, numbers, typography, "
            "sponsor names, logos with lettering, decals with writing, or faux text blocks. "
            "The livery must be purely visual graphics with zero legible characters. "
        )
    else:
        graphics_hint = (
            "Include sponsor logos, stripes, and racing graphics aligned to each UV panel. "
        )
        text_block = ""

    if ref_block.strip():
        ref_block = (
            "\nReference style notes (COLORS and MOOD only — do NOT change the UV layout):\n"
            f"{reference_analysis.strip()}\n"
        )

    layout_lock = (
        "FINAL — UV LAYOUT LOCK: Output the SAME flat 2048x2048 UV unwrap as the reference image. "
        "Keep identical panel positions, island shapes, and black gaps between panels. "
        "Paint inside each UV panel only. "
        "NEVER draw a side-view car, 3/4 perspective, or single car silhouette — "
        "this is a texture sheet, not a picture of a car."
    )

    return (
        f"Professional iRacing sim-racing livery on the official UV template for a {car.display_name}. "
        f"{template_rules}"
        f"{targeted_block}"
        f"{atlas_block}"
        f"Flat orthographic UV unwrap — this is NOT a side-view car picture. "
        f"{style_intensity} race car design. "
        f"{number_hint}"
        f"{graphics_hint}"
        f"Sharp vector-like graphics, no 3D car render, no background scenery. "
        f"Colors must be vivid and print-ready for iRacing. "
        f"{user_prompt}. "
        f"{text_block}"
        f"{ref_block}"
        f"{layout_lock}"
    )


def _build_vision_prompt(
    user_prompt: str,
    constraints: PromptConstraints,
) -> str:
    vision_prompt = (
        "You are an expert iRacing livery painter. Analyze this race car photo "
        "and describe everything needed to recreate the livery as a sim-racing paint:\n"
        "- Primary, secondary, and accent colors (name exact shades)\n"
        "- Stripe patterns, gradients, and geometric shapes\n"
    )
    if constraints.no_text:
        vision_prompt += (
            "- Graphic motifs and visual theme ONLY — ignore all sponsor text and lettering\n"
            "- Do NOT describe any words, logos with text, or car numbers\n"
        )
    else:
        vision_prompt += (
            "- Sponsor logos and text (describe placement on hood, doors, rear)\n"
            "- Car number style, size, and location\n"
        )
    vision_prompt += (
        "- Material finishes: matte vs gloss vs metallic/chrome areas\n"
        "- Overall theme and mood\n"
        "Be concise but specific. Output plain text only."
    )
    if user_prompt.strip():
        vision_prompt += f"\nUser also wants: {user_prompt}"
    if constraints.no_text:
        vision_prompt += (
            "\nIMPORTANT: The user requested NO TEXT on the final livery. "
            "Describe colors, patterns, and mood only — never suggest adding words or numbers."
        )
    return vision_prompt


def _analyze_with_xai_sdk(
    image: Image.Image,
    vision_prompt: str,
    api_key: str,
) -> str:
    from xai_sdk import Client
    from xai_sdk.chat import image as xai_image
    from xai_sdk.chat import user as xai_user

    client = Client(api_key=api_key, timeout=3600)
    chat = client.chat.create(model=XAI_VISION_MODEL)
    chat.append(
        xai_user(
            vision_prompt,
            xai_image(image_url=_image_to_base64_uri(image), detail="high"),
        )
    )
    response = chat.sample()
    text = getattr(response, "content", None) or _extract_response_text(response)
    if isinstance(text, str) and text.strip():
        return text.strip()
    raise RuntimeError("xAI vision returned empty content")


def _analyze_with_xai_responses(
    image: Image.Image,
    vision_prompt: str,
    api_key: str,
) -> str:
    import httpx
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1",
        timeout=httpx.Timeout(3600.0),
    )
    b64_uri = _image_to_base64_uri(image)
    response = client.responses.create(
        model=XAI_VISION_MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": b64_uri,
                        "detail": "high",
                    },
                    {
                        "type": "input_text",
                        "text": vision_prompt,
                    },
                ],
            }
        ],
    )
    text = _extract_response_text(response)
    if text:
        return text
    raise RuntimeError("xAI responses API returned empty content")


def analyze_reference_image(
    image: Optional[Image.Image],
    user_prompt: str = "",
    constraints: Optional[PromptConstraints] = None,
) -> str:
    """
    Use vision AI to describe a reference race-car photo.
    Falls back to basic color analysis if no API key is available.
    """
    if image is None:
        return ""

    xai_key = os.getenv("XAI_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")

    if constraints is None:
        constraints = parse_prompt_constraints(user_prompt)

    vision_prompt = _build_vision_prompt(user_prompt, constraints)

    # --- xAI vision (grok-4.3 image understanding) ---
    if _is_configured_api_key(xai_key, prefix="xai-"):
        for label, fn in (
            ("xAI SDK", _analyze_with_xai_sdk),
            ("xAI responses", _analyze_with_xai_responses),
        ):
            try:
                return fn(image, vision_prompt, xai_key)
            except ImportError:
                continue
            except Exception as exc:
                logger.warning("xAI vision via %s failed: %s", label, exc)

    # --- OpenAI vision (only when a real key is configured) ---
    if _is_configured_api_key(openai_key, prefix="sk-"):
        try:
            from openai import OpenAI

            client = OpenAI(api_key=openai_key)
            b64_uri = _image_to_base64_uri(image)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": vision_prompt},
                            {"type": "image_url", "image_url": {"url": b64_uri}},
                        ],
                    }
                ],
                temperature=0.3,
            )
            text = _extract_response_text(response)
            if text:
                return text
        except Exception as exc:
            logger.warning("OpenAI vision failed: %s", exc)

    # --- Local color fallback ---
    logger.info("Using local color analysis for reference image (no vision API available).")
    return _local_color_analysis(image)


def _local_color_analysis(image: Image.Image) -> str:
    """Extract dominant colors when no vision API is available."""
    thumb = image.convert("RGB").resize((128, 128))
    arr = np.array(thumb).reshape(-1, 3)
    # Simple k-means-ish: bucket by quantized color
    quantized = (arr // 32) * 32
    unique, counts = np.unique(quantized, axis=0, return_counts=True)
    top = unique[np.argsort(counts)[-5:][::-1]]
    color_names = []
    for rgb in top:
        r, g, b = rgb
        color_names.append(f"RGB({r},{g},{b})")
    return (
        "Reference image color analysis (local fallback):\n"
        f"Dominant colors: {', '.join(color_names)}.\n"
        "Recreate stripe patterns and sponsor placement from these colors."
    )


def _generate_xai(
    prompt: str,
    reference_image: Optional[Image.Image] = None,
    creativity: float = 0.7,
) -> Image.Image:
    """Generate via xAI Grok Imagine API."""
    api_key = os.getenv("XAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("XAI_API_KEY not set")

    # Prefer official xAI SDK when installed.
    try:
        import xai_sdk

        client = xai_sdk.Client(api_key=api_key)
        kwargs: dict = {
            "prompt": prompt,
            "model": "grok-imagine-image",
        }
        if reference_image is not None:
            kwargs["image_url"] = _image_to_base64_uri(reference_image)
        response = client.image.sample(**kwargs)
        if hasattr(response, "url") and response.url:
            return _download_image(response.url)
        if hasattr(response, "image") and response.image:
            return _decode_b64_image(response.image)
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("xai_sdk failed, trying REST: %s", exc)

    # REST fallback (OpenAI-compatible endpoint).
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
    if reference_image is not None:
        # Image edit endpoint
        buf = io.BytesIO()
        reference_image.convert("RGB").save(buf, format="PNG")
        buf.seek(0)
        response = client.images.edit(
            model="grok-imagine-image",
            image=buf,
            prompt=prompt,
        )
    else:
        response = client.images.generate(
            model="grok-imagine-image",
            prompt=prompt,
            n=1,
        )

    item = response.data[0]
    if item.b64_json:
        return _decode_b64_image(item.b64_json)
    if item.url:
        return _download_image(item.url)
    raise RuntimeError("xAI returned no image data")


def _generate_openai(prompt: str, creativity: float = 0.7) -> Image.Image:
    """Generate via OpenAI DALL-E 3."""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt[:4000],
        size="1024x1024",
        quality="hd",
        n=1,
    )
    url = response.data[0].url
    return _download_image(url)


def _generate_stability(prompt: str, creativity: float = 0.7) -> Image.Image:
    """Generate via Stability AI SD3."""
    api_key = os.getenv("STABILITY_API_KEY", "")
    if not api_key:
        raise RuntimeError("STABILITY_API_KEY not set")

    resp = requests.post(
        "https://api.stability.ai/v2beta/stable-image/generate/sd3",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "image/*",
        },
        files={"none": ""},
        data={
            "prompt": prompt[:10000],
            "output_format": "png",
            "aspect_ratio": "1:1",
        },
        timeout=180,
    )
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGBA")


def _build_flat_livery_prompt(
    user_prompt: str,
    car: IRacingCar,
    reference_analysis: str = "",
    constraints: Optional[PromptConstraints] = None,
) -> str:
    """Concise design-only prompt for text-to-image backends (no UV instructions).

    Text-to-image models cannot see the UV template, so the verbose UV-layout
    rules are dropped here. Keeps the prompt short enough for a GET URL and
    focused on the visual design (colors, theme, graphics, text policy).
    """
    if constraints is None:
        constraints = parse_prompt_constraints(user_prompt)

    parts = [
        f"A flat race car livery texture sheet for a {car.display_name}, "
        "painted on a plain dark background, top-down flat layout — "
        "NOT a 3D car render and NOT a side-view photo.",
        f"Livery design: {user_prompt}.",
    ]
    if constraints.no_text:
        parts.append(
            "Strictly NO text, NO letters, NO numbers, NO logos with writing — "
            "abstract graphics, stripes, and patterns only."
        )
    else:
        parts.append(
            "Include sponsor-style logos, racing numbers, and racing stripes."
        )
    if reference_analysis.strip():
        parts.append(
            f"Match these colors/style cues: {reference_analysis.strip()}"
        )
    parts.append("Bold, high-contrast, crisp vector-like racing graphics.")
    return " ".join(parts)


def _generate_pollinations(prompt: str, creativity: float = 0.7) -> Image.Image:
    """Generate via Pollinations.ai — free, no API key required (Flux model)."""
    import urllib.parse

    url = (
        "https://image.pollinations.ai/prompt/"
        + urllib.parse.quote(prompt)
        + "?width=1024&height=1024&nologo=true&model=flux"
    )
    return _download_image(url)


def extract_car_number(prompt: str) -> Optional[str]:
    """Pull car number from the user prompt only — never from Customer ID."""
    match = re.search(r"\bnumber\s+(\d{1,3})\b", prompt, re.I)
    if match:
        return match.group(1)
    match = re.search(r"#\s*(\d{1,3})\b", prompt)
    if match:
        return match.group(1)
    return None


def _extract_colors(prompt: str) -> list[tuple[int, int, int]]:
    """Map common color words in the prompt to RGB values."""
    color_map = {
        "black": (20, 20, 20),
        "white": (240, 240, 240),
        "red": (200, 30, 30),
        "blue": (30, 80, 200),
        "green": (30, 160, 60),
        "yellow": (230, 200, 30),
        "orange": (230, 120, 30),
        "purple": (120, 40, 180),
        "silver": (180, 180, 190),
        "gold": (200, 170, 50),
        "pink": (230, 100, 150),
        "cyan": (30, 200, 200),
        "navy": (20, 40, 100),
        "maroon": (100, 20, 30),
    }
    lower = prompt.lower()
    found = []
    for name, rgb in color_map.items():
        if name in lower:
            found.append(rgb)
    if not found:
        found = [(30, 30, 35), (200, 30, 30), (240, 240, 240)]
    return found[:4]


def _rect_from_region(
    region: tuple[float, float, float, float],
    size: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = region
    return (
        int(x0 * size),
        int(y0 * size),
        int(x1 * size),
        int(y1 * size),
    )


def _center_of_region(
    region: tuple[float, float, float, float],
    size: int,
) -> tuple[int, int]:
    x0, y0, x1, y1 = _rect_from_region(region, size)
    return (x0 + x1) // 2, (y0 + y1) // 2


def _generate_local_fallback(
    prompt: str,
    car: IRacingCar,
    customer_id: str,
    template: CarTemplate,
    size: int = 2048,
    constraints: Optional[PromptConstraints] = None,
) -> Image.Image:
    """
    Procedural livery generator used when no AI API key is configured.
    Creates a credible flat paint layout with stripes, number, and sponsors.
    """
    if constraints is None:
        constraints = parse_prompt_constraints(prompt)

    colors = _extract_colors(prompt)
    primary, secondary, accent = colors[0], colors[1], colors[2 % len(colors)]
    number = extract_car_number(prompt) if constraints.allow_car_number else None
    is_matte = any(k in prompt.lower() for k in MATTE_KEYWORDS)

    mask = template.paintable_mask.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    design = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    base = Image.new("RGBA", (size, size), primary + (255,))
    design.paste(base, mask=mask)
    draw = ImageDraw.Draw(design)

    targeted_regions: list[tuple[str, tuple[float, float, float, float]]] = []
    if template.uv_atlas is not None:
        for region in resolve_regions_from_prompt(prompt, template.uv_atlas):
            targeted_regions.append((region.id, region.bbox))

    panels = template.panel_regions or {}
    if targeted_regions:
        for region_id, bbox in targeted_regions:
            box = _rect_from_region(bbox, size)
            patch = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            ImageDraw.Draw(patch).rectangle(box, fill=accent + (255,), outline=secondary + (255,), width=4)
            design = Image.alpha_composite(design, patch)
    else:
        panel_items = list(panels.items())
        for i, (name, region) in enumerate(panel_items):
            box = _rect_from_region(region, size)
            fill = secondary if i % 2 == 0 else primary
            patch = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            ImageDraw.Draw(patch).rectangle(box, fill=fill + (255,), outline=accent + (255,), width=3)
            design = Image.alpha_composite(design, patch)
    draw = ImageDraw.Draw(design)

    # Diagonal racing stripes clipped to the UV mask.
    stripe_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    stripe_draw = ImageDraw.Draw(stripe_layer)
    stripe_w = 80
    for x in range(-size, size * 2, stripe_w * 3):
        stripe_draw.polygon(
            [(x, 0), (x + stripe_w, 0), (x + stripe_w + 400, size), (x + 400, size)],
            fill=accent + (200,),
        )
    stripe_layer.putalpha(ImageChops.multiply(stripe_layer.split()[3], mask))
    design = Image.alpha_composite(design, stripe_layer)
    draw = ImageDraw.Draw(design)

    sponsor_region = panels.get("rear_bumper") or panels.get("rear") or panels.get("front_bumper")
    if sponsor_region and not constraints.no_text:
        sx0, sy0, sx1, sy1 = _rect_from_region(sponsor_region, size)
        sponsor_h = max((sy1 - sy0) // 5, 80)
        sponsor_y = sy0 + (sy1 - sy0) // 2 - sponsor_h // 2
        slot_w = max((sx1 - sx0) // 4, 120)
        labels = ["GROK", "RACING", "SIM", "AI"]
        try:
            font = ImageFont.truetype("arial.ttf", max(28, sponsor_h // 3))
        except OSError:
            font = ImageFont.load_default()
        for i, label in enumerate(labels):
            x0 = sx0 + i * slot_w + 8
            if x0 + slot_w - 16 > sx1:
                break
            draw.rectangle(
                (x0, sponsor_y, x0 + slot_w - 16, sponsor_y + sponsor_h),
                fill=(255, 255, 255, 230),
            )
            draw.rectangle(
                (x0 + 4, sponsor_y + 4, x0 + slot_w - 20, sponsor_y + sponsor_h - 4),
                outline=accent + (255,),
                width=2,
            )
            draw.text((x0 + 18, sponsor_y + sponsor_h // 4), label, fill=primary + (255,), font=font)

    if number:
        door_region = None
        if template.uv_atlas is not None:
            door = template.uv_atlas.region_by_id("left_side_door")
            if door:
                door_region = door.bbox
        if door_region is None:
            door_region = panels.get("driver_side") or panels.get("passenger_side")
        try:
            num_font = ImageFont.truetype("arialbd.ttf", 220)
        except OSError:
            num_font = ImageFont.load_default()
        if door_region:
            x0, y0, x1, y1 = _rect_from_region(door_region, size)
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            num_bbox = draw.textbbox((0, 0), number, font=num_font, stroke_width=5)
            tw, th = num_bbox[2] - num_bbox[0], num_bbox[3] - num_bbox[1]
            draw.text(
                (cx - tw // 2, cy - th // 2),
                number,
                fill=accent + (255,),
                font=num_font,
                stroke_width=5,
                stroke_fill=(0, 0, 0, 255),
            )

    if is_matte:
        overlay = Image.new("RGBA", (size, size), (0, 0, 0, 30))
        design = Image.alpha_composite(design, overlay)

    try:
        small = ImageFont.truetype("arial.ttf", 28)
    except OSError:
        small = ImageFont.load_default()
    draw.text(
        (60, 30),
        f"DEMO MODE — {car.display_name} — Set XAI_API_KEY for AI generation",
        fill=(255, 200, 50, 255),
        font=small,
    )

    from paint_processor import (
        apply_paint_to_template,
        fill_unpainted_mask_areas,
        strip_guide_overlays,
    )

    design = apply_regional_overrides(design, prompt, template)
    design = strip_guide_overlays(design, template, passes=3)
    return apply_paint_to_template(design, template)


def generate_livery(
    user_prompt: str,
    car: IRacingCar,
    customer_id: str,
    reference_image: Optional[Image.Image] = None,
    creativity: float = 0.7,
    backend_preference: str = "auto",
    template: Optional[CarTemplate] = None,
    no_text: bool = False,
) -> GenerationResult:
    """
    Generate a livery image using the best available backend.
    Output is masked to the official iRacing UV template for the selected car.
    """
    from paint_processor import (
        apply_paint_to_template,
        fill_unpainted_mask_areas,
        strip_guide_overlays,
        strip_template_artifacts,
    )

    if template is None:
        template = get_car_template(car)

    constraints = parse_prompt_constraints(user_prompt, no_text_option=no_text)

    reference_analysis = ""
    if reference_image is not None:
        reference_analysis = analyze_reference_image(
            reference_image, user_prompt, constraints=constraints
        )

    full_prompt = build_livery_prompt(
        user_prompt,
        car,
        template=template,
        reference_analysis=reference_analysis,
        creativity=creativity,
        uses_official_template=True,
        constraints=constraints,
    )

    template_ref = template.ai_guide_image

    pollinations_prompt = _build_flat_livery_prompt(
        user_prompt, car, reference_analysis, constraints
    )

    backends: list[tuple[str, callable]] = []

    if backend_preference == "xai" or backend_preference == "auto":
        if _is_configured_api_key(os.getenv("XAI_API_KEY", ""), prefix="xai-"):
            backends.append(
                ("xAI Grok Imagine", lambda: _generate_xai(full_prompt, template_ref, creativity))
            )
    if backend_preference == "openai" or backend_preference == "auto":
        if _is_configured_api_key(os.getenv("OPENAI_API_KEY", ""), prefix="sk-"):
            backends.append(("OpenAI DALL-E 3", lambda: _generate_openai(full_prompt, creativity)))
    if backend_preference == "stability" or backend_preference == "auto":
        if _is_configured_api_key(os.getenv("STABILITY_API_KEY", ""), prefix="sk-"):
            backends.append(("Stability AI SD3", lambda: _generate_stability(full_prompt, creativity)))
    if backend_preference == "pollinations" or backend_preference == "auto":
        backends.append(
            ("Pollinations Flux (free)", lambda: _generate_pollinations(pollinations_prompt, creativity))
        )

    errors: list[str] = []
    for name, fn in backends:
        try:
            raw = fn()
            if raw.size != (template.resolution, template.resolution):
                raw = raw.resize(
                    (template.resolution, template.resolution),
                    Image.Resampling.LANCZOS,
                )
            cleaned = strip_template_artifacts(raw, template)
            cleaned = strip_guide_overlays(cleaned, template, passes=2)
            filled = fill_unpainted_mask_areas(cleaned, template)
            fitted = apply_paint_to_template(filled, template)
            return GenerationResult(
                image=fitted,
                backend=f"{name} + UV template ({template.source_zip})",
                prompt_used=full_prompt,
                reference_analysis=reference_analysis,
            )
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            logger.warning("Backend %s failed: %s", name, exc)

    local = _generate_local_fallback(
        full_prompt, car, customer_id, template, car.resolution, constraints=constraints
    )
    note = ""
    if errors:
        note = " API errors: " + "; ".join(errors)
    return GenerationResult(
        image=local,
        backend=f"Local Demo + UV template ({template.source_zip})" + note,
        prompt_used=full_prompt,
        reference_analysis=reference_analysis,
    )

def infer_material_hints(prompt: str, reference_analysis: str = "") -> dict:
    """Derive material flags from prompt text for spec-map generation."""
    combined = (prompt + " " + reference_analysis).lower()
    return {
        "matte": any(k in combined for k in MATTE_KEYWORDS),
        "gloss": any(k in combined for k in GLOSS_KEYWORDS),
        "metallic": any(k in combined for k in METALLIC_KEYWORDS),
        "chrome": any(k in combined for k in CHROME_KEYWORDS),
    }