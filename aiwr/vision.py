"""Optional local vision-language diagnosis for the existing per-leaf workflow.

Leaf detection and color-change severity still come from diagnostic.py. A
pretrained Qwen vision-language model supplies the initial disease choice.
No model weights are bundled with this repository.
"""

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
import json
from pathlib import Path
import re
from typing import Callable

import fitz
from PIL import Image, ImageOps

from .knowledge import RULES, sources_for


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_OPTIONS = (
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
)
MODEL_CACHE = ROOT / ".model_cache"
LEAF_LABELS = tuple(rule.name for rule in RULES if rule.tissue == "leaf")
ALLOWED_LABELS = frozenset((*LEAF_LABELS, "Healthy", "Unknown / Other"))


@dataclass(frozen=True)
class VisionFinding:
    diagnosis: str
    observed_cues: tuple[str, ...]
    secondary_candidates: tuple[str, ...]
    explanation: str


@lru_cache(maxsize=1)
def knowledge_context() -> str:
    """Read leaf criteria from the protocol and the accompanying CCM guide."""
    protocol = (ROOT / "docs" / "diagnostic_protocol.txt").read_text(encoding="utf-8")
    criteria = protocol.split("Part 1: Diagnostic Criteria", 1)[1].split(
        "Fusarium Head Blight", 1)[0].strip()
    guide = ROOT / "knowledge" / "WildRice_diseases_CCM_v2_compressed.pdf"
    excerpts = []
    if guide.exists():
        with fitz.open(guide) as document:
            for page_index in range(1, min(19, len(document))):
                page_text = " ".join(document[page_index].get_text().split())
                if page_text:
                    excerpts.append(f"{guide.name}, PDF page {page_index + 1}: {page_text[:650]}")
    return (
        "Diagnostic criteria from docs/diagnostic_protocol.txt, Part 1:\n"
        + criteria + "\n\nExtracted reference text (illustrations are not included):\n"
        + "\n".join(excerpts)
    )


def build_prompt() -> str:
    labels = ", ".join(LEAF_LABELS)
    return (
        "You are reviewing a crop of ONE candidate wild rice leaf. The crop may contain "
        "background or another leaf. Assess visible signs only. Do not claim species or "
        "pathogen confirmation, laboratory evidence, or calibrated confidence. "
        "Compare the image with the supplied wild rice criteria. This task asks for the "
        "nearest *visual pattern*, not a confirmed pathogen diagnosis. If visible spots, "
        "blotches, or streaks resemble a listed category, put the single best category "
        "in diagnosis even when other categories remain plausible; put alternatives in "
        "secondary_candidates and describe the uncertainty in explanation. Do not "
        "choose Unknown / Other merely because laboratory confirmation is absent. "
        "Use Unknown / Other only when the crop is unclear, is not a leaf, or has no "
        "discriminating visible pattern. The diagnosis must agree with the explanation. "
        "Choose exactly one diagnosis from: " + labels + ", Healthy, Unknown / Other. "
        "Healthy means no visible symptoms. List only cues actually visible in the image; "
        "do not copy possible symptoms from the reference text as observations. "
        "Return only a JSON object with keys diagnosis (string), observed_cues "
        "(array of short strings), secondary_candidates (array of names from the same "
        "list), and explanation (one short sentence describing visible evidence and "
        "uncertainty). Do not invent citations.\n\n" + knowledge_context()
    )


def parse_finding(output: str) -> VisionFinding:
    """Reject malformed or unsupported outputs instead of silently using rules."""
    match = re.search(r"\{.*\}", output, re.DOTALL)
    if not match:
        raise ValueError("Vision model did not return a JSON finding.")
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as exc:
        raise ValueError("Vision model returned invalid JSON.") from exc
    diagnosis = data.get("diagnosis")
    if diagnosis not in ALLOWED_LABELS:
        raise ValueError(f"Vision model returned an unsupported diagnosis: {diagnosis!r}.")
    cues = data.get("observed_cues")
    secondary = data.get("secondary_candidates")
    explanation = data.get("explanation")
    if not isinstance(cues, list) or not all(isinstance(item, str) for item in cues):
        raise ValueError("Vision model did not provide a valid observed_cues list.")
    if not isinstance(secondary, list) or not all(
        item in ALLOWED_LABELS and item != diagnosis for item in secondary
    ):
        raise ValueError("Vision model did not provide valid secondary candidates.")
    if not isinstance(explanation, str) or not explanation.strip():
        raise ValueError("Vision model did not provide an explanation.")
    return VisionFinding(diagnosis, tuple(cues[:6]), tuple(secondary[:3]), explanation.strip())


@lru_cache(maxsize=len(MODEL_OPTIONS))
def load_local_model(model_id: str = DEFAULT_MODEL):
    """Load weights only when vision mode is run on a CUDA-capable node."""
    if model_id not in MODEL_OPTIONS:
        raise ValueError(f"Unsupported vision model: {model_id}")
    try:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except (ImportError, OSError, ValueError) as exc:
        raise RuntimeError(
            "Vision mode needs a working PyTorch/Transformers install. See the README vision setup."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Vision mode requires a CUDA GPU session; this node has no usable GPU.")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=MODEL_CACHE)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype, cache_dir=MODEL_CACHE).to("cuda").eval()
    return processor, model


def generate_local(image: Image.Image, prompt: str, model_id: str = DEFAULT_MODEL) -> str:
    processor, model = load_local_model(model_id)
    import torch
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": prompt},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=300, do_sample=False)
    new_tokens = generated[0][inputs["input_ids"].shape[-1]:]
    return processor.decode(new_tokens, skip_special_tokens=True)


def _leaf_crop(image: Image.Image, mask: Image.Image, box) -> Image.Image:
    x, y, width, height = box
    pad = max(8, round(min(width, height) * 0.05))
    crop_box = (
        max(0, x - pad), max(0, y - pad),
        min(image.width, x + width + pad), min(image.height, y + height + pad),
    )
    foreground = image.crop(crop_box)
    crop = Image.composite(
        foreground, Image.new("RGB", foreground.size, (128, 128, 128)),
        mask.crop(crop_box).convert("L"),
    )
    crop.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
    return crop


def analyze_with_vision(
    base_result, raw: bytes, model_id: str = DEFAULT_MODEL,
    generator: Callable[[Image.Image, str, str], str] = generate_local,
):
    """Replace only diagnoses; retain existing leaf masks and severity estimates."""
    if base_result.status != "screened" or not base_result.leaves:
        raise ValueError("No detected leaves are available for vision diagnosis.")
    with Image.open(BytesIO(raw)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    image = image.resize(base_result.analysis_size, Image.Resampling.LANCZOS)
    prompt = build_prompt()
    result = deepcopy(base_result)
    for finding in result.leaves:
        crop = _leaf_crop(
            image, result.leaf_masks[finding.leaf_identifier], finding.bounding_box)
        response = parse_finding(generator(crop, prompt, model_id))
        candidate = "" if response.diagnosis == "Unknown / Other" else response.diagnosis
        finding.auto_candidate = candidate
        finding.primary_diagnosis = (
            f"{candidate} (tentative vision-model diagnosis)" if candidate else
            "Insufficient evidence for a disease-specific diagnosis"
        )
        finding.secondary_notes = (
            "Other visual possibilities: " + ", ".join(response.secondary_candidates)
            if response.secondary_candidates else
            "No supported secondary disease was identified."
        ) + " Expert review required."
        cues = "; ".join(response.observed_cues) or "No specific visual cues reported"
        finding.diagnostic_justification = (
            f"Vision model {model_id} reported: {cues}. {response.explanation} "
            "The severity percentage comes from the separate rule-based color mask."
        )
        finding.sources_used = (
            "; ".join(sources_for(candidate)) if candidate and candidate != "Healthy"
            else "docs/diagnostic_protocol.txt, Part 1"
        ) + "; knowledge/WildRice_diseases_CCM_v2_compressed.pdf"
        finding.quality_flags.append("unvalidated_vision_model_differential")
    return result
