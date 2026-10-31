"""Conservative per-leaf image screening for controlled-background photographs.

Color masks are a reproducible prototype, not trained disease or species
recognition. All auto results require review; no confidence is fabricated.
"""

from dataclasses import dataclass, field
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .autodiagnosis import suggest_diagnosis


NO_LEAF = "No wild rice leaves detected in the provided image. Please provide a valid specimen image."
INVALID_PLANT = "ERROR: INVALID_IMAGE. This model only diagnoses wild rice leaves."
INVALID_NONPLANT = "ERROR: INVALID_IMAGE. Please upload a clear photo of a wild rice leaf."


@dataclass
class LeafFinding:
    image_filename: str
    leaf_identifier: str
    primary_diagnosis: str
    auto_candidate: str
    secondary_notes: str
    diagnostic_justification: str
    disease_severity_percent: float
    severity_classification: str
    sources_used: str
    leaf_area_pixels: int
    symptom_area_pixels: int
    review_required: bool = True
    diagnostic_confirmation_status: str = "unconfirmed"
    bounding_box: Tuple[int, int, int, int] = (0, 0, 0, 0)
    quality_flags: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "Image Filename": self.image_filename,
            "Leaf Identifier": self.leaf_identifier,
            "Primary Diagnosis": self.primary_diagnosis,
            "Auto Candidate": self.auto_candidate,
            "Secondary / Co-infection Notes": self.secondary_notes,
            "Diagnostic Justification": self.diagnostic_justification,
            "Disease Severity (%)": self.disease_severity_percent,
            "Severity Classification": self.severity_classification,
            "Sources Used": self.sources_used,
            "Leaf Area Pixels (analysis resolution)": self.leaf_area_pixels,
            "Symptom Area Pixels (analysis resolution)": self.symptom_area_pixels,
            "Review Required": self.review_required,
            "Diagnostic Confirmation Status": self.diagnostic_confirmation_status,
            "Bounding Box (analysis resolution)": self.bounding_box,
            "Quality Flags": "; ".join(self.quality_flags),
        }


@dataclass
class AnalysisResult:
    filename: str
    image_sha256: str
    original_size: Tuple[int, int]
    analysis_size: Tuple[int, int]
    status: str
    message: str
    leaves: List[LeafFinding]
    overlay: Optional[Image.Image] = None
    leaf_masks: Dict[str, Image.Image] = field(default_factory=dict)
    symptom_masks: Dict[str, Image.Image] = field(default_factory=dict)
    quality_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "image_sha256": self.image_sha256,
            "original_size": self.original_size,
            "analysis_size": self.analysis_size,
            "status": self.status,
            "message": self.message,
            "quality_flags": self.quality_flags,
            "leaves": [leaf.as_dict() for leaf in self.leaves],
        }


def severity_class(percent: float) -> str:
    """Apply the docs/diagnostic_protocol.txt Part 2 bins to a visible symptom percentage."""
    if percent <= 0:
        return "Healthy"
    if percent <= 10:
        return "Low"
    if percent <= 30:
        return "Moderate"
    if percent <= 60:
        return "High"
    return "Severe"


def _load(source: Union[str, Path, bytes], filename: Optional[str]):
    if isinstance(source, bytes):
        raw = source
        name = filename or "uploaded_image"
    else:
        path = Path(source)
        raw = path.read_bytes()
        name = filename or path.name
    with Image.open(BytesIO(raw)) as image:
        image = ImageOps.exif_transpose(image)
        original_size = image.size
        image = image.convert("RGB")
    return image, name, sha256(raw).hexdigest(), original_size


def _resize(image: Image.Image, max_side: int) -> Image.Image:
    if max(image.size) <= max_side:
        return image
    scale = max_side / max(image.size)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def _candidate_contours(rgb: np.ndarray):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # Controlled gray-background photos: foliage has much higher saturation.
    saturated = ((hsv[:, :, 0] < 90) & (hsv[:, :, 1] > 55) &
                 (hsv[:, :, 2] > 25)).astype(np.uint8) * 255
    saturated = cv2.morphologyEx(
        saturated, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    saturated = cv2.morphologyEx(
        saturated, cv2.MORPH_CLOSE, np.ones((15, 7), np.uint8))
    contours, _ = cv2.findContours(
        saturated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return hsv, saturated, contours


def _symptom_mask(rgb: np.ndarray, hsv: np.ndarray, leaf_mask: np.ndarray):
    r, g, b = [rgb[:, :, i].astype(np.float32) for i in range(3)]
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    leaf_vals = val[leaf_mask]
    baseline = float(np.median(leaf_vals)) if leaf_vals.size else 100.0
    # Lesion/chlorosis candidate pixels. These rules measure visible color
    # anomalies, not proven infection; shadows and senescence can confound them.
    brown = (hue < 30) & (sat > 60) & (r > b * 1.1)
    chlorotic = (hue >= 30) & (hue < 32) & (sat > 75) & (r > g * 0.92)
    dark = (val < baseline * 0.60) & (sat > 45) & (r > b * 0.85)
    interior = cv2.erode(leaf_mask.astype(np.uint8),
                          np.ones((5, 5), np.uint8)).astype(bool)
    candidate = ((brown | chlorotic | dark) & interior).astype(np.uint8)
    candidate = cv2.morphologyEx(
        candidate, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate, connectivity=8)
    cleaned = np.zeros(candidate.shape, dtype=np.uint8)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= 3:
            cleaned[labels == label] = 1
    return cleaned.astype(bool)


def analyze_image(
    source: Union[str, Path, bytes],
    filename: Optional[str] = None,
    max_side: int = 1600,
) -> AnalysisResult:
    """Screen one photograph and return per-leaf findings with masks."""
    if max_side < 400:
        raise ValueError("max_side must be at least 400")
    original, name, digest, original_size = _load(source, filename)
    image = _resize(original, max_side)
    rgb = np.asarray(image)
    height, width = rgb.shape[:2]
    hsv, saturated, contours = _candidate_contours(rgb)
    candidates = []
    broad_plant_like = False
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if area < width * height * 0.003 or h < height * 0.20:
            continue
        if h / max(w, 1) < 3.0:
            broad_plant_like = True
            continue
        candidates.append((x, contour))
    candidates.sort(key=lambda item: item[0])

    if not candidates:
        if broad_plant_like:
            message = INVALID_PLANT
        elif float(np.count_nonzero(saturated)) / saturated.size > 0.02:
            message = NO_LEAF
        else:
            message = INVALID_NONPLANT
        return AnalysisResult(
            name, digest, original_size, image.size, "invalid_image",
            message, [], image.copy(), quality_flags=["no_leaf_candidate"])

    flags = []
    # Strong edge color variation indicates that a controlled-background
    # threshold may be unreliable. Keep the result but flag it for review.
    edge = np.concatenate((rgb[0, ::8], rgb[-1, ::8],
                           rgb[::8, 0], rgb[::8, -1]), axis=0)
    if float(np.mean(np.std(edge.astype(np.float32), axis=0))) > 38:
        flags.append("complex_background_or_edge_objects")
    if len(candidates) > 20:
        flags.append("many_leaf_candidates")

    overlay = rgb.astype(np.float32).copy()
    leaf_masks = {}
    symptom_masks = {}
    findings = []
    tint_colors = [(60, 190, 255), (255, 185, 55), (165, 110, 255),
                   (80, 240, 150), (255, 105, 175)]
    for index, (_, contour) in enumerate(candidates, start=1):
        identifier = f"Leaf {index}"
        leaf = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(leaf, [contour], -1, 1, thickness=-1)
        leaf_bool = leaf.astype(bool)
        symptoms = _symptom_mask(rgb, hsv, leaf_bool)
        leaf_area = int(leaf_bool.sum())
        symptom_area = int(symptoms.sum())
        percent = round(100 * symptom_area / leaf_area, 2) if leaf_area else 0.0
        x, y, w, h = cv2.boundingRect(contour)
        local_flags = list(flags)
        if x == 0 or y == 0 or x + w >= width or y + h >= height:
            local_flags.append("leaf_cropped_by_frame")
        suggestion = suggest_diagnosis(hsv, leaf_bool, percent)
        diagnosis = (
            f"{suggestion.candidate} (tentative visual diagnosis)"
            if suggestion.candidate else
            "Insufficient evidence for a disease-specific diagnosis")
        notes = suggestion.secondary + " Expert review required."
        justification = (
            suggestion.rationale + " "
            f"Automated color screen marked {symptom_area} of {leaf_area} "
            f"visible leaf pixels ({percent:.2f}%) as discolored. "
            "The mask is a heuristic, not a reviewed lesion boundary."
        )
        findings.append(LeafFinding(
            image_filename=name,
            leaf_identifier=identifier,
            primary_diagnosis=diagnosis,
            auto_candidate=suggestion.candidate or "",
            secondary_notes=notes,
            diagnostic_justification=justification,
            disease_severity_percent=percent,
            severity_classification=severity_class(percent),
            sources_used="; ".join(suggestion.sources) + "; docs/diagnostic_protocol.txt, Part 2 "
                         "(severity bins); WildRice_diseases_CCM_v2_compressed.pdf, "
                         "p. 2 (reference severity photographs)",
            leaf_area_pixels=leaf_area,
            symptom_area_pixels=symptom_area,
            bounding_box=(x, y, w, h),
            quality_flags=local_flags + ["unvalidated_automatic_differential"],
        ))
        color = np.array(tint_colors[(index - 1) % len(tint_colors)])
        overlay[leaf_bool] = overlay[leaf_bool] * 0.82 + color * 0.18
        overlay[symptoms] = overlay[symptoms] * 0.35 + np.array([255, 50, 45]) * 0.65
        leaf_masks[identifier] = Image.fromarray(leaf * 255)
        symptom_masks[identifier] = Image.fromarray(symptoms.astype(np.uint8) * 255)
    overlay_image = Image.fromarray(np.uint8(np.clip(overlay, 0, 255)))
    draw = ImageDraw.Draw(overlay_image)
    for finding in findings:
        x, y, w, h = finding.bounding_box
        draw.text((min(x + 4, width - 80), max(y + 6, 0)),
                  finding.leaf_identifier, fill=(255, 255, 255),
                  stroke_width=2, stroke_fill=(0, 0, 0))
    return AnalysisResult(
        name, digest, original_size, image.size, "screened",
        "Automatic screening completed; every finding requires expert review.",
        findings, overlay_image, leaf_masks, symptom_masks, flags)
