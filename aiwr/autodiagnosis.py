"""Tentative visual differentials from measured leaf color and shape.

This is a transparent rule-based baseline. It does not recognize pathogens
or verify Zizania species, and must not be interpreted as calibrated accuracy.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .knowledge import sources_for


@dataclass(frozen=True)
class AutoSuggestion:
    candidate: Optional[str]
    secondary: str
    rationale: str
    sources: Tuple[str, ...]


def suggest_diagnosis(hsv, leaf_mask, percent) -> AutoSuggestion:
    """Choose the best supported visual diagnosis; abstain if evidence is weak."""
    area = int(np.count_nonzero(leaf_mask))
    if not area:
        return AutoSuggestion(
            None, "No usable leaf area.",
            "Leaf mask contains no pixels.", ("docs/diagnostic_protocol.txt, Part 1",))
    interior = cv2.erode(
        leaf_mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    hue, sat, value = [hsv[:, :, channel] for channel in range(3)]
    baseline = float(np.median(value[leaf_mask]))
    dark = ((value < baseline * 0.80) & (hue < 35) &
            (sat > 45) & interior).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    min_area = max(8, int(area * 0.00003))
    round_spots = 0
    elongated_spots = 0
    oval_blotches = 0
    long_streaks = 0
    for component in range(1, count):
        x, y, w, h, pixels = stats[component]
        if pixels < min_area or pixels > area * 0.08:
            continue
        aspect = max(w, h) / max(1, min(w, h))
        if aspect >= 8 and h >= leaf_mask.shape[0] * 0.08:
            long_streaks += 1
        elif 2.7 <= aspect < 8 and pixels >= min_area * 2:
            elongated_spots += 1
        elif aspect < 2.7:
            round_spots += 1
            if pixels >= area * 0.001:
                oval_blotches += 1

    dark_percent = 100 * int(np.count_nonzero(dark)) / area
    summary = (
        f"Automated screen found {round_spots} compact dark spots, "
        f"{elongated_spots} elongated dark areas, and {long_streaks} long streaks. "
        "These shapes overlap among wild rice diseases."
    )
    if percent == 0 and dark_percent < 0.2 and round_spots < 3:
        return AutoSuggestion(
            "Healthy",
            "Subtle or invisible symptoms remain possible; expert review required.",
            summary + " No substantial visible lesion pattern was detected.",
            ("docs/diagnostic_protocol.txt, Part 2 (healthy category)",))
    if long_streaks >= 2 and round_spots < 5:
        name = "Bacterial Leaf Streak"
        return AutoSuggestion(
            name,
            "Fungal lesions or leaf-edge artifacts can mimic dark streaks.",
            summary + " Long, narrow dark streaks favor a bacterial leaf streak pattern.",
            sources_for(name))
    if elongated_spots >= 2 and round_spots >= 3:
        name = "Fungal Brown Spot"
        return AutoSuggestion(
            name,
            "Spot Blotch and other brown-spot conditions remain plausible.",
            summary + " Multiple elongated lesions among brown spots favor fungal brown spot.",
            sources_for(name))
    if oval_blotches >= 2 and round_spots >= 3:
        name = "Spot Blotch"
        return AutoSuggestion(
            name,
            "Fungal Brown Spot and Leaf Spot remain plausible without halo review.",
            summary + " Larger oval dark blotches favor spot blotch.",
            sources_for(name))
    if round_spots >= 5:
        name = "Leaf Spot"
        return AutoSuggestion(
            name,
            "Early Fungal Brown Spot or Spot Blotch can also appear as small dark spots.",
            summary + " Numerous compact dark spots favor a leaf-spot pattern.",
            sources_for(name))
    return AutoSuggestion(
        None,
        "Visible discoloration may reflect infection, senescence, injury, or artifacts.",
        summary + " The measured pattern is insufficient for a disease-specific choice.",
        ("docs/diagnostic_protocol.txt, Part 1 (diagnostic criteria)",))
