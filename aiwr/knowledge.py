"""Diagnostic cues transcribed from docs/diagnostic_protocol.txt and the local knowledge PDFs.

Page numbers are PDF page numbers (one-based), not printed book pagination.
These are visual differentials for reviewer use, not validated model classes.
"""

from dataclasses import dataclass
from typing import FrozenSet, Tuple


PICTORIAL = "Pictorial_Guide_95-07-03.pdf"
CCM_V2 = "WildRice_diseases_CCM_v2_compressed.pdf"
CCM_SHORT = "wild_rice_diseases_CCM.pdf"

CUES = {
    "spindle_halo": "Spindle-shaped brown lesion with darker edge and yellow halo",
    "oval_no_halo": "Oval or irregular dark brown blotch, initially without halo",
    "eye_border": "Eye-shaped pale center with a sharp dark reddish-brown border",
    "small_black_spots": "Small round dark brown to black spots",
    "diffuse_edge_pycnidia": "Large diffuse edge blotch with pepper-like black dots",
    "linear_black_pustules": "Slightly raised, linear coal-black pustules",
    "johnson_necrosis": "Expanding irregular brown necrosis on Johnson germplasm",
    "water_soaked_ellipse": "Water-soaked spot becoming an elliptical brown lesion",
    "water_soaked_streak": "Long translucent water-soaked or necrotic linear streak",
    "bacterial_exudate": "White flakes or yellow droplets of exudate",
    "pink_spikelets": "Shrunken tan spikelets with pink/orange fungal growth",
    "sclerotia": "Hardened pink/purple/black masses replacing kernels",
    "basal_rot": "Dark lesions and decomposition at water-level stem or crown",
}

@dataclass(frozen=True)
class Rule:
    name: str
    required: FrozenSet[str]
    supporting: FrozenSet[str]
    explanation: str
    sources: Tuple[str, ...]
    tissue: str = "leaf"


RULES = (
    Rule("Fungal Brown Spot", frozenset({"spindle_halo"}), frozenset({"small_black_spots"}),
         "Spindle-shaped brown lesions with dark margins and yellow halos.",
         (f"{CCM_V2} pp. 2–4 (Fig. 1)", f"{PICTORIAL} pp. 5–6")),
    Rule("Spot Blotch", frozenset({"oval_no_halo"}), frozenset({"small_black_spots"}),
         "Oval or irregular dark blotches, initially without chlorotic halos.",
         (f"{CCM_V2} pp. 5–7 (Fig. 2)", f"{PICTORIAL} pp. 19–20")),
    Rule("Zonate Eyespot", frozenset({"eye_border"}), frozenset({"water_soaked_ellipse"}),
         "Pale eye-shaped centers with contrasting reddish-brown or dark borders.",
         (f"{CCM_V2} pp. 8–13 (Fig. 3)", f"{PICTORIAL} pp. 9–10")),
    Rule("Leaf Spot", frozenset({"small_black_spots"}), frozenset(),
         "Small round to oval dark brown or black spots; overlaps other diseases.",
         (f"{CCM_V2} p. 14 (Fig. 4)",)),
    Rule("Leaf Blotch", frozenset({"diffuse_edge_pycnidia"}), frozenset(),
         "Diffuse beige-gray edge blotch with pepper-like pycnidia.",
         (f"{CCM_V2} p. 15 (Fig. 5)", f"{PICTORIAL} pp. 11–12")),
    Rule("Smut", frozenset({"linear_black_pustules"}), frozenset(),
         "Raised, narrow, coal-black pustules embedded in tissue.",
         (f"{CCM_V2} p. 16 (Fig. 6)", f"{PICTORIAL} pp. 17–18")),
    Rule("Johnson Brown Lesions", frozenset({"johnson_necrosis"}), frozenset(),
         "Irregular expanding necrosis in Johnson germplasm; physiological cause.",
         (f"{CCM_V2} p. 17 (Fig. 7)", f"{PICTORIAL} pp. 27–28")),
    Rule("Bacterial Brown Spot", frozenset({"water_soaked_ellipse"}), frozenset({"bacterial_exudate"}),
         "Water-soaked spots developing into elliptical brown lesions.",
         (f"{CCM_V2} p. 18 (Fig. 8)", f"{PICTORIAL} pp. 21–22")),
    Rule("Bacterial Leaf Streak", frozenset({"water_soaked_streak"}), frozenset({"bacterial_exudate"}),
         "Long translucent water-soaked streaks becoming striped necrosis.",
         (f"{CCM_V2} p. 19 (Fig. 9)", f"{PICTORIAL} pp. 23–24")),
    Rule("Fusarium Head Blight", frozenset({"pink_spikelets"}), frozenset(),
         "Shrunken tan spikelets, sometimes with pink/orange mycelium.",
         (f"{CCM_V2} p. 20 (Fig. 10)", f"{PICTORIAL} pp. 15–16"), tissue="panicle"),
    Rule("Ergot", frozenset({"sclerotia"}), frozenset(),
         "Hard sclerotia replacing kernels.",
         (f"{CCM_V2} p. 21 (Fig. 11)", f"{PICTORIAL} pp. 7–8"), tissue="panicle"),
    Rule("Leaf Sheath and Stem Rot", frozenset({"basal_rot"}), frozenset(),
         "Dark water-line lesions and decomposing lower stem or crown.",
         (f"{CCM_V2} p. 22 (Fig. 12)", f"{PICTORIAL} pp. 13–14"), tissue="stem"),
)


def differential(selected_cues, tissue="leaf"):
    """Return evidence-matched rules, without treating matches as probabilities."""
    selected = set(selected_cues)
    matches = []
    for rule in RULES:
        if rule.tissue != tissue or not rule.required.issubset(selected):
            continue
        score = len(rule.required) * 2 + len(rule.supporting & selected)
        matches.append((score, rule))
    matches.sort(key=lambda item: (-item[0], item[1].name))
    return [rule for _, rule in matches]


def sources_for(name):
    for rule in RULES:
        if rule.name == name:
            return rule.sources
    return ("docs/diagnostic_protocol.txt, Part 1 (diagnostic criteria)",)
