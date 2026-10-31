"""Contract tests for the optional vision-model route, without model weights."""

import json
from pathlib import Path

import pytest

from aiwr.diagnostic import analyze_image
from aiwr.vision import (
    ALLOWED_LABELS, DEFAULT_MODEL, analyze_with_vision, build_prompt, parse_finding,
)


SAMPLE = (Path(__file__).resolve().parents[1] / "samples_images" /
          "99_2024_Surv_bp_Waskish_August" / "RR1.TIF")


def test_vision_prompt_uses_local_knowledge():
    prompt = build_prompt()
    assert "Spindle-shaped" in prompt or "spindle-shaped" in prompt
    assert "WildRice_diseases_CCM_v2_compressed.pdf" in prompt
    assert "Unknown / Other" in prompt
    assert "diagnosis" in prompt


def test_vision_response_rejects_unsupported_label():
    with pytest.raises(ValueError, match="unsupported diagnosis"):
        parse_finding(json.dumps({
            "diagnosis": "Certainly bacterial infection",
            "observed_cues": [], "secondary_candidates": [], "explanation": "dark leaf",
        }))


def test_vision_route_replaces_only_diagnosis_and_preserves_rule_result():
    if not SAMPLE.exists():
        pytest.skip("sample image not installed")
    raw = SAMPLE.read_bytes()
    base = analyze_image(raw, filename=SAMPLE.name, max_side=900)
    original = base.leaves[0]
    calls = []

    def fake_generator(image, prompt, model_id):
        calls.append((image.size, model_id))
        assert "Fungal Brown Spot" in prompt
        return json.dumps({
            "diagnosis": "Zonate Eyespot",
            "observed_cues": ["eye-shaped pale lesion with a dark border"],
            "secondary_candidates": ["Fungal Brown Spot"],
            "explanation": "Visual pattern is suggestive but not diagnostic.",
        })

    result = analyze_with_vision(base, raw, generator=fake_generator)
    finding = result.leaves[0]
    assert calls and calls[0][1] == DEFAULT_MODEL
    assert finding.auto_candidate in ALLOWED_LABELS
    assert finding.auto_candidate == "Zonate Eyespot"
    assert finding.primary_diagnosis.startswith("Zonate Eyespot")
    assert finding.disease_severity_percent == original.disease_severity_percent
    assert finding.symptom_area_pixels == original.symptom_area_pixels
    assert original.auto_candidate != finding.auto_candidate
    assert "eye-shaped pale lesion" in finding.diagnostic_justification
    assert "Fig. 3" in finding.sources_used
    assert finding.review_required
