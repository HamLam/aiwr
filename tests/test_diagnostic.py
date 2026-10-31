from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from aiwr.diagnostic import (
    INVALID_NONPLANT, INVALID_PLANT, NO_LEAF, analyze_image, severity_class,
)
from aiwr.knowledge import differential


SAMPLES = Path(__file__).resolve().parents[1] / "samples_images" / "99_2024_Surv_bp_Waskish_August"


@pytest.mark.parametrize("name,expected", [
    ("RR1.TIF", 1),
    ("RR2.TIF", 2),
    ("RR3.TIF", 1),
    ("RR8.TIF", 2),
    ("RR13.TIF", 2),
    ("RR20.TIF", 2),
    ("RR30.TIF", 2),
    ("RR40.TIF", 2),
    ("RR50.TIF", 2),
    ("RR60.TIF", 3),
])
def test_visible_leaf_count_on_sample_review(name, expected):
    if not SAMPLES.exists():
        pytest.skip("sample images not installed")
    result = analyze_image(SAMPLES / name, max_side=900)
    assert result.status == "screened"
    assert len(result.leaves) == expected
    assert all(leaf.review_required for leaf in result.leaves)
    assert all(0 <= leaf.disease_severity_percent <= 100 for leaf in result.leaves)


def _png(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_rejects_blank_nonplant_image():
    result = analyze_image(_png(Image.new("RGB", (600, 800), (120, 130, 135))), "blank.png")
    assert result.message == INVALID_NONPLANT
    assert not result.leaves


def test_rejects_broad_green_shape():
    pixels = np.full((800, 600, 3), (120, 130, 135), dtype=np.uint8)
    pixels[100:700, 180:430] = (70, 140, 45)
    result = analyze_image(_png(Image.fromarray(pixels)), "broad.png")
    assert result.message == INVALID_PLANT


@pytest.mark.parametrize("percent,grade", [
    (0, "Healthy"), (0.1, "Low"), (10, "Low"), (10.1, "Moderate"),
    (30, "Moderate"), (30.1, "High"), (60, "High"), (60.1, "Severe"),
])
def test_severity_bins(percent, grade):
    assert severity_class(percent) == grade


def test_distinctive_reference_cues():
    matches = differential(["eye_border"], tissue="leaf")
    assert matches[0].name == "Zonate Eyespot"
    assert "Fig. 3" in matches[0].sources[0]
    assert differential(["sclerotia"], tissue="leaf") == []
    assert differential(["sclerotia"], tissue="panicle")[0].name == "Ergot"


def test_evaluation_does_not_credit_abstention(tmp_path):
    from aiwr.evaluate import evaluate

    prediction = tmp_path / "predictions.csv"
    prediction.write_text(
        "Image Filename,Leaf Identifier,Primary Diagnosis,Disease Severity (%)\n"
        "sample.tif,Leaf 1,Unconfirmed visible symptoms,5.0\n")
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "image_filename,leaf_identifier,diagnosis,severity_percent,confirmation_source\n"
        "sample.tif,Leaf 1,Fungal Brown Spot,7.0,expert\n")
    report = evaluate(prediction, labels)
    assert report["accepted_diagnoses"] == 0
    assert report["accuracy_among_accepted_percent"] is None
    assert report["overall_accuracy_percent"] == 0
    assert report["meets_90_percent_accuracy_target"] is False
    assert report["severity_mean_absolute_error_points"] == 2.0


def test_plant_like_pixels_without_leaf_return_no_leaf_message():
    pixels = np.full((800, 600, 3), (120, 130, 135), dtype=np.uint8)
    pixels[100:220, 100:230] = (70, 140, 45)
    pixels[400:520, 330:460] = (70, 140, 45)
    result = analyze_image(_png(Image.fromarray(pixels)), "no_leaf.png")
    assert result.message == NO_LEAF


def test_automatic_candidate_precedes_review_input():
    if not SAMPLES.exists():
        pytest.skip("sample images not installed")
    result = analyze_image(SAMPLES / "RR1.TIF")
    finding = result.leaves[0]
    assert finding.auto_candidate
    assert finding.primary_diagnosis.startswith(finding.auto_candidate)
    assert "tentative visual diagnosis" in finding.primary_diagnosis
    assert ".pdf" in finding.sources_used
    assert finding.review_required


def test_evaluator_scores_auto_candidate_not_reviewer_override(tmp_path):
    from aiwr.evaluate import evaluate

    prediction = tmp_path / "predictions.csv"
    prediction.write_text(
        "Image Filename,Leaf Identifier,Primary Diagnosis,Auto Candidate,Disease Severity (%)\n"
        "sample.tif,Leaf 1,Fungal Brown Spot,Leaf Spot,5.0\n")
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "image_filename,leaf_identifier,diagnosis,severity_percent,confirmation_source\n"
        "sample.tif,Leaf 1,Fungal Brown Spot,7.0,expert\n")
    report = evaluate(prediction, labels)
    assert report["accepted_diagnoses"] == 1
    assert report["overall_accuracy_percent"] == 0


def test_evaluator_does_not_credit_reviewer_label_after_auto_abstention(tmp_path):
    from aiwr.evaluate import evaluate

    prediction = tmp_path / "predictions.csv"
    prediction.write_text(
        "Image Filename,Leaf Identifier,Primary Diagnosis,Auto Candidate,Disease Severity (%)\n"
        "sample.tif,Leaf 1,Fungal Brown Spot,,5.0\n")
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "image_filename,leaf_identifier,diagnosis,severity_percent,confirmation_source\n"
        "sample.tif,Leaf 1,Fungal Brown Spot,7.0,expert\n")
    report = evaluate(prediction, labels)
    assert report["accepted_diagnoses"] == 0
    assert report["overall_accuracy_percent"] == 0
