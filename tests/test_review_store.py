from hashlib import sha256
from io import BytesIO
import json

from PIL import Image
import pytest

from aiwr.diagnostic import AnalysisResult, LeafFinding
from aiwr.review_store import export_manifest, save_review


def _case():
    image = Image.new("RGB", (100, 120), (80, 150, 40))
    buffer = BytesIO()
    image.save(buffer, "PNG")
    raw = buffer.getvalue()
    finding = LeafFinding(
        "sample.png", "Leaf 1", "Leaf Spot (tentative)", "Leaf Spot", "",
        "Visual spots", 12.0, "Moderate", "guide", 4000, 480,
        bounding_box=(20, 10, 40, 100),
    )
    mask = Image.new("L", image.size, 0)
    mask.paste(255, (20, 10, 60, 110))
    result = AnalysisResult(
        "sample.png", sha256(raw).hexdigest(), image.size, image.size,
        "screened", "", [finding], leaf_masks={"Leaf 1": mask},
    )
    return result, finding, raw


def _save(store, **changes):
    result, finding, raw = _case()
    fields = dict(reviewer_id="reviewer-1", label="Leaf Spot",
                  cues=["small_black_spots"], tissue="leaf",
                  confirmation_source="expert_visual", diagnostic_method="rule_based",
                  specimen_id="plant-1", store=store)
    fields.update(changes)
    return save_review(result, finding, raw, **fields)


def test_save_persists_image_mask_and_auditable_event(tmp_path):
    event = _save(tmp_path)
    assert (tmp_path / event["source_image"]).is_file()
    assert (tmp_path / event["crop_image"]).is_file()
    assert (tmp_path / event["leaf_mask"]).is_file()
    assert json.loads((tmp_path / "events" / f"{event['review_id']}.json").read_text()) == event
    with Image.open(tmp_path / event["crop_image"]) as crop:
        assert crop.size[0] > 40
    assert export_manifest(tmp_path) == 1
    row = json.loads((tmp_path / "dataset.jsonl").read_text().strip())
    assert row["label"] == "Leaf Spot"
    assert row["specimen_id"] == "plant-1"


def test_latest_review_controls_dataset_eligibility(tmp_path):
    first = _save(tmp_path)
    second = _save(tmp_path, confirmation_source="needs_adjudication")
    assert first["review_id"] != second["review_id"]
    assert len(list((tmp_path / "events").glob("*.json"))) == 2
    assert export_manifest(tmp_path) == 0
    _save(tmp_path, label="Healthy")
    assert export_manifest(tmp_path) == 1
    assert json.loads((tmp_path / "dataset.jsonl").read_text())["label"] == "Healthy"


@pytest.mark.parametrize("field,value", [
    ("reviewer_id", ""), ("label", "No reviewer label"),
    ("confirmation_source", "Choose status"),
])
def test_incomplete_review_creates_no_event(tmp_path, field, value):
    with pytest.raises(ValueError):
        _save(tmp_path, **{field: value})
    assert not list((tmp_path / "events").glob("*.json"))
