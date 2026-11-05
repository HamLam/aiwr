"""Batch integration checks use a fake vision response, so no GPU is needed."""

import csv
from io import BytesIO
import json

from PIL import Image, ImageDraw

from aiwr.vision_batch import run_batch


def _leaf_image(path, color=(55, 145, 45)):
    image = Image.new("RGB", (600, 800), (125, 130, 135))
    ImageDraw.Draw(image).rectangle((280, 100, 315, 700), fill=color)
    output = BytesIO()
    image.save(output, format="PNG")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(output.getvalue())


def test_batch_writes_vision_results_and_resumes_only_matching_inputs(tmp_path):
    source = tmp_path / "images" / "leaf.png"
    output = tmp_path / "outputs"
    _leaf_image(source)
    calls = []

    def diagnose(result, raw, model_id):
        calls.append(model_id)
        result.leaves[0].auto_candidate = "Leaf Spot"
        result.leaves[0].primary_diagnosis = "Leaf Spot (tentative vision-model diagnosis)"
        return result

    assert run_batch([source], output, max_side=900, diagnose=diagnose) == 0
    assert calls == ["Qwen/Qwen2.5-VL-3B-Instruct"]
    record = json.loads((output / "results.json").read_text())[0]
    assert record["diagnostic_method"] == "vision_model"
    assert record["leaves"][0]["Primary Diagnosis"].startswith("Leaf Spot")
    with (output / "leaves.csv").open(newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["Vision Model"] == calls[0]
    assert row["Image SHA256"] == record["image_sha256"]

    assert run_batch([source], output, max_side=900, resume=True, diagnose=diagnose) == 0
    assert len(calls) == 1
    _leaf_image(source, color=(65, 150, 55))
    assert run_batch([source], output, max_side=900, resume=True, diagnose=diagnose) == 0
    assert len(calls) == 2
    assert run_batch([source], output, max_side=1000, resume=True, diagnose=diagnose) == 0
    assert len(calls) == 3


def test_batch_keeps_duplicate_names_separate_and_reports_failure(tmp_path):
    first = tmp_path / "a" / "leaf.png"
    second = tmp_path / "b" / "leaf.png"
    bad = tmp_path / "a" / "broken.png"
    _leaf_image(first)
    _leaf_image(second)
    bad.write_text("not an image")
    output = tmp_path / "outputs"

    def diagnose(result, raw, model_id):
        return result

    assert run_batch([tmp_path / "a", tmp_path / "b"], output, diagnose=diagnose) == 1
    records = json.loads((output / "results.json").read_text())
    assert len(records) == 3
    assert sum(record["status"] == "screened" for record in records) == 2
    assert sum(record["status"] == "error" for record in records) == 1
    assert len(list((output / "records").glob("leaf_*.json"))) == 2
    with (output / "leaves.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert rows[0]["Source Path"] != rows[1]["Source Path"]
