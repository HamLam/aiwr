"""Durable, append-only per-leaf reviews and a curated training manifest."""

import argparse
import json
import os
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageOps

from .knowledge import CUES, RULES


DEFAULT_STORE = Path(__file__).resolve().parents[1] / "reviews"
CONFIRMATION_SOURCES = {
    "expert_visual": "Expert visual review",
    "laboratory_confirmed": "Laboratory confirmed",
    "needs_adjudication": "Needs adjudication",
}
LABELS = {rule.name for rule in RULES} | {"Healthy", "Unknown / Other"}
SOURCE_SUFFIXES = {"TIFF": ".tif", "JPEG": ".jpg", "PNG": ".png"}


def _atomic_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _png_bytes(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def save_review(
    result, finding, raw: bytes, *, reviewer_id: str, label: str,
    cues, tissue: str, confirmation_source: str, diagnostic_method: str,
    notes: str = "", specimen_id: str = "", store: Path = DEFAULT_STORE,
):
    """Save image assets first and publish a complete review event last."""
    reviewer_id = reviewer_id.strip()
    if not reviewer_id:
        raise ValueError("Enter a reviewer ID before saving.")
    if label not in LABELS:
        raise ValueError("Choose a reviewer-entered label before saving.")
    if confirmation_source not in CONFIRMATION_SOURCES:
        raise ValueError("Choose a confirmation status before saving.")
    if tissue not in {"leaf", "stem", "panicle"}:
        raise ValueError("Invalid tissue type.")
    if not set(cues).issubset(CUES):
        raise ValueError("Invalid reviewer-observed cue.")
    if finding not in result.leaves or finding.leaf_identifier not in result.leaf_masks:
        raise ValueError("Finding is not part of this image analysis.")
    digest = sha256(raw).hexdigest()
    if digest != result.image_sha256:
        raise ValueError("Source image does not match the analysis.")

    with Image.open(BytesIO(raw)) as opened:
        suffix = SOURCE_SUFFIXES.get(opened.format)
        if suffix is None:
            raise ValueError("Unsupported source image format.")
        image = ImageOps.exif_transpose(opened).convert("RGB")
    if image.size != result.analysis_size:
        image = image.resize(result.analysis_size, Image.Resampling.LANCZOS)
    mask = result.leaf_masks[finding.leaf_identifier].convert("L")
    if mask.size != image.size:
        raise ValueError("Leaf mask size does not match the analysis.")
    x, y, width, height = finding.bounding_box
    if width <= 0 or height <= 0:
        raise ValueError("Invalid leaf bounding box.")
    pad = max(8, round(min(width, height) * 0.05))
    box = (max(0, x - pad), max(0, y - pad),
           min(image.width, x + width + pad), min(image.height, y + height + pad))
    crop = Image.composite(image.crop(box), Image.new("RGB", image.crop(box).size, (128, 128, 128)),
                           mask.crop(box))
    review_id = uuid4().hex
    store = Path(store)
    source_path = store / "sources" / f"{digest}{suffix}"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_temp = source_path.with_name(f".{digest}.{uuid4().hex}.tmp")
    try:
        source_temp.write_bytes(raw)
        try:
            os.link(source_temp, source_path)
        except FileExistsError:
            pass
    finally:
        source_temp.unlink(missing_ok=True)
    if source_path.exists():
        if sha256(source_path.read_bytes()).hexdigest() != digest:
            raise ValueError("Stored source image checksum mismatch.")
    crop_path = store / "crops" / f"{review_id}.png"
    mask_path = store / "masks" / f"{review_id}.png"
    _atomic_bytes(crop_path, _png_bytes(crop))
    _atomic_bytes(mask_path, _png_bytes(mask.crop(box)))
    event = {
        "schema_version": 1,
        "review_id": review_id,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "reviewer_id": reviewer_id,
        "specimen_id": specimen_id.strip(),
        "source_filename": result.filename,
        "source_sha256": digest,
        "source_image": str(source_path.relative_to(store)),
        "leaf_identifier": finding.leaf_identifier,
        "bounding_box_analysis": list(finding.bounding_box),
        "analysis_size": list(result.analysis_size),
        "crop_image": str(crop_path.relative_to(store)),
        "leaf_mask": str(mask_path.relative_to(store)),
        "tissue": tissue,
        "reviewer_label": label,
        "reviewer_observed_cues": list(cues),
        "confirmation_source": confirmation_source,
        "reviewer_notes": notes.strip(),
        "diagnostic_method": diagnostic_method,
        "initial_primary_diagnosis": finding.primary_diagnosis,
        "auto_candidate": finding.auto_candidate,
        "automatic_severity_percent": finding.disease_severity_percent,
        "automatic_severity_class": finding.severity_classification,
        "quality_flags": finding.quality_flags,
    }
    _atomic_bytes(store / "events" / f"{review_id}.json",
                  json.dumps(event, indent=2, ensure_ascii=False).encode("utf-8"))
    return event


def export_manifest(store: Path = DEFAULT_STORE, output: Path = None) -> int:
    """Export latest eligible leaf reviews; retain all prior events on disk."""
    store = Path(store)
    output = Path(output) if output else store / "dataset.jsonl"
    latest = {}
    for path in (store / "events").glob("*.json"):
        event = json.loads(path.read_text(encoding="utf-8"))
        key = (event["source_sha256"], event["leaf_identifier"])
        if key not in latest or (event["saved_at_utc"], event["review_id"]) > (
            latest[key]["saved_at_utc"], latest[key]["review_id"]
        ):
            latest[key] = event
    records = []
    for event in latest.values():
        if (event["confirmation_source"] not in {"expert_visual", "laboratory_confirmed"}
                or event["tissue"] != "leaf" or event["reviewer_label"] == "Unknown / Other"):
            continue
        crop = store / event["crop_image"]
        if not crop.is_file():
            raise FileNotFoundError(f"Missing crop for review {event['review_id']}: {crop}")
        records.append({
            "image": os.path.relpath(crop, output.parent),
            "label": event["reviewer_label"],
            "review_id": event["review_id"],
            "source_sha256": event["source_sha256"],
            "leaf_identifier": event["leaf_identifier"],
            "specimen_id": event["specimen_id"],
            "reviewer_observed_cues": event["reviewer_observed_cues"],
            "confirmation_source": event["confirmation_source"],
        })
    records.sort(key=lambda item: (item["source_sha256"], item["leaf_identifier"]))
    _atomic_bytes(output, "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records).encode("utf-8"))
    return len(records)


def main():
    parser = argparse.ArgumentParser(description="Export latest confirmed leaf reviews to JSONL")
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    count = export_manifest(args.store, args.output)
    print(f"Exported {count} reviewed leaves to {args.output or args.store / 'dataset.jsonl'}")


if __name__ == "__main__":
    main()
