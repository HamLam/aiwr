"""Run the existing per-leaf vision diagnosis across image files or folders."""

import argparse
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
from uuid import uuid4

from .diagnostic import analyze_image
from .vision import DEFAULT_MODEL, MODEL_OPTIONS, analyze_with_vision


IMAGE_SUFFIXES = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}
EMPTY_CSV_COLUMNS = [
    "Image Filename", "Leaf Identifier", "Primary Diagnosis", "Auto Candidate",
    "Image SHA256", "Source Path", "Diagnostic Method", "Vision Model",
]


def _image_paths(inputs, output_dir):
    output_dir = output_dir.resolve()
    seen = set()
    for value in inputs:
        root = Path(value)
        if root.is_dir():
            candidates = sorted(p for p in root.rglob("*") if p.is_file()
                                and p.suffix.lower() in IMAGE_SUFFIXES)
        else:
            candidates = [root]
        for path in candidates:
            resolved = path.resolve()
            if resolved in seen or resolved.is_relative_to(output_dir):
                continue
            seen.add(resolved)
            yield path


def _atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _record_path(path, output_dir):
    key = sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:16]
    return output_dir / "records" / f"{path.stem}_{key}.json"


def _previous_record(path, digest, model_id, max_side):
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (record.get("image_sha256") == digest
            and record.get("vision_model") == model_id
            and record.get("max_side") == max_side
            and record.get("status") in {"screened", "invalid_image"}):
        return record
    return None


def run_batch(inputs, output_dir, *, model_id=DEFAULT_MODEL, max_side=1600,
              resume=False, diagnose=analyze_with_vision):
    """Process images sequentially, saving an atomic record after each one."""
    if model_id not in MODEL_OPTIONS:
        raise ValueError(f"Unsupported vision model: {model_id}")
    if max_side < 400:
        raise ValueError("max_side must be at least 400")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    rows = []
    for path in _image_paths(inputs, output_dir):
        source_path = str(path.resolve())
        record_path = _record_path(path, output_dir)
        try:
            raw = path.read_bytes()
            digest = sha256(raw).hexdigest()
            prior = _previous_record(record_path, digest, model_id, max_side) if resume else None
            if prior is not None:
                record = prior
                action = "resumed"
            else:
                result = analyze_image(raw, filename=path.name, max_side=max_side)
                if result.status == "screened" and result.leaves:
                    result = diagnose(result, raw, model_id=model_id)
                record = result.to_dict()
                record.update({
                    "source_path": source_path,
                    "diagnostic_method": "vision_model",
                    "vision_model": model_id,
                    "max_side": max_side,
                })
                _atomic_write(record_path, json.dumps(record, indent=2, ensure_ascii=False))
                action = "processed"
        except Exception as exc:
            record = {
                "filename": path.name,
                "source_path": source_path,
                "status": "error",
                "error": str(exc),
                "leaves": [],
                "diagnostic_method": "vision_model",
                "vision_model": model_id,
                "max_side": max_side,
            }
            _atomic_write(record_path, json.dumps(record, indent=2, ensure_ascii=False))
            action = "failed"
        records.append(record)
        for leaf in record["leaves"]:
            row = dict(leaf)
            row.update({
                "Image SHA256": record["image_sha256"],
                "Source Path": source_path,
                "Diagnostic Method": "vision_model",
                "Vision Model": model_id,
            })
            rows.append(row)
        print(f"{path}: {record['status']}; {len(record['leaves'])} leaves ({action})", flush=True)

    _atomic_write(output_dir / "results.json", json.dumps(records, indent=2, ensure_ascii=False))
    csv_path = output_dir / "leaves.csv"
    columns = list(rows[0]) if rows else EMPTY_CSV_COLUMNS
    temporary = csv_path.with_name(f".{csv_path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, csv_path)
    finally:
        temporary.unlink(missing_ok=True)
    return 0 if records and all(record["status"] == "screened" for record in records) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Diagnose all detected leaves in image files or folders with a local vision model.")
    parser.add_argument("inputs", nargs="+", help="TIFF, JPEG, or PNG files or directories")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/vision_batch"))
    parser.add_argument("--model", choices=MODEL_OPTIONS, default=DEFAULT_MODEL)
    parser.add_argument("--max-side", type=int, default=1600)
    parser.add_argument("--resume", action="store_true",
                        help="Reuse successful records when source, model, and resolution match")
    args = parser.parse_args(argv)
    return run_batch(args.inputs, args.output_dir, model_id=args.model,
                     max_side=args.max_side, resume=args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
