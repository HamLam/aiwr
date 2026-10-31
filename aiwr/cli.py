"""Command-line screening and export for wild rice sample images."""

import argparse
import csv
import json
from pathlib import Path

from .diagnostic import analyze_image


def _image_paths(inputs):
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            yield from sorted(p for p in path.rglob("*")
                              if p.suffix.lower() in {".tif", ".tiff", ".jpg", ".jpeg", ".png"})
        else:
            yield path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Screen wild rice photographs per leaf.")
    parser.add_argument("inputs", nargs="+", help="Image files or directories")
    parser.add_argument("--output-dir", default="outputs", help="Directory for CSV, JSON, masks, overlays")
    parser.add_argument("--max-side", type=int, default=1600, help="Maximum analysis image dimension")
    args = parser.parse_args(argv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    records = []
    for path in _image_paths(args.inputs):
        try:
            result = analyze_image(path, max_side=args.max_side)
            record = result.to_dict()
            record["source_path"] = str(path.resolve())
            records.append(record)
            if result.overlay is not None:
                result.overlay.save(output / (path.stem + "_overlay.png"))
            for finding in result.leaves:
                stem = path.stem + "_" + finding.leaf_identifier.replace(" ", "_")
                result.leaf_masks[finding.leaf_identifier].save(output / (stem + "_leaf_mask.png"))
                result.symptom_masks[finding.leaf_identifier].save(
                    output / (stem + "_symptom_mask.png"))
                row = finding.as_dict()
                row["Image SHA256"] = result.image_sha256
                row["Source Path"] = str(path.resolve())
                rows.append(row)
            print(f"{path.name}: {result.status}; {len(result.leaves)} leaves; "
                  + ", ".join(
                      f"{leaf.leaf_identifier}: "
                      f"{leaf.auto_candidate or 'undetermined'} "
                      f"({leaf.disease_severity_percent:.2f}% visible change)"
                      for leaf in result.leaves))
        except Exception as exc:
            records.append({"filename": path.name, "source_path": str(path),
                            "status": "error", "error": str(exc), "leaves": []})
            print(f"{path.name}: ERROR: {exc}")
    with (output / "results.json").open("w", encoding="utf-8") as stream:
        json.dump(records, stream, indent=2, ensure_ascii=False)
    if rows:
        with (output / "leaves.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return 0 if all(record["status"] == "screened" for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
