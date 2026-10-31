"""Evaluate predictions against independently confirmed per-leaf labels."""

import argparse
import csv
import json
from pathlib import Path

from .knowledge import RULES


KNOWN = {rule.name for rule in RULES} | {"Healthy"}
CONFIRMED = {"expert", "laboratory"}


def evaluate(predictions_path, labels_path):
    with Path(predictions_path).open(newline="", encoding="utf-8") as stream:
        predictions = list(csv.DictReader(stream))
    with Path(labels_path).open(newline="", encoding="utf-8") as stream:
        labels = list(csv.DictReader(stream))
    required = {"image_filename", "leaf_identifier", "diagnosis", "confirmation_source"}
    if not labels or not required.issubset(labels[0]):
        raise ValueError("Labels CSV must contain: " + ", ".join(sorted(required)))
    pred_by_key = {
        (row["Image Filename"], row["Leaf Identifier"]): row
        for row in predictions
    }
    confirmed_rows = [
        row for row in labels
        if row["confirmation_source"].strip().lower() in CONFIRMED
    ]
    if not confirmed_rows:
        raise ValueError("No expert- or laboratory-confirmed labels found.")
    accepted = 0
    correct = 0
    matched = 0
    severity_errors = []
    missing = []
    for label in confirmed_rows:
        key = (label["image_filename"], label["leaf_identifier"])
        prediction = pred_by_key.get(key)
        if prediction is None:
            missing.append(key)
            continue
        matched += 1
        predicted = (prediction["Auto Candidate"].strip()
                     if "Auto Candidate" in prediction else
                     prediction["Primary Diagnosis"].strip())
        if predicted in KNOWN:
            accepted += 1
            correct += predicted == label["diagnosis"].strip()
        if label.get("severity_percent"):
            try:
                severity_errors.append(abs(
                    float(prediction["Disease Severity (%)"]) -
                    float(label["severity_percent"])))
            except (ValueError, KeyError):
                pass
    return {
        "confirmed_labels": len(confirmed_rows),
        "matched_predictions": matched,
        "missing_predictions": missing,
        "accepted_diagnoses": accepted,
        "diagnostic_coverage_percent": round(100 * accepted / matched, 2)
        if matched else None,
        "accuracy_among_accepted_percent": round(100 * correct / accepted, 2)
        if accepted else None,
        "overall_accuracy_percent": round(100 * correct / len(confirmed_rows), 2),
        "severity_mean_absolute_error_points": round(
            sum(severity_errors) / len(severity_errors), 2)
        if severity_errors else None,
        "severity_pairs": len(severity_errors),
        "meets_90_percent_accuracy_target": bool(
            correct / len(confirmed_rows) >= 0.90),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions_csv")
    parser.add_argument("confirmed_labels_csv")
    args = parser.parse_args(argv)
    report = evaluate(args.predictions_csv, args.confirmed_labels_csv)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
