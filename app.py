"""Streamlit review interface for source-backed wild rice image screening."""

from io import BytesIO
from pathlib import Path
import csv
import json

import fitz
import streamlit as st
from PIL import Image

from aiwr.diagnostic import analyze_image
from aiwr.knowledge import CUES, RULES, differential
from aiwr.vision import DEFAULT_MODEL, MODEL_OPTIONS, analyze_with_vision


ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples_images"
KNOWLEDGE = ROOT / "knowledge"
REPORT_COLUMNS = [
    "Image Filename", "Leaf Identifier", "Primary Diagnosis",
    "Secondary / Co-infection Notes", "Diagnostic Justification",
    "Disease Severity (%)", "Severity Classification", "Sources Used",
]


@st.cache_data(show_spinner=False)
def analyze_cached(raw, filename, max_side):
    return analyze_image(raw, filename=filename, max_side=max_side)


@st.cache_data(show_spinner=False)
def pdf_preview(path, page_number, width=950):
    document = fitz.open(path)
    try:
        page = document[page_number - 1]
        factor = width / page.rect.width
        return page.get_pixmap(matrix=fitz.Matrix(factor, factor)).tobytes("png")
    finally:
        document.close()


def csv_bytes(rows):
    output = BytesIO()
    text = __import__("io").TextIOWrapper(output, encoding="utf-8", newline="", write_through=True)
    writer = csv.DictWriter(text, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    text.flush()
    return output.getvalue()


def markdown_table(rows):
    def safe(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(REPORT_COLUMNS) + " |",
             "| " + " | ".join("---" for _ in REPORT_COLUMNS) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(safe(row.get(c, "")) for c in REPORT_COLUMNS) + " |")
    return "\n".join(lines)


st.set_page_config(page_title="Wild Rice Disease Review", layout="wide")
st.title("Wild Rice Disease Review")
st.caption("Choose rule-based or local vision-model diagnosis. Both are unvalidated visual differentials.")
st.info("Every automated finding requires expert review. A photograph cannot establish species or pathogen identity by itself. The percentage describes visible color change, not confirmed disease area.")

diagnostic_method = st.radio(
    "Diagnostic method", ["Rule-based", "Vision model (local GPU)"], horizontal=True)
model_id = (
    st.selectbox("Vision model", MODEL_OPTIONS,
                 format_func=lambda value: value.rsplit("/", 1)[-1])
    if diagnostic_method == "Vision model (local GPU)" else DEFAULT_MODEL
)
sample_files = sorted(SAMPLES.rglob("*.TIF")) if SAMPLES.exists() else []
source_mode = st.radio("Image source", ["Sample image", "Upload image"], horizontal=True)
raw = None
filename = None
if source_mode == "Sample image":
    if sample_files:
        choices = {str(path.relative_to(SAMPLES)): path for path in sample_files}
        chosen = st.selectbox("Sample image", list(choices))
        path = choices[chosen]
        raw = path.read_bytes()
        filename = path.name
    else:
        st.warning("No sample images found.")
else:
    uploaded = st.file_uploader("Upload TIFF, JPEG, or PNG", type=["tif", "tiff", "jpg", "jpeg", "png"])
    if uploaded is not None:
        raw = uploaded.getvalue()
        filename = uploaded.name

with st.expander("Analysis settings"):
    max_side = st.slider("Maximum analysis dimension (pixels)", 800, 2400, 1600, 200)

if raw is not None:
    try:
        result = analyze_cached(raw, filename, max_side)
    except Exception as exc:
        st.error(f"Could not decode or analyze the image: {exc}")
        st.stop()
    if result.status != "screened":
        st.error(result.message)
        st.stop()
    left, right = st.columns(2)
    with left:
        original_preview = Image.open(BytesIO(raw)).convert("RGB")
        original_preview.thumbnail((1600, 1600))
        st.image(original_preview, caption=f"Original: {filename}", width="stretch")
    with right:
        st.image(result.overlay, caption="Leaf regions and color-change candidates (red)",
                 width="stretch")
    st.caption(f"{len(result.leaves)} leaf region(s) detected · "
               f"analysis size {result.analysis_size[0]}×{result.analysis_size[1]} · "
               f"source SHA-256 {result.image_sha256[:12]}…")
    if result.quality_flags:
        st.warning("Quality flags: " + ", ".join(result.quality_flags))

    if diagnostic_method == "Vision model (local GPU)":
        st.caption(
            f"{model_id} examines each detected leaf using the diagnostic protocol "
            "and reference-guide text. Leaf detection and visible color-change severity "
            "still use the rule-based image masks. A CUDA GPU is required.")
        vision_key = f"vision:{result.image_sha256}:{max_side}:{model_id}"
        if st.button("Run vision diagnosis", type="primary"):
            try:
                with st.spinner("Examining each leaf with the vision model..."):
                    st.session_state[vision_key] = analyze_with_vision(
                        result, raw, model_id=model_id)
            except Exception as exc:
                st.error(f"Vision diagnosis could not run: {exc}")
                st.stop()
        if vision_key not in st.session_state:
            st.info("Select Run vision diagnosis to load the model and create findings.")
            st.stop()
        result = st.session_state[vision_key]

    st.subheader("Diagnostic findings")
    st.caption("Initial image-based diagnoses appear below before reviewer input. They are tentative visual differentials, not confirmed pathogen calls.")
    report_rows = [finding.as_dict() for finding in result.leaves]
    for row in report_rows:
        row["Diagnostic Method"] = (
            f"vision_model:{model_id}"
            if diagnostic_method == "Vision model (local GPU)" else "rule_based")
    findings_table = st.empty()
    findings_table.markdown(markdown_table(report_rows))

    st.subheader("Per-leaf review")
    st.write("Select observed cues for each leaf. Matches are visual differentials, never calibrated probabilities or confirmed causes.")
    for finding, row in zip(result.leaves, report_rows):
        with st.expander(f"{finding.leaf_identifier}: {finding.disease_severity_percent:.2f}% visible color change",
                         expanded=len(result.leaves) <= 3):
            st.write("Initial diagnosis: " + finding.primary_diagnosis)
            selected = st.multiselect(
                "Reviewer-observed cues", list(CUES), format_func=lambda key: CUES[key],
                key=f"cues_{filename}_{finding.leaf_identifier}")
            tissue = st.selectbox("Tissue examined", ["leaf", "stem", "panicle"],
                                  key=f"tissue_{filename}_{finding.leaf_identifier}")
            matches = differential(selected, tissue)
            if selected:
                row["Diagnostic Justification"] = (
                    "Reviewer-selected cues: " + "; ".join(CUES[key] for key in selected) + ". "
                    + row["Diagnostic Justification"])
                if matches:
                    row["Primary Diagnosis"] = (
                        matches[0].name + " (visual differential; unconfirmed)"
                        if len(matches) == 1 and selected != ["small_black_spots"]
                        else "Indeterminate visual differential")
                    row["Secondary / Co-infection Notes"] = (
                        "Other cue matches: " + ", ".join(rule.name for rule in matches[1:])
                        if len(matches) > 1 else
                        "The same visual pattern may have other causes; expert confirmation required.")
                    row["Sources Used"] = "; ".join(
                        dict.fromkeys(source for rule in matches for source in rule.sources))
                else:
                    row["Primary Diagnosis"] = "Indeterminate visual symptoms"
                    row["Secondary / Co-infection Notes"] = (
                        "Selected cues do not uniquely match the available guide criteria.")
            st.write(row["Primary Diagnosis"])
            st.caption(row["Sources Used"])
            reviewed = st.selectbox(
                "Reviewer-entered label (optional; stored in this export only)",
                ["No reviewer label"] + [rule.name for rule in RULES if rule.tissue == tissue]
                + ["Healthy", "Unknown / Other"],
                key=f"label_{filename}_{finding.leaf_identifier}")
            if reviewed != "No reviewer label":
                row["Primary Diagnosis"] = reviewed
                row["Diagnostic Confirmation Status"] = "reviewer_entered_unverified"
                row["Secondary / Co-infection Notes"] += (
                    " Reviewer-entered label; independent confirmation status not established.")
    findings_table.markdown(markdown_table(report_rows))
    st.download_button("Download per-leaf CSV", csv_bytes(report_rows),
                       file_name=f"{Path(filename).stem}_findings.csv", mime="text/csv")
    st.download_button("Download result JSON", json.dumps(
        {"diagnostic_method": diagnostic_method, "image": result.to_dict(),
         "reviewed_rows": report_rows},
        indent=2, default=str).encode("utf-8"),
        file_name=f"{Path(filename).stem}_findings.json", mime="application/json")
    overlay_bytes = BytesIO()
    result.overlay.save(overlay_bytes, format="PNG")
    st.download_button("Download overlay", overlay_bytes.getvalue(),
                       file_name=f"{Path(filename).stem}_overlay.png", mime="image/png")

st.divider()
st.subheader("Knowledge references")
st.write("Diagnostic cues come from docs/diagnostic_protocol.txt and the local wild rice disease guides. PDF page numbers below are file page numbers.")
pdfs = sorted(KNOWLEDGE.glob("*.pdf")) if KNOWLEDGE.exists() else []
if pdfs:
    selected_pdf = st.selectbox("Reference guide", pdfs, format_func=lambda path: path.name)
    with fitz.open(selected_pdf) as document:
        page_count = len(document)
    page_number = st.number_input("PDF page", min_value=1, max_value=page_count, value=1)
    st.image(pdf_preview(str(selected_pdf), page_number),
             caption=f"{selected_pdf.name}, PDF page {page_number}",
             width="stretch")
else:
    st.warning("No PDFs found in knowledge/.")
