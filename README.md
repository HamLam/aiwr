# Wild Rice Leaf Disease Diagnosis

This repository contains a Streamlit app and command-line prototype for screening photographs of wild rice leaves. It identifies separate leaf regions, proposes a **tentative visual diagnosis** for each leaf, estimates visible color-change area, and presents the findings for expert review. The current prototype uses transparent image-processing rules; it has **no trained disease classifier** and has not been validated against expert-confirmed diagnoses.

The longer-term plan is a GPU-capable pipeline for larger image collections. See [the full pipeline plan](docs/plan.md) and [the diagnostic protocol](docs/diagnostic_protocol.txt). The planned models and throughput figures are proposals, not measured performance of this app.

## Repository layout

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit interface for image selection, findings, review, references, and downloads |
| `aiwr/diagnostic.py` | Leaf detection, provisional symptom masks, and severity calculation |
| `aiwr/autodiagnosis.py` | Automatic visual diagnosis rules |
| `aiwr/knowledge.py` | Reviewer cue rules and guide references |
| `aiwr/cli.py`, `aiwr/evaluate.py` | Batch exports and evaluation against confirmed labels |
| `docs/` | Full pipeline plan and diagnostic protocol |
| `examples/labels_template.csv` | Template for independently confirmed per-leaf labels |
| `reports/sample_runs.md` | Prototype results and validation limits |
| `tests/` | Automated checks |
| `run_app.sh`, `requirements.txt` | App launcher and Python dependencies |
| `knowledge/` | Reference PDFs used to document the diagnostic criteria |
| `samples_images/`, `outputs/` | Local sample TIFFs and generated results; excluded from Git |

The app accepts uploaded TIFF, JPEG, and PNG images without the local sample folder. The reference guides are included in `knowledge/` for the PDF viewer and source citations. The large sample TIFF collection remains local.

## Current prototype workflow

1. **Load and prepare an image.** The app reads an uploaded image or local sample, corrects its orientation, converts it to RGB, and scales it to the selected analysis size. It records the source image's SHA-256 checksum.
2. **Find individual leaves.** It separates saturated plant-colored regions from the controlled gray background and keeps tall, narrow shapes as leaf candidates. Each candidate receives a separate leaf ID and mask. This shape check cannot verify wild rice species.
3. **Mark visible changes.** Within each leaf, color thresholds mark candidate brown, yellowed, and unusually dark pixels. Small isolated groups and pixels near the leaf edge are removed. The red overlay shows these provisional areas; shadows, aging, and damage may also be marked.
4. **Make an initial diagnosis.** The app counts compact dark spots, elongated spots, larger oval blotches, and long streaks. The first matching rule yields a tentative diagnosis in the **Primary Diagnosis** table before reviewer input. Weak evidence yields “Insufficient evidence for a disease-specific diagnosis.”
5. **Estimate severity.** The percentage of marked pixels within each leaf mask determines the visible color-change area and its severity category. This calculation is separate from the disease choice.
6. **Review and export.** A reviewer may select observed cues from the guides and revise the visual differential, or enter a label. The app retains the original automatic candidate separately and exports per-leaf CSV, JSON, and an overlay. The CLI also writes leaf and symptom masks.

### How the initial diagnosis is chosen

The app uses handwritten rules, **not a model trained to recognize pathogens**. It groups dark pixels into connected areas and measures each group's size and width-to-height ratio. The rules run in this order:

| Tentative result | Image pattern required by the automatic rule |
| --- | --- |
| Healthy | No marked color change and almost no dark spots |
| Bacterial Leaf Streak | At least two long, narrow dark streaks and fewer than five compact spots |
| Fungal Brown Spot | At least two elongated dark areas among at least three compact spots |
| Spot Blotch | At least two larger oval dark areas among at least three compact spots |
| Leaf Spot | At least five compact dark spots |
| No disease-specific choice | None of the above has enough evidence |

For example, several elongated areas among many compact spots favor a tentative *Fungal Brown Spot* result. Because the rules are checked in order, a leaf with several different patterns receives the first applicable result. The table includes a justification, possible alternatives, and references to the diagnostic guides.

The automatic rules do not reliably recognize finer features such as yellow halos, water-soaked tissue, exudate, or fungal structures. A dark streak alone does not establish a bacterial cause. The **reviewer-observed cues** menu supplies guide-based visual differentials for features a person can identify; the **reviewer-entered label** menu can override the displayed diagnosis. Neither action by itself establishes laboratory confirmation.

### How severity is assigned

For **each leaf**, the app calculates:

**Visible color-change area (%) = 100 × marked pixels inside the leaf mask ÷ all visible pixels inside the leaf mask.**

It applies the categories in [Part 2 of the diagnostic protocol](docs/diagnostic_protocol.txt):

| Measured area | Severity category |
| --- | --- |
| 0% | Healthy |
| More than 0% to 10% | Low |
| More than 10% to 30% | Moderate |
| More than 30% to 60% | High |
| More than 60% | Severe |

For example, 150 marked pixels among 1,000 visible leaf pixels give **15% — Moderate**. One [sample run](reports/sample_runs.md) gave a leaf a tentative *Leaf Spot* diagnosis and **15.39% — Moderate** visible color change.

**Diagnosis and severity are separate calculations.** The diagnosis uses spot and streak patterns; severity uses the area of all marked color changes. Changing the diagnosis during review does not recalculate the percentage for that disease. The percentage is **not confirmed disease area**: senescence, shadows, injury, or other stresses can affect it. “Healthy” in the severity column means that the color mask measured 0%; it is not independent proof that the leaf is healthy.

## Run the app

On the MSI system, create a local environment once. The shared Python installation lacks a Streamlit WebSocket dependency, so the launcher uses this environment:

```bash
module load python3/3.12.4_anaconda2024.06-1_libmamba
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install 'websockets>=12,<16.1'
./run_app.sh
```

For a clean Python environment elsewhere, install `requirements.txt` and run `python -m streamlit run app.py`. On MSI, launch Streamlit from an interactive or Open OnDemand session with an accessible browser port. Upload an image if the local sample folder is absent.

For command-line screening and tests:

```bash
.venv/bin/python -m aiwr.cli samples_images/99_2024_Surv_bp_Waskish_August/RR8.TIF --output-dir outputs/rr8
.venv/bin/python -m pytest -q tests
```

The CLI writes per-leaf CSV and JSON results, overlays, leaf masks, and provisional symptom masks.

## Validation and next steps

The sample images do not have independent, expert-confirmed per-leaf diagnosis labels or reviewed lesion masks. **Diagnostic accuracy and severity error have not been measured; the 90% diagnostic accuracy goal remains unverified.** The [sample run report](reports/sample_runs.md) records prototype results and the checks completed so far.

To evaluate automatic diagnoses, copy `examples/labels_template.csv`, enter expert- or laboratory-confirmed labels, and run:

```bash
.venv/bin/python -m aiwr.evaluate outputs/rr8/leaves.csv confirmed_labels.csv
```

The evaluator scores the retained automatic candidate, not a reviewer override, and counts abstentions and missing predictions in overall accuracy. Severity accuracy requires expert-reviewed symptom masks or independent severity measurements. The next work is to assemble those labels and masks, test on an independent image set, and then compare trained classifiers and segmentation models as described in [the plan](docs/plan.md).
