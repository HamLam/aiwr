# Wild Rice Leaf Disease Diagnosis

This repository contains a Streamlit app for reviewing photographs of wild rice leaves. It offers two ways to produce an **initial, tentative visual diagnosis**: a rule-based method and a pretrained vision-language model. Both work one detected leaf at a time and show the diagnosis before reviewer input. Neither method has been validated against expert-confirmed diagnoses, so every finding requires review.

The longer-term GPU pipeline is described in [the full plan](docs/plan.md). The current app is a prototype, not the validated classifier and lesion segmentation system proposed there.

**MSI project location:** `/projects/standard/support/shared/lamx0031/Codex/aiwr`. All relative commands and paths below, including `.venv/bin/python` and `.venv-hcl/bin/python`, assume this directory is the current working directory:

```bash
cd /projects/standard/support/shared/lamx0031/Codex/aiwr
```

## Repository layout

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit interface, method selector, reviewer controls, PDF viewer, and downloads |
| `aiwr/diagnostic.py` | Shared leaf detection, visible color-change masks, and severity calculation |
| `aiwr/autodiagnosis.py` | Rule-based disease suggestions |
| `aiwr/vision.py` | Vision-model loading, knowledge context, and per-leaf diagnosis |
| `aiwr/knowledge.py` | Reviewer cues and source mappings |
| `aiwr/review_store.py` | Append-only review storage and curated dataset export |
| `aiwr/cli.py`, `aiwr/evaluate.py` | Rule-based batch export and evaluation against confirmed labels |
| `docs/` | [Pipeline plan](docs/plan.md) and [diagnostic protocol](docs/diagnostic_protocol.txt) |
| `knowledge/` | Reference PDFs cited by the app; includes the CCM disease guide |
| `examples/labels_template.csv` | Template for independently confirmed per-leaf labels |
| `reports/` | [Rule-based](reports/sample_runs.md) and [vision-model](reports/vision_sample_runs.md) smoke runs |
| `tests/` | Automated checks |
| `requirements*.txt`, `run_app.sh` | Dependencies and launcher |
| `samples_images/`, `outputs/`, `reviews/` | Local TIFF collection, generated results, and saved review data; excluded from Git |

The app accepts uploaded TIFF, JPEG, and PNG images without the local sample folder. Vision-model weights download into the ignored `.model_cache/` directory when first used.

## Shared app workflow

1. Select a sample or upload an image, then choose **Rule-based** or **Vision model (local GPU)**.
2. The app corrects image orientation, converts it to RGB, and scales it to the selected analysis size. It records a checksum of the original file.
3. Color and shape thresholds separate tall, narrow leaf-like regions from the controlled gray background. Each region gets a leaf ID and mask. This step does not verify wild rice species and can fail on field backgrounds or overlapping leaves.
4. The selected diagnostic method produces a tentative **Primary Diagnosis** for each detected leaf. The findings table appears before the reviewer controls.
5. The shared symptom mask and severity calculation produce a visible color-change percentage and category for each leaf, regardless of diagnostic method.
6. A reviewer selects observed cues, enters a label, chooses a confirmation status, and clicks **Save reviewed leaf**. The app immediately saves that leaf's review and image assets under `reviews/`. CSV, JSON, and overlay downloads remain available.

## Rule-based diagnosis

### How it chooses a diagnosis

The rule-based method uses handwritten image-processing rules; it is **not a trained disease classifier**. Within each leaf, it groups dark pixels into spots and measures each group's size and width-to-height ratio. It counts compact spots, elongated spots, larger oval blotches, and long narrow streaks. The first matching rule supplies the tentative result:

| Tentative result | Measured pattern |
| --- | --- |
| Healthy | No marked color change and almost no dark spots |
| Bacterial Leaf Streak | At least two long narrow dark streaks and fewer than five compact spots |
| Fungal Brown Spot | At least two elongated dark areas among at least three compact spots |
| Spot Blotch | At least two larger oval dark areas among at least three compact spots |
| Leaf Spot | At least five compact dark spots |
| No disease-specific choice | No rule has enough evidence |

For example, several elongated areas among many compact spots favor *Fungal Brown Spot (tentative visual diagnosis)*. The automatic rules do not reliably identify yellow halos, water-soaked tissue, exudate, or fungal structures. Different diseases and noninfectious damage can produce similar shapes.

### How rule-based severity is determined

The app marks candidate brown, yellowed, and unusually dark pixels inside each leaf mask, then removes tiny groups and pixels near the leaf edge. It calculates the percentage and category using the **shared severity method** below. The severity mask does not identify which disease caused a marked pixel.

### Run the rule-based app or CLI

On this MSI system, the rule-based launcher uses a local environment to supply Streamlit's WebSocket dependency:

```bash
cd /projects/standard/support/shared/lamx0031/Codex/aiwr
module load python3/3.12.4_anaconda2024.06-1_libmamba
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install 'websockets>=12,<16.1'
./run_app.sh
```

For batch screening and tests:

```bash
.venv/bin/python -m aiwr.cli samples_images/99_2024_Surv_bp_Waskish_August/RR8.TIF --output-dir outputs/rr8
.venv/bin/python -m pytest -q tests
```

The CLI is rule-based only. It writes per-leaf CSV and JSON records, overlays, leaf masks, and provisional symptom masks.

## Vision-model diagnosis

### How it chooses a diagnosis

Select **Vision model (local GPU)**, choose [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) or [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), then select **Run vision diagnosis**. The 3B model is the default. The 7B model uses a larger checkpoint and more GPU memory; its accuracy on these wild rice images has not been established.

The app crops each detected leaf and masks surrounding pixels to gray. It sends that crop to the pretrained vision-language model with two sources of text context: the leaf disease criteria in `docs/diagnostic_protocol.txt` and extractable text from `knowledge/WildRice_diseases_CCM_v2_compressed.pdf`. The PDF illustrations are available in the app viewer but are **not** sent to the model as reference images. Neither model was fine-tuned on this wild rice collection.

The model returns a proposed category, observed visual cues, possible alternatives, and a short explanation. It may select one of the listed leaf conditions, **Healthy**, or **Unknown / Other** when the image does not support a disease-specific choice. The app checks that the category is allowed, places the proposed result in **Primary Diagnosis**, and adds guide references from its source mapping. Those references document the category criteria; they do not confirm that the photographed leaf has that disease. If the model returns an invalid response or cannot run, the app shows an error instead of silently substituting a rule-based diagnosis.

The model's output is a **tentative visual differential**, not a pathogen identification or a calibrated probability. In [initial GPU runs](reports/vision_sample_runs.md), 3B abstained on RR1 and both RR8 leaves; 7B gave RR1 a tentative Spot Blotch label. Without expert-confirmed labels, these differences cannot show which model is more accurate.

### How vision-model severity is determined

The vision model **does not calculate the displayed severity percentage**. Before the model examines each crop, `aiwr/diagnostic.py` creates the same color-change mask used by the rule-based method. The app divides its marked pixels by detected leaf pixels and applies the shared severity scale below. Therefore, changing from 3B to 7B can change the diagnosis while leaving the severity number unchanged. Changing the diagnosis during review also leaves that number unchanged.

### Run the vision-model app on a GPU node

On MSI, the `DeepLearningHCL` environment provides GPU PyTorch. The project `.venv-hcl` overlay supplies compatible Transformers and Streamlit packages while keeping the conda environment unchanged:

```bash
cd /projects/standard/support/shared/lamx0031/Codex/aiwr
PYTHONNOUSERSITE=1 /users/9/lamx0031/.conda/envs/DeepLearningHCL/bin/python -m venv --system-site-packages .venv-hcl
PYTHONNOUSERSITE=1 .venv-hcl/bin/python -m pip install -r requirements-vision-hcl.txt
env -u LD_LIBRARY_PATH -u CUDA_HOME -u CUDA_PATH \
  PYTHONNOUSERSITE=1 USE_TF=0 \
  .venv-hcl/bin/python -m streamlit run app.py
```

The final launch command was verified on a GPU node. It prevents CUDA paths inherited from a shell or OnDemand session from overriding the libraries used by the conda PyTorch build. No separate CUDA or cuDNN module is required. A separate clean GPU environment can instead use `requirements-vision.txt`, which includes PyTorch. The first run of each model downloads its weights into `.model_cache/`. Both 3B and 7B loaded together on an A40 using about 22.6 GiB of allocated GPU memory; restart Streamlit to release loaded model memory. A Transformers warning about an ignored `temperature` generation flag is nonfatal.

## Shared severity method

For **each detected leaf**, the app calculates:

**Visible color-change area (%) = 100 × marked pixels inside the leaf mask ÷ all visible pixels inside the leaf mask.**

It applies the categories in [Part 2 of the diagnostic protocol](docs/diagnostic_protocol.txt):

| Measured area | Severity category |
| --- | --- |
| 0% | Healthy |
| More than 0% to 10% | Low |
| More than 10% to 30% | Moderate |
| More than 30% to 60% | High |
| More than 60% | Severe |

For example, 150 marked pixels among 1,000 visible leaf pixels give **15% — Moderate**. This is an area measurement, **not a diagnosis confidence score**. The percentage includes all color changes detected by the heuristic mask, regardless of the selected diagnosis. Shadows, senescence, injury, and other stresses may be counted; subtle lesions may be missed. “Healthy” in the severity column means the mask measured 0%, not that the leaf is independently confirmed healthy. Neither the leaf masks nor symptom masks have been validated against expert-reviewed boundaries.

## Review, exports, and evaluation

The reviewer-observed cue menu compares human-selected features with guide-based differentials. The reviewer-entered label menu can override the displayed diagnosis. Saving requires a reviewer ID, a label, and an explicit confirmation status: **Expert visual review**, **Laboratory confirmed**, or **Needs adjudication**. The confirmation status records the reviewer's stated evidence level; selecting it does not independently verify the diagnosis. The app also accepts an optional specimen or collection ID and reviewer notes.

Each click on **Save reviewed leaf** creates an immutable JSON review event in `reviews/events/`. The event links to the original uploaded image (stored once by SHA-256), a masked leaf crop, and a leaf mask. It also stores the initial automatic diagnosis, method, severity estimate, reviewer label and cues, reviewer ID, and timestamp. Later corrections create another event, preserving review history. `reviews/` is excluded from Git and should be backed up as research data.

Launch the app and save reviews first; no command is needed before reviewing. To create or refresh the training manifest after reviews have been saved, run this from the project directory:

```bash
.venv/bin/python -m aiwr.review_store
```

This writes `reviews/dataset.jsonl` with image paths and labels. It uses the latest review for each source image and leaf ID, and includes only leaf-tissue reviews marked expert visual or laboratory confirmed. Reviews marked **Needs adjudication**, **Unknown / Other**, and stem or panicle reviews remain in the audit store but are excluded from this leaf-classification manifest. Group train/validation/test splits by specimen or collection ID when available; leaves from the same image must stay in the same split. The manifest is a curated input for a later fine-tuning script, not a fine-tuned model or a measured accuracy result. CSV and JSON downloads retain **Auto Candidate** and **Diagnostic Method** for evaluating automatic suggestions separately from reviewer labels.

The sample images do not include independent expert-confirmed per-leaf diagnoses or reviewed lesion masks. **Diagnostic accuracy and severity error have not been measured; the 90% accuracy target remains unverified.** To evaluate automatic diagnoses, copy `examples/labels_template.csv`, enter expert- or laboratory-confirmed labels, and run:

```bash
.venv/bin/python -m aiwr.evaluate outputs/rr8/leaves.csv confirmed_labels.csv
```

The evaluator counts abstentions and missing predictions in overall accuracy and reports coverage separately. Severity accuracy requires expert-reviewed symptom masks or independent severity measurements. See [the pipeline plan](docs/plan.md) for the proposed validation and model-development work.
