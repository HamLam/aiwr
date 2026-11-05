# Wild Rice Disease Review and Diagnostic

This repository contains a Streamlit app and command-line tools for screening wild rice leaf photographs. The app shows a tentative diagnosis for each detected leaf before a reviewer enters observations or a label. It offers a custom rule-based method and a pretrained Qwen2.5 vision-language model. Neither method has measured diagnostic accuracy against an expert-labeled test set, so their outputs require expert review.

The current implementation is a prototype. [The pipeline plan](docs/plan.md) describes the proposed training, segmentation, validation, and larger-scale GPU workflow.

## Project location and contents

On the MSI system, this project is at `/projects/standard/support/shared/lamx0031/Codex/aiwr`. The commands below assume you start there:

```bash
cd /projects/standard/support/shared/lamx0031/Codex/aiwr
```

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit interface for diagnosis, per-leaf review, reference PDFs, and downloads |
| `aiwr/diagnostic.py` | Leaf detection, image masks, and visible color-change severity |
| `aiwr/autodiagnosis.py` | Handwritten rule-based disease suggestions |
| `aiwr/vision.py` | Qwen model loading, knowledge context, and per-leaf vision diagnosis |
| `aiwr/cli.py` | Rule-based processing of image files or folders |
| `aiwr/vision_batch.py` | Resumable vision diagnosis of image files or folders |
| `aiwr/review_store.py` | Saved reviewer decisions and curated dataset manifest |
| `aiwr/evaluate.py` | Comparison with independently confirmed labels |
| `docs/` | [Pipeline plan](docs/plan.md) and [diagnostic protocol](docs/diagnostic_protocol.txt) |
| `knowledge/` | Reference PDFs, including the CCM wild rice disease guide |
| `examples/labels_template.csv` | Template for confirmed per-leaf labels |
| `reports/` | [Rule-based](reports/sample_runs.md) and [vision-model](reports/vision_sample_runs.md) sample runs |
| `tests/` | Automated checks |
| `samples_images/`, `outputs/`, `reviews/`, `.model_cache/` | Local images, results, reviewer data, and downloaded model weights; excluded from Git |

Uploaded TIFF, JPEG, and PNG images work without the local sample collection. The reference PDFs and protocol ship with the repository; the Qwen model weights download into `.model_cache/` on first use.

## Run the app

### Rule-based setup

On MSI, the rule-based launcher uses the local Python environment and supplies Streamlit's WebSocket dependency:

```bash
module load python3/3.12.4_anaconda2024.06-1_libmamba
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install 'websockets>=12,<16.1'
./run_app.sh
```

The rule-based method uses image processing rules and needs no model training or GPU. It still needs validation against expert-labeled images. For a rule-based folder run, use:

```bash
.venv/bin/python -m aiwr.cli samples_images/99_2024_Surv_bp_Waskish_August \
  --output-dir outputs/rule_batch
```

This writes `results.json`, `leaves.csv`, overlays, leaf masks, and provisional symptom masks.

### Vision-model setup on an MSI GPU node

The `DeepLearningHCL` environment provides GPU PyTorch. A local `.venv-hcl` overlay supplies compatible Transformers and Streamlit packages without changing the conda environment:

```bash
PYTHONNOUSERSITE=1 /users/9/lamx0031/.conda/envs/DeepLearningHCL/bin/python \
  -m venv --system-site-packages .venv-hcl
PYTHONNOUSERSITE=1 .venv-hcl/bin/python -m pip install -r requirements-vision-hcl.txt
env -u LD_LIBRARY_PATH -u CUDA_HOME -u CUDA_PATH \
  PYTHONNOUSERSITE=1 USE_TF=0 \
  .venv-hcl/bin/python -m streamlit run app.py
```

The launch command has worked on an MSI GPU node. Clearing inherited CUDA library paths prevents them from overriding the libraries used by this PyTorch environment; no separate CUDA or cuDNN module was needed there. A clean GPU environment can instead use `requirements-vision.txt`, which includes PyTorch. Model weights are cached in `.model_cache/`. Both 3B and 7B checkpoints loaded together on an A40 using about 22.6 GiB of allocated GPU memory; restart Streamlit to release cached models. A warning about an ignored `temperature` generation flag is nonfatal.

### Single-image workflow

1. Select **Rule-based** or **Vision model (local GPU)**, then choose a sample image or upload TIFF, JPEG, or PNG.
2. The app corrects image orientation, converts it to RGB, and shrinks the whole image proportionally if its longest side exceeds **Maximum analysis dimension**. The slider runs from 800 to 2400 pixels, with 1600 as the default. Smaller images are not enlarged. More pixels preserve small visual details but take more memory and time.
3. Color and shape rules detect tall, narrow leaf-like regions against the controlled background. Each region receives a leaf ID and mask. Detection may fail on field backgrounds or overlapping leaves and does not verify the plant species.
4. The selected method gives each leaf a tentative **Primary Diagnosis** before reviewer input. The shared image masks give a visible color-change percentage and severity class.
5. A reviewer can select observed cues, enter a label, choose a confirmation status, and click **Save reviewed leaf**. CSV, JSON, and overlay downloads remain available.

## How diagnosis works

### Rule-based method

This method has no trained model weights. Inside each detected leaf, it groups dark pixels and measures spot size and shape. The first matching handwritten rule supplies the tentative diagnosis:

| Tentative result | Pattern measured by the rules |
| --- | --- |
| Healthy | No marked color change and almost no dark spots |
| Bacterial Leaf Streak | At least two long narrow dark streaks and fewer than five compact spots |
| Fungal Brown Spot | At least two elongated dark areas among at least three compact spots |
| Spot Blotch | At least two larger oval dark areas among at least three compact spots |
| Leaf Spot | At least five compact dark spots |
| No disease-specific choice | No rule has enough evidence |

These measurements do not reliably identify yellow halos, water-soaked tissue, exudate, or fungal structures. Similar shapes can arise from different diseases or noninfectious damage. The method can be tuned, but its present accuracy is unknown.

### Vision-model method

The app offers [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) by default and [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) as an option. Neither checkpoint has been fine-tuned on this wild rice collection.

After leaf detection, the app crops each leaf from the resized analysis image and masks surrounding pixels to gray. It sends the crop to Qwen with text from `docs/diagnostic_protocol.txt` and extractable text from `knowledge/WildRice_diseases_CCM_v2_compressed.pdf`. The PDF illustrations are visible in the reference viewer but are not supplied to the model as reference images.

The model returns a proposed label, observed cues, possible alternatives, and an explanation. The app checks that the label is allowed and displays it in **Primary Diagnosis**. It can also report **Healthy** or abstain when evidence is insufficient. Source references document category criteria; they do not confirm the cause of symptoms in a photographed leaf. If the model fails or returns an invalid response, the app reports the error rather than substituting a rule-based diagnosis. [Initial GPU runs](reports/vision_sample_runs.md) show differing 3B and 7B outputs, but do not establish which is more accurate.

### Shared severity scale

Both methods use the same color-change mask made by `aiwr/diagnostic.py`. The mask marks candidate brown, yellowed, and unusually dark pixels inside the detected leaf, then removes tiny groups and edge pixels. The vision model does not calculate the displayed severity percentage.

**Visible color-change area (%) = 100 × marked pixels inside the leaf mask ÷ all visible pixels inside the leaf mask.**

| Measured area | Severity class |
| --- | --- |
| 0% | Healthy |
| More than 0% to 10% | Low |
| More than 10% to 30% | Moderate |
| More than 30% to 60% | High |
| More than 60% | Severe |

For example, 150 marked pixels among 1,000 leaf pixels give **15% — Moderate**. This is an area estimate, not diagnosis confidence or confirmed disease coverage. Shadows, age, injury, and other stresses can affect it; subtle lesions can be missed. “Healthy” in this column means the mask measured 0%, not that a reviewer confirmed a healthy leaf. See [Part 2 of the diagnostic protocol](docs/diagnostic_protocol.txt).

## Vision diagnosis for a folder

On a GPU node, process TIFF, JPEG, and PNG files recursively with the same vision workflow as the app:

```bash
env -u LD_LIBRARY_PATH -u CUDA_HOME -u CUDA_PATH \
  PYTHONNOUSERSITE=1 USE_TF=0 \
  .venv-hcl/bin/python -m aiwr.vision_batch \
  samples_images/99_2024_Surv_bp_Waskish_August \
  --output-dir outputs/vision_batch --resume
```

The default model is 3B. Add `--model Qwen/Qwen2.5-VL-7B-Instruct` for 7B or `--max-side 2400` for higher analysis resolution. Individual files can also be given. The selected model is loaded once per process; detected leaves are diagnosed sequentially. This command processes a folder, but does not yet combine leaf crops into simultaneous GPU inference batches.

Each image gets a JSON record under `outputs/vision_batch/records/`. The folder also receives `results.json` and per-leaf `leaves.csv`. A failed or invalid image is recorded, and the command exits with a nonzero status. `--resume` reuses a successful record only when its source checksum, model, and analysis dimension still match; error records are retried. These automatic results are unreviewed and do not enter `reviews/` automatically.

The batch command was exercised with a real RR1 TIFF on an L40S GPU and resumed its saved result on a second run. The complete sample collection has not been evaluated for diagnostic accuracy.

## Save reviews and build a dataset

In the Streamlit app, **Save reviewed leaf** writes a record immediately after a reviewer enters an ID and label and selects **Expert visual review**, **Laboratory confirmed**, or **Needs adjudication**. The status records the reviewer's stated evidence level; choosing it does not independently verify a diagnosis. A specimen or collection ID and notes can also be entered.

Each save creates an immutable JSON event in `reviews/events/`, plus a masked leaf crop and leaf mask. The original source image is stored once by SHA-256 checksum. Events retain the automatic diagnosis, method, severity estimate, reviewer observations, reviewer label, ID, and timestamp. Corrections create new events, preserving the history. `reviews/` is excluded from Git and should be backed up as research data.

After reviews have been saved, create or refresh the fine-tuning manifest with:

```bash
.venv/bin/python -m aiwr.review_store
```

This writes `reviews/dataset.jsonl`. It uses the latest review for each source image and leaf ID. Only leaf-tissue reviews marked expert visual or laboratory confirmed are included. **Needs adjudication**, **Unknown / Other**, and stem or panicle records remain in the review store but are excluded from this leaf-classification manifest. Use specimen or collection IDs to keep related images together when making train, validation, and test splits. The manifest is an input for a future fine-tuning workflow; the repository does not yet contain a fine-tuning command.

## Validation status

The sample images do not have independent expert-confirmed per-leaf labels or reviewed lesion masks. **Diagnostic accuracy and severity error have not been measured; the 90% accuracy target remains unverified.** To compare automatic results against confirmed labels, copy `examples/labels_template.csv`, enter expert- or laboratory-confirmed labels, and run:

```bash
.venv/bin/python -m aiwr.evaluate outputs/rr8/leaves.csv confirmed_labels.csv
```

The evaluator counts abstentions and missing predictions in overall accuracy and reports coverage separately. Severity validation needs expert-reviewed symptom masks or independent measurements. Automated checks can be run with `.venv/bin/python -m pytest -q tests`.

## AI co-authorship

This repository was developed by a human researcher with assistance from AI coding tools. Generated code and diagnostic output require review and validation before research or operational use.
