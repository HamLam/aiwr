# Wild Rice Leaf Disease Diagnosis

This project aims to build and validate a computer vision pipeline for wild rice leaf images. The intended outputs are disease predictions, leaf and lesion masks, visible symptom severity, and records for expert review. The system is still in the planning stage; no model or diagnostic accuracy has been validated.

## To-do list

### 1. Inventory images and references

- [ ] Count images and expert-confirmed reference images by disease and healthy category.
- [ ] Record file formats, image dimensions, species or variety, and collection metadata where available.
- [ ] Assign stable image IDs and checksums; retain the original high-resolution files.
- [ ] Identify near-duplicates and images from the same specimen or collection session.

### 2. Review representative images

- [ ] Select 5–10 images covering healthy leaves, each suspected disease, different severity levels, and difficult cases.
- [ ] Check field versus uniform backgrounds, multiple leaves, image quality, and visibility of small lesions after resizing.
- [ ] Test leaf isolation and produce sample leaf masks and lesion overlays for expert review.

### 3. Define labels and severity rules with an expert

- [ ] Establish disease, healthy, unknown, and noninfectious-stress categories.
- [ ] Keep confirmed diagnoses separate from suspected visual labels.
- [ ] Decide whether co-occurring conditions require multiple labels per leaf.
- [ ] Define which visible symptoms count toward severity for each disease, and when severity should be withheld.

### 4. Prepare an evaluation-ready dataset

- [ ] Create train, validation, and independent test splits grouped by specimen, collection session, plot, or field to prevent leakage.
- [ ] Select representative images for verified labels and reviewed pixel-level leaf and lesion masks.
- [ ] Preserve the independent test set during model selection and active learning.

### 5. Build classification baselines

- [ ] Extract embeddings with DINOv2 and a DINOv3 small candidate using consistent preprocessing.
- [ ] Compare reference-image retrieval and a lightweight supervised classifier.
- [ ] Cluster unlabeled images to find representative annotation candidates and possible missing categories.
- [ ] Add an explicit unknown or uncertain-case review route.

### 6. Build and assess segmentation and severity

- [ ] Evaluate a lightweight segmentation model, such as SegFormer-B0, for separate leaf and lesion masks.
- [ ] Test image resolution and high-resolution crops or tiles for small lesions.
- [ ] Calculate visible affected-area percentage per leaf from reviewed mask definitions.
- [ ] Compare predicted masks and percentages with independently reviewed masks and expert assessments.

### 7. Compare models and set acceptance criteria

- [ ] Compare the embedding baseline with a fine-tuned classifier when enough verified labels exist.
- [ ] Measure per-class precision and recall, macro-F1, mask Dice/IoU, severity error and bias, and review workload.
- [ ] Calibrate confidence and set thresholds for automatic results versus expert review using validation data.
- [ ] Report performance by disease and severity group on the independent test set.

### 8. Build the reproducible HPC pipeline

- [ ] Package selected models and preprocessing in a versioned workflow for the available GPU environment.
- [ ] Implement batched inference, quality checks, resumable image shards, and explicit failure records.
- [ ] Export per-leaf CSV or Parquet records, masks, overlays, model versions, and review flags.
- [ ] Benchmark end-to-end throughput, GPU memory, decoding, storage I/O, and output writing on actual hardware.

## Inputs needed to begin

- Image dataset location and approximate image count.
- Number of expert-confirmed references per disease.
- Typical image setup: field backgrounds, isolated leaves, or both.
- Available GPU environment: Slurm cluster, workstation GPU, or undecided.
- Access to a wild rice pathologist for label and severity review.
