# Wild Rice Leaf Disease Diagnosis

A proposed GPU-accelerated pipeline for classifying wild rice leaf symptoms, segmenting leaves and lesions, estimating visible symptom severity, and routing uncertain cases to expert review. The target is thousands of photographs, with an HPC path for larger collections.

**Status:** Planning. The model choices, severity grades, and throughput figures below are proposals, not validated wild rice diagnostic results. Expert-confirmed labels and an independent test set are required before using predictions in research.

## Processing flow

1. Ingest original photographs and expert-confirmed reference images. Preserve high-resolution originals, stable image IDs, checksums, and collection metadata.
2. Check image quality; decode, resize, and normalize color with consistent, versioned preprocessing.
3. Isolate leaves and assign a separate leaf ID and mask to each visible leaf.
4. Predict disease or stress categories, allowing multiple labels when the annotation scheme supports them.
5. Segment lesions and other disease-specific symptoms within each leaf mask.
6. Evaluate confidence and unfamiliar cases; send poor images, uncertain predictions, and unreliable masks for expert review.
7. Calculate visible symptom severity per leaf when the masks and disease-specific rules support it.
8. Export per-leaf records, masks, overlays, and a review queue or dashboard.

Specialized vision models are the starting point for bulk processing. A multimodal vision-language model may help review difficult cases, but its diagnoses and severity estimates also need validation. A plant pathologist should confirm uncertain diagnoses.

## Roadmap and tasks

### Phase 1: Data audit and proof of concept

- [ ] Locate the image collection and count images, healthy examples, and expert-confirmed references per disease.
- [ ] Record image format, dimensions, species or variety, leaf age, lighting, collection session, specimen, plot or field, and diagnosis confirmation status where available.
- [ ] Identify duplicates and near-duplicates; retain original TIFFs and other high-resolution files.
- [ ] Review 5–10 representative images spanning healthy leaves, suspected diseases, severity levels, field and uniform backgrounds, multiple leaves, and difficult cases.
- [ ] Test quality checks, leaf isolation, and small-lesion visibility after resizing. Try separate leaf masks and lesion overlays on a sample such as RR8.TIF, then compare with expert assessment.
- [ ] Extract frozen DINOv2 and DINOv3 small embeddings using the same preprocessing for reference and unlabeled images; store embeddings with stable IDs.
- [ ] Compare nearest-neighbor reference retrieval with a lightweight classifier, such as logistic regression. Cluster unlabeled embeddings to select annotation candidates and expose possible missing categories.
- [ ] Record where diseases separate visually, where they overlap, and what additional labels or metadata are needed.

A handful of references can support a prototype, but cannot establish diagnostic accuracy. Similarity scores are not calibrated disease probabilities. Cultivated rice models are not validated for North American wild rice without wild rice evaluation.

### Phase 2: Expert labels and segmentation

- [ ] Have a wild rice pathologist define disease, healthy, unknown, and noninfectious-stress categories, including nutrient deficiency, senescence, and environmental damage.
- [ ] Keep suspected visual labels separate from expert- or laboratory-confirmed diagnoses. Capture field or laboratory evidence when visual symptoms overlap.
- [ ] Decide whether co-occurring diseases and stresses require multiple labels per leaf.
- [ ] Select representative images across embedding clusters, severity levels, lighting, collection sessions, and varieties for annotation.
- [ ] Create reviewed leaf and lesion masks. Use SAM 2 only to assist annotation; review its boundaries, especially small spots and subtle discoloration.
- [ ] Define disease-specific severity rules: which areas count as chlorosis, necrosis, or lesions; whether area, lesion count, lesion length, or symptom pattern is the appropriate measure.
- [ ] Establish when image quality or mask uncertainty requires review or a missing severity value.

### Phase 3: Model comparison and validation

- [ ] Compare frozen DINOv2 and DINOv3 embeddings, a fine-tuned ConvNeXt V2 classifier when labels permit, and any justified fine-tuned transformer model.
- [ ] Evaluate SegFormer-B0/B2 or a lightweight U-Net for separate leaf and lesion masks.
- [ ] Test segmentation resolution and high-resolution crops or tiles; map tile predictions to original image coordinates before area calculations.
- [ ] Implement explicit unknown-case detection and an abstention route. Train a multi-label head only if verified multi-label data exists.
- [ ] Consider a shared encoder with classification and segmentation heads after independent baselines work; keep it only if measured efficiency gains preserve accuracy.
- [ ] Split training, validation, and independent test data by plant, specimen, collection session, plot, or field. Keep crops and near-duplicates from one original in the same split.
- [ ] Calibrate confidence and choose automatic-acceptance thresholds on validation data. Measure accuracy among accepted cases and the fraction sent for review.
- [ ] Evaluate on an independent expert-labeled test set before using automated results in scientific analysis. Report results by disease and severity group.

### Phase 4: Production inference and review

- [ ] Package the selected models, preprocessing, and configuration in a reproducible, versioned workflow.
- [ ] Load each model once per worker; process batches or image shards, with Slurm job arrays when they help. Make shards resumable and record failures explicitly.
- [ ] Benchmark batch size, image resolution, BF16/FP16 if supported, GPU memory, CPU preprocessing, TIFF decoding, storage I/O, mask writing, and end-to-end throughput on the actual hardware.
- [ ] Export CSV and/or Parquet records plus PNG masks and overlays. Build a review queue or dashboard for uncertain cases.
- [ ] Track model and preprocessing versions, image checksums, inference dates, review decisions, and retraining data.
- [ ] Use active learning to prioritize uncertain images, underrepresented groups, and disagreements between models and reference retrieval; periodically audit representative confident cases.

## Severity measurement

For a disease where affected area is an appropriate measure:

**Visible affected area (%) = 100 × affected pixels inside the leaf mask / visible leaf pixels.**

Compute this separately for each leaf. Do not count background, shadows, reflections, normal yellowing, or unrelated stress as disease. If the cause has not been confirmed, label the result **visible symptom area**, not confirmed disease area. The measure covers photographed leaf tissue only, not hidden tissue, the whole plant, or field-level burden.

The following grades are illustrative and need pathologist approval:

| Grade | Visible affected area | Description |
| --- | --- | --- |
| 0 | 0% | No visible symptoms |
| 1 | >0–5% | Very mild |
| 2 | >5–15% | Mild |
| 3 | >15–30% | Moderate |
| 4 | >30–50% | Severe |
| 5 | >50% | Very severe |

Compare predicted percentages against independently reviewed masks and expert assessments. Report mean absolute error in percentage points and systematic bias; small segmentation errors can materially affect severity.

## Candidate tools and models

| Role | Candidates |
| --- | --- |
| Image loading and preprocessing | Pillow, OpenCV |
| Embeddings and initial classification | Frozen DINOv2 or DINOv3 small; nearest neighbors or logistic regression |
| Later disease classification | Fine-tuned ConvNeXt V2 or transformer encoder |
| Leaf and lesion segmentation | SegFormer-B0/B2 or lightweight U-Net |
| Mask annotation assistance | SAM 2 with human review |
| Difficult-case review | Validated multimodal vision-language model and plant pathologist |
| Training and experiment tracking | PyTorch, MLflow |
| Annotation | CVAT or Label Studio |
| HPC orchestration | Slurm |
| Outputs | CSV or Parquet; PNG masks and overlays |

Choose the actual models based on held-out disease accuracy, mask quality, severity error, memory, throughput, and deployment constraints. Classification quality does not establish lesion-boundary quality.

## Evaluation and acceptance

- **Disease prediction:** macro-F1, per-class precision and recall, confusion matrices, and appropriate per-label metrics for multi-label cases.
- **Segmentation:** Dice and intersection over union for leaf and lesion masks.
- **Severity:** mean absolute error in percentage points and bias by disease and severity group.
- **Operations:** end-to-end throughput, GPU memory, processing failures, and expert review workload.
- **Automatic use:** define an acceptance threshold only after confidence calibration and independent evaluation; otherwise flag the case for review.

Throughput examples in the source plan (20, 100, or 250 images per second) are planning scenarios, not measured performance. Training, annotation, queue delays, model loading, and human review are additional costs.

## Output records

Write one record per leaf, linked to its source image. Include:

- Image ID, leaf ID, specimen ID, source checksum, species or variety, and collection/session/plot/field IDs where known.
- Predicted disease or multi-label scores, documented confidence, alternative predictions, and diagnostic confirmation status.
- Visible severity percentage and any approved grade, leaf and lesion pixel areas, and the rule used to calculate severity.
- Quality flags, prediction status, review-required flag, abstention or error reason, and review outcome.
- Leaf-mask, lesion-mask, and overlay paths; model version, preprocessing version, and inference date.

Keep masks and overlays for audit and expert review. Never silently skip unreadable images or force unfamiliar cases into a known class.

## Inputs needed to start

1. Image dataset location and approximate count.
2. Number of expert-confirmed references per disease, including healthy and noninfectious examples.
3. Typical photographs: field backgrounds, isolated leaves, or a mix.
4. Available GPU environment: Slurm cluster, single workstation GPU, or undecided.
5. Access to a wild rice pathologist for labels, severity definitions, and review.

The first executable milestone is an image inventory and a 5–10-image sample review.
