HIGH-THROUGHPUT AI PIPELINE FOR WILD RICE LEAF DISEASE DIAGNOSIS

Complete text version of the original discussion

PURPOSE AND RECOMMENDED ARCHITECTURE

For thousands of wild rice leaf images, build a GPU-accelerated computer vision pipeline combining disease classification, lesion segmentation, and severity quantification. Use specialized vision models for bulk processing, with a capable vision-language model supporting difficult-case review and a plant pathologist confirming uncertain diagnoses.

Processing flow:
Original photographs and expert-confirmed reference images
  -> Image quality checks, resizing, and color normalization
  -> Leaf isolation and separation of individual leaves
  -> Disease classification
  -> Lesion segmentation
  -> Confidence evaluation and unknown-case detection
  -> Disease prediction and severity scoring
  -> CSV/Parquet records, masks, overlays, and review dashboard

Large vision-language models should not be the default tool for diagnosing every image independently. They can be slower, more expensive, and inconsistent for pixel-based severity estimation. Specialized models offer a practical route to efficient bulk inference once validated for this dataset.

This architecture is suitable for an HPC GPU environment and can scale from thousands to hundreds of thousands of images. A shared encoder may eventually combine classification and segmentation, but independently trained models provide a useful initial baseline.

This document preserves the recommendations and planning estimates from the original discussion. It is an implementation proposal, not a validated wild rice diagnostic system. Model availability and technical details should be checked when implementation begins.


1. WHICH VISION MODELS SHOULD YOU USE?

Evaluate several model families rather than committing to one before testing.

Model family       | Primary role                    | Reason to evaluate
-------------------|---------------------------------|----------------------------------------
DINOv2 / DINOv3     | Classification and embeddings   | Strong pretrained visual representations; useful with limited labels
ConvNeXt V2        | Fast disease classification     | Efficient inference and straightforward fine-tuning
SegFormer-B0/B2    | Leaf and lesion segmentation    | Pixel-level estimates of affected tissue
SAM 2             | Annotation assistance           | Can help create masks; does not diagnose disease
Multimodal VLM     | Difficult-case review           | Can compare references and describe symptoms; needs validation

Preferred initial baseline: DINOv2 embeddings plus a lightweight classifier and SegFormer-B0. Benchmark a DINOv3 small variant as an alternative rather than assuming it improves performance on this dataset.

The original discussion also proposed DINOv3-Small for the first embedding experiment. Treat the DINOv2 and DINOv3 options as comparison candidates: select the actual backbone based on held-out accuracy, speed, memory, and deployment constraints.

With very few confirmed examples, begin with a frozen pretrained encoder and nearest-neighbor retrieval or a lightweight classifier. As verified labels accumulate, compare this baseline with a fine-tuned ConvNeXt model and, if justified, a fine-tuned transformer encoder.

Segmentation and classification solve different problems. An encoder that separates image categories does not automatically produce accurate lesion boundaries. SAM-assisted masks also require review, especially for small spots and subtle discoloration.


2. HOW TO USE YOUR EXISTING REFERENCE IMAGES

Wild rice diseases may be poorly represented in general-purpose plant disease datasets. Expert-confirmed reference images are therefore central to a useful system.

Use the references to distinguish disease-specific visual patterns, including spot shape, discoloration, lesion boundaries, and distribution. Include noninfectious damage such as nutrient deficiency, senescence, and environmental stress so the model does not equate every abnormal color with infection.

Proposed embedding-based few-shot workflow:

1. Extract feature vectors for all images using a pretrained DINOv3 small encoder, with DINOv2 as a comparison baseline.
2. Store image embeddings with stable image identifiers.
3. Extract embeddings from expert-labeled reference images using the same preprocessing and encoder.
4. Classify unknown images using nearest-neighbor similarity or a lightweight supervised classifier.
5. Flag uncertain examples and images unlike the known references for expert review.
6. Update the labeled dataset and retrain as verified examples become available.

Clustering embeddings can help select representative images for annotation and expose groups that need additional disease categories or metadata. Similarity scores are not automatically calibrated probabilities of disease.

A handful of examples can support a prototype, but cannot establish reliable disease-level accuracy. Reference images should span severity levels, lighting conditions, collection sessions, leaf ages, and relevant species or varieties. Keep confirmed labels distinct from suspected visual labels.

Cultivated rice (Oryza sativa) and North American wild rice (Zizania species) have different disease contexts. A model trained on cultivated rice should not be treated as a validated wild rice diagnostic model without appropriate evaluation.


3. SEVERITY QUANTIFICATION

Produce both a continuous affected-area percentage and, where appropriate, a categorical severity grade.

For diseases where visible affected area is a valid severity measure:

Severity (%) = 100 x diseased leaf pixels / total visible leaf pixels

Segment each visible leaf separately when an image contains multiple leaves. Restrict the lesion mask to the corresponding leaf mask. Define explicitly whether chlorosis, necrosis, dark spots, and other symptoms count toward severity for each disease.

Do not count normal yellowing, shadows, reflections, background, or unrelated stress as infection. When the cause is unconfirmed, describe the measurement as visible symptom area rather than confirmed disease area.

Illustrative grading scale, subject to plant-pathologist approval:

Grade | Affected leaf area | Interpretation
------|--------------------|----------------
0     | 0%                 | No visible symptoms
1     | >0-5%              | Very mild
2     | >5-15%             | Mild
3     | >15-30%            | Moderate
4     | >30-50%            | Severe
5     | >50%               | Very severe

These thresholds are illustrative, not an established wild rice disease scale. Some diseases may require lesion count, lesion length, or symptom-pattern measures instead of area alone.

Retain high-resolution originals because small lesions may disappear during resizing. Consider high-resolution crops or tiles for lesion segmentation, and map outputs back to original image coordinates before calculating area. Validate segmentation at the intended processing resolution.

Severity reflects the visible photographed leaf area. It does not measure hidden tissue, the whole plant, or field-level disease burden. Poor images or unreliable masks should receive a review flag or no severity value.

Segmentation errors can systematically bias severity estimates. Compare predicted percentages with independently reviewed masks and expert assessments before using the measurements in research.


4. A MORE EFFICIENT MODEL ARCHITECTURE: SHARED VISUAL ENCODER

For a larger production system, investigate one encoder serving classification and segmentation.

DINOv3 small or ConvNeXt tiny backbone
  -> Classification head: disease probabilities, multi-label predictions
  -> Segmentation decoder: leaf mask and lesion mask
  -> Confidence/unknown evaluation
  -> Combined report: disease, severity, uncertainty, and masks

A shared backbone can reduce duplicate feature extraction. Dense pretrained features may support lightweight downstream segmentation heads, but the complete system still needs task-specific training and validation.

Benchmark the shared model against independently trained classification and segmentation models. Measure disease accuracy, lesion-mask quality, severity error, GPU memory, and end-to-end throughput. Choose the shared design only if its efficiency benefits preserve the required accuracy.

Unknown-case detection requires an explicit method and validation; it does not arise automatically from adding a classification head. Likewise, co-occurring diseases require multi-label training data and an appropriate prediction head.


5. GPU AND HPC EXECUTION STRATEGY

For several thousand images, one modern GPU may be sufficient for inference. Multiple GPUs become useful when workload size, processing resolution, or turnaround requirements justify them.

Suggested stack:

Component             | Technology
----------------------|-------------------------------------------
Model training        | PyTorch
Image preprocessing   | OpenCV and Pillow
Image encoder         | DINOv3 small candidate; benchmark DINOv2
Classification        | Logistic regression initially, then a trained/fine-tuned head
Segmentation          | SegFormer-B0 or lightweight U-Net
Annotation            | CVAT or Label Studio
Experiment tracking   | MLflow
HPC orchestration     | Slurm
GPU optimization      | BF16/FP16 where supported, batched inference
Results               | Parquet, CSV, and PNG masks/overlays

Load each model once per worker and process batches or image shards. Avoid submitting one Slurm job per image. Use job arrays for independent shards when dataset size or scheduling needs justify them.

Benchmark batch size and image resolution on the actual GPU. Verify that mixed precision preserves predictions and mask quality. Parallelize CPU image loading and preprocessing where needed, and measure TIFF decoding, storage I/O, mask writing, and output serialization as part of total runtime.

Retain original TIFF files and stable image checksums. Make shard processing resumable so completed images need not be processed again after an interruption. Record failures explicitly rather than silently skipping unreadable images.

Approximate throughput planning scenarios (not measured benchmarks):

Assumed sustained throughput | Time for 10,000 images
-----------------------------|-----------------------
20 images/second             | 500 seconds (8 minutes 20 seconds)
100 images/second            | 100 seconds (1 minute 40 seconds)
250 images/second            | 40 seconds

Processing time in seconds = image count / sustained images per second.

These scenarios exclude annotation, training, model loading, queue delays, and human review. Actual speed depends on resolution, segmentation complexity, hardware, decoding, storage, and output generation. A 100-images/second end-to-end pipeline is a planning assumption, not a guaranteed rate for high-resolution wild rice photographs.


6. HOW TO ACHIEVE HIGH DIAGNOSTIC ACCURACY

Labeling and validation are likely to matter more than simply selecting a larger model.

Expert ground truth:
Have a plant pathologist define and confirm disease categories. Images alone may not identify pathogens with overlapping symptoms; some cases require field metadata or laboratory confirmation.

Multi-label prediction:
Allow more than one disease or stress condition per leaf when supported by the annotation scheme.

Unknown detection:
Do not force every image into a known disease class. Include an abstention or review pathway for unfamiliar symptoms and distribution shifts.

Active learning:
Prioritize uncertain images, underrepresented groups, and cases where models or reference retrieval disagree. Include representative confident cases in periodic audits.

Data leakage prevention:
Split training, validation, and test data by plant, specimen, collection session, plot, or field. Keep near-duplicate images and crops from the same original in the same split.

Calibrated confidence:
Use validation data to calibrate predictions and set automatic-acceptance thresholds. Evaluate performance specifically among automatically accepted images and report how many images are routed to review.

Evaluation metrics:
- Disease classification: macro-F1, per-class recall, precision, and confusion matrices.
- Multi-label classification: appropriate per-label and aggregate metrics.
- Leaf/lesion masks: Dice and intersection over union (IoU).
- Severity: mean absolute error in percentage points and checks for systematic bias.
- Operations: end-to-end throughput, GPU memory, processing failures, and review workload.

Use an independent expert-labeled test set before allowing automated results into scientific analysis. Preserve test-set independence during model selection and active learning. Report results by disease and severity group, since strong aggregate performance can hide failure on rare or subtle symptoms.


7. RECOMMENDED IMPLEMENTATION ROADMAP

Phase 1: Rapid proof of concept
- Extract DINOv3 features, with DINOv2 as a useful baseline.
- Use confirmed reference images for similarity classification.
- Cluster the unlabeled dataset and inspect representative examples.
- Assess which diseases separate visually and where more labels are needed.
Suggested initial effort: 1-3 working days.

Phase 2: Expert annotation and segmentation
- Select representative images across clusters and severity levels.
- Create verified disease labels and a smaller collection of reviewed pixel-level masks.
- Define disease-specific severity rules and distinguish suspected from confirmed diagnoses.
Suggested initial effort: 3-10 working days, depending on expert availability.

Phase 3: Model comparison
- Compare a frozen DINOv3 classifier with fine-tuned ConvNeXt and a segmentation model.
- Evaluate held-out accuracy, severity error, memory, and throughput.
- Test resolution and tiling choices for small lesions.
Suggested initial effort: 2-5 working days.

Phase 4: Production HPC pipeline
- Package selected models in a reproducible Slurm workflow.
- Implement batched inference, quality control, resumable shards, and structured output.
- Add uncertainty review, overlays, model/version tracking, and acceptance criteria.
Suggested initial effort: 2-5 working days.

These are engineering estimates for a prototype, not guarantees of research-grade accuracy. Expert labeling, laboratory confirmation, and dataset diversity may determine the true schedule.


8. EXAMPLE OUTPUT

Each image should generate a structured record. If there are multiple leaves, use one record per leaf with a shared image identifier, or an image record linked to per-leaf records.

Illustrative record from the original discussion:

Field                  | Example
-----------------------|-----------------------------
image_id               | leaf_000127.jpg
predicted_disease      | disease_A
confidence             | 0.94
second_prediction      | disease_B
severity_percent       | 23.6
severity_grade         | 3
leaf_area_pixels       | 186420
lesion_area_pixels     | 43995
review_required        | false
segmentation_mask      | masks/leaf_000127.png

All values are illustrative. The mask filename and disease names are placeholders. A review_required value of false is appropriate only after validation establishes the required acceptance criteria. A numeric confidence value must have a documented meaning and calibration method.

Additional reproducibility fields:
- leaf_id and specimen_id
- species or variety, if known
- collection session, plot, or field identifiers
- original image checksum
- model version and preprocessing version
- inference date
- prediction status and quality-control flags
- diagnostic confirmation status
- class probabilities or multi-label scores where appropriate
- leaf-mask, lesion-mask, and overlay paths
- error or abstention reason when no usable result is produced

CSV supports convenient inspection; Parquet is useful for efficient analysis. Preserve masks and overlays for audit and expert review. Keep suspected visual diagnoses separate from laboratory- or expert-confirmed labels.


9. THREE QUESTIONS THAT WILL DETERMINE THE OPTIMAL DESIGN

1. How many confirmed reference images do you have per disease?
   - Fewer than 10
   - 10-100
   - More than 100

2. What do the images look like?
   - Leaves photographed in the field, with backgrounds
   - Individual leaves on uniform backgrounds
   - A mixture of both

3. What GPU infrastructure will you use?
   - HPC GPU cluster with Slurm
   - Single workstation GPU
   - Not yet decided

The next useful step is to assess 5-10 representative images, including healthy leaves and examples of each suspected disease at different severity levels. Confirm reference diagnoses with a wild rice pathologist where possible. Use those samples to evaluate leaf isolation, small-lesion visibility, disease separability, and annotation needs before scaling training.

For a sample such as RR8.TIF, separate leaf masks and lesion overlays would be a useful experiment, followed by comparison with an expert assessment. This pipeline document does not contain measured segmentation outputs or a confirmed diagnosis for that sample.

END OF DOCUMENT
