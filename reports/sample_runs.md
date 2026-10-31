# Prototype test runs — 2026-10-30

These are automated screening results from the 0.1 prototype on controlled-background TIFF images. Disease identity and severity were **not** compared with expert ground truth. The percentages are heuristic visible color-change areas and must not be interpreted as confirmed disease area.

| Image | Leaf | Provisional color-change area | Prompt severity bin | Initial visual diagnosis |
| --- | --- | ---: | --- | --- |
| RR1.TIF | Leaf 1 | 2.65% | Low | Fungal Brown Spot (tentative) |
| RR8.TIF | Leaf 1 | 15.39% | Moderate | Leaf Spot (tentative) |
| RR8.TIF | Leaf 2 | 15.61% | Moderate | Leaf Spot (tentative) |
| RR60.TIF | Leaf 1 | 0.49% | Low | Leaf Spot (tentative) |
| RR60.TIF | Leaf 2 | 2.59% | Low | Fungal Brown Spot (tentative) |
| RR60.TIF | Leaf 3 | 15.60% | Moderate | Fungal Brown Spot (tentative) |

The three runs found 1, 2, and 3 leaf regions, respectively. A 10-image visual check of leaf counts matched all 10 inspected images, but this is not diagnostic accuracy. A batch decode/segmentation smoke test processed all 68 sample TIFFs without exceptions; the detected leaf-count distribution was 7 images with 1 leaf, 50 with 2, 7 with 3, 3 with 4, and 1 with 5.

Validation checks: 26 focused tests passed; the Streamlit app loaded a sample and rendered a cue-based Zonate Eyespot differential without exceptions. No independent per-leaf diagnosis labels or reviewed lesion masks were found, so pathogen accuracy, mask Dice/IoU, and severity error cannot yet be measured. The 90% diagnostic accuracy goal remains unverified.

The initial visual diagnoses are unvalidated rule-based choices from spot shape and color. The reviewer cue controls can refine the table, and reviewer-entered labels can override it. The automatic candidate is retained separately for later accuracy evaluation.
