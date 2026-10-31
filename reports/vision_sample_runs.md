# Local vision-model smoke runs — 2026-10-30

Runtime: `agc04` NVIDIA A40; `DeepLearningHCL` PyTorch 2.4.1 with user-site packages disabled; project `.venv-hcl` overlay with Transformers 4.57.6. Models: pretrained Qwen2.5-VL 3B and 7B, neither fine-tuned on wild rice. Leaf crops and provisional severity masks came from the existing rule-based image analysis. Each image used `max_side=900`.

| Model | Image | Leaf | Vision-model primary diagnosis | Reported visible cue | Provisional color-change area |
| --- | --- | --- | --- | --- | ---: |
| 3B | RR1.TIF | Leaf 1 | Insufficient evidence for a disease-specific diagnosis | Small brown or dark-purple spots; explanation mentioned Fungal Brown Spot as a possibility | 2.49% |
| 3B | RR8.TIF | Leaf 1 | Insufficient evidence for a disease-specific diagnosis | Small circular or oval dark spots | 10.90% |
| 3B | RR8.TIF | Leaf 2 | Insufficient evidence for a disease-specific diagnosis | Small circular or oval dark spots | 11.13% |
| 7B | RR1.TIF | Leaf 1 | Spot Blotch (tentative vision-model diagnosis) | Small dark brown spots without visible yellow halos | 2.49% |

Both checkpoints loaded and produced valid structured outputs. The 3B model abstained on all three tested leaves, including one explanation that described a possible fungal pattern. Rewording the prompt to request the closest tentative category did not change that result. The 7B model selected Spot Blotch on RR1. The app correctly retains each model's own output rather than substituting the rule-based prediction. These runs demonstrate working GPU inference and different model behavior; they do **not** measure diagnostic accuracy or show that 7B is more accurate. Expert-confirmed labels and reviewed masks are still required. Reference PDF illustrations are not currently supplied to the model as images.
