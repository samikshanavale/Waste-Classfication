# Week 4 — Advanced Model Version 1

- **Date:** [FILL]
- **Work completed:** Built MobileNetV2 transfer-learning models: frozen ImageNet features (E3) and the same model with augmentation (E4). Added a named `logits` layer before softmax so calibration is possible.
- **Evidence:** `src/models.py:build_mobilenetv2`, `experiments/logs/E3.csv`, `experiments/logs/E4.csv`, `experiments/figures/curves_E3.png`, model checkpoints in `experiments/models/`.
- **Experiments & Results:** E3 validation accuracy 0.8338 (macro F1 0.7851); E4 with augmentation 0.6095 (F1 0.5611). Transfer learning clearly beats the CNN baseline (E1 0.2216); augmentation clearly hurt the transfer model. See `experiments/results/experiment_table.json`.
- **Problems / Failure Analysis:** Saving/reloading a model with a `Lambda` preprocessing layer failed; replaced it with an equivalent `Rescaling(2/255, -1)` layer, which is self-contained and serialisable.
- **Technical Decisions:** Keep the pretrained backbone frozen initially; use a smaller learning rate for later fine-tuning rather than updating all weights at once.
- **References used this week:** Sandler et al. (2018) MobileNetV2 (architecture choice); ImageNet pretrained weights (initialisation).
- **Individual contributions:** Member 3 [FILL] (lead), others [FILL].
- **Next week plan:** Complete the ablation (augmentation and fine-tuning E5) and compile the comparison table.
