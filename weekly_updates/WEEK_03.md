# Week 3 — Baseline System

- **Date:** [FILL]
- **Work completed:** Implemented the custom CNN (three conv blocks + batch norm + global average pooling + dropout) with a reproducible training loop, early stopping and checkpointing. Defined the evaluation metrics before training.
- **Evidence:** `src/models.py:build_custom_cnn`, `src/train.py`, `src/evaluate.py`, `experiments/logs/E1.csv`, `experiments/figures/curves_E1.png`.
- **Experiments & Results:** E1 (no augmentation) validation accuracy 0.2216 (macro F1 0.0743); E2 (with augmentation) 0.2348 (F1 0.0634). CNN is the reference point for all later comparisons; augmentation gave a marginal accuracy gain but lowered macro F1, so it was not a clear win on these clean-background images.
- **Problems / Failure Analysis:** Small dataset and class imbalance cause overfitting; early stopping and augmentation are used to address it. First short run (4 epochs) was under-trained, so epoch count was increased.
- **Technical Decisions:** Use macro F1 in addition to accuracy because the residual class is small; keep the test set untouched.
- **References used this week:** No external references used this week (baseline implemented from course material).
- **Individual contributions:** Member 2 [FILL] (lead), others [FILL].
- **Next week plan:** Implement MobileNetV2 transfer learning (E3/E4) with an explicit logits layer for later calibration.
