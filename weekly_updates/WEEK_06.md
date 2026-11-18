# Week 6 — Improvement & Robustness

- **Date:** [FILL]
- **Work completed:** Fitted temperature scaling on validation logits, selected an abstention threshold from the validation risk–coverage curve, evaluated once on the untouched test set, and ran a cross-dataset robustness test on a fixed RealWaste subset.
- **Evidence:** `src/evaluate.py`, `src/robustness.py`, `experiments/figures/calibration.png`, `experiments/figures/risk_coverage.png`, `experiments/figures/robustness_comparison.png`, `experiments/results/summary.json`.
- **Experiments & Results:** ECE 0.0444 → 0.0350 after temperature scaling (test accuracy unchanged at 0.8263); abstention threshold 0.75 gives 0.6553 coverage and 0.9438 accuracy on answered images (131 flagged). RealWaste accuracy 0.4292 versus 0.8263 in-distribution, with mean confidence 0.2665 on the shifted set.
- **Problems / Failure Analysis:** Accuracy drops on realistic landfill photographs because of illumination and contamination differences; this is reported as a limitation and motivates abstention.
- **Technical Decisions:** Never tune on the test set; threshold and temperature come from validation only. RealWaste is evaluation-only, never mixed into training.
- **References used this week:** Guo et al. (2017) temperature scaling and ECE; Single et al. (2023) RealWaste dataset.
- **Individual contributions:** Member 3 [FILL] (lead calibration/robustness), Member 4 [FILL] (metrics), others [FILL].
- **Next week plan:** Add Grad-CAM explanations and finish the Flask web prototype.
