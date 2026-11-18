# Week 5 — Comparison / Ablation

- **Date:** [FILL]
- **Work completed:** Completed the full experiment matrix E1–E5, including the augmentation ablation and fine-tuning of the last MobileNetV2 block at a smaller learning rate. Produced the controlled comparison table.
- **Evidence:** `experiments/results/experiment_table.json`, `experiments/logs/*.csv`, `report/tables.tex` (macro `\\experimentTable`).
- **Experiments & Results:** E1 0.2216 / F1 0.0743 (391,398 params, 9.9 ms), E2 0.2348 / 0.0634 (10.2 ms), E3 0.8338 / 0.7851 (2,265,670 params, 8.9 ms), E4 0.6095 / 0.5611 (13.4 ms), E5 0.6517 / 0.5703 (9.2 ms). E3 (frozen MobileNetV2) won on validation accuracy; the CNN baseline overfit, and for the transfer model both augmentation and fine-tuning degraded this clean, small dataset.
- **Problems / Failure Analysis:** Augmentation clearly hurt the transfer-learning model (E4<E3) and only shifted the CNN's accuracy/F1 trade-off (E2 accuracy slightly above E1, F1 below), consistent with clean studio photos; fine-tuning the last block also reduced validation accuracy (E5<E3), so we kept the frozen backbone. Increasing epoch count and tuning the learning rate resolved the under-training seen in week 3.
- **Technical Decisions:** Select the final model using validation accuracy only; the test set is still sealed.
- **References used this week:** Sandler et al. (2018) and Guo et al. (2017) (planning the calibration step).
- **Individual contributions:** Members 2 and 3 [FILL], Members 1 and 4 [FILL].
- **Next week plan:** Add temperature scaling, abstention analysis and the cross-dataset robustness experiment.
