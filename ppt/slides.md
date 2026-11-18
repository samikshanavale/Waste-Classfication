# 7-slide viva presentation (speaker notes included)

## Slide 1 — Title
{'id': 'E3', 'name': 'MobileNetV2 transfer (frozen)', 'checkpoint_name': 'mobilenetv2_frozen', 'selection_criterion': 'validation accuracy'}

## Slide 2 — Problem & real-life scenario
- Contamination, user uncertainty, hostel/canteen example, need to abstain.

## Slide 3 — Abstract & pipeline
- Objective and full data → model → confidence → bin workflow.

## Slide 4 — Implementation
- TrashNet 70/15/15, RealWaste robustness, CNN baseline, MobileNetV2, calibration, Flask.

## Slide 5 — New innovation
- Uncertainty + abstention, Grad-CAM evidence, measured efficiency.

## Slide 6 — Results
- Test accuracy: 0.8263157894736842; ECE: 0.059 → 0.0419.
- Abstention: {'threshold': 0.7, 'coverage': 0.7026, 'accuracy_on_answered': 0.9139, 'flagged_for_review': 113}.

## Slide 7 — Demo & impact
- Live Flask demo, real-life scenario, limitations, future work.

See `slides.pptx` for the formatted deck and `viva/cheat_sheet.md` for the defence script.