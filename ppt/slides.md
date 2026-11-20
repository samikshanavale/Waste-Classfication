# 7-slide viva presentation (speaker notes included)

## Slide 1 — Title
MobileNetV2 transfer (frozen)

## Slide 2 — Problem & real-life scenario
- Contamination, user uncertainty, hostel/canteen example, need to abstain.

## Slide 3 — Abstract & pipeline
- Objective and full data → model → confidence → bin workflow.

## Slide 4 — Implementation
- TrashNet 70/15/15, RealWaste robustness, CNN baseline, MobileNetV2, calibration, Flask.

## Slide 5 — New innovation
- Uncertainty + abstention, Grad-CAM evidence, measured efficiency.

## Slide 6 — Results
- Test accuracy: 0.824; macro F1: 0.799; ECE: 0.042 → 0.042.
- Abstention at τ=0.70: coverage 0.737, accuracy on answered 0.904, flagged 100.

## Slide 7 — Demo & impact
- Live Flask demo, real-life scenario, limitations, future work.

See `slides.pptx` for the formatted deck and `viva/cheat_sheet.md` for the defence script.