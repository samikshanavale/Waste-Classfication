# Viva defence sheet

**Rule:** never quote a number you cannot point to in `experiments/results/summary.json`.
Open that file (and the Flask app) before the viva starts.

## 90-second opening script

> "Our problem is that people mis-sort waste because they are unsure which bin an item
> belongs to. We built an image-based assistant: a user photographs an item, a deep model
> identifies the material, and the app recommends a bin. The important part is that it is
> confidence-aware: if the model is uncertain, it does not give an instruction — it asks
> for manual verification. We compared a custom CNN baseline with MobileNetV2 transfer
> learning, calibrated the probabilities on validation data, evaluated once on a sealed
> test set, tested robustness on realistic landfill photos, and deployed the result as a
> Flask web app. [Demo.]"

## Questions you should be able to answer

**1. Why this topic?**
It is a real, measurable decision-support problem: incorrect segregation contaminates
recyclables. The input (an image) and output (a material category plus bin) are well defined,
and a wrong answer has a real cost, which makes uncertainty handling necessary.

**2. How is it different from the sample "Intelligent Waste Sorting" topic?**
The sample is a generic sorting/detection problem. Our problem is *disposal guidance*:
the deliverable is a calibrated decision with abstention, an explainability view for the
operator, and a measured deployment trade-off. The learning formulation, evaluation
(calibration/ECE/risk–coverage) and user-facing output are different.

**3. Why classification and not object detection or segmentation?**
The demo assumes one prominent item at a disposal point — a bottle, a can, a carton. For
that scenario classification is sufficient and much cheaper. We explicitly state the
limitation: if a camera sees a bin containing many overlapping objects, detection would be
the correct formulation. This is future work, not an oversight.

**4. Why TrashNet and why RealWaste?**
TrashNet is small, MIT-licensed and has exactly the five material classes in the proposal
plus a residual class, so we can run controlled experiments quickly. Its clean backgrounds
are unrealistic, so we added RealWaste — real landfill photographs — as an evaluation-only
robustness set. This follows the guideline's demand to study distribution shift.

**5. Why MobileNetV2 over a bigger model or EfficientNet?**
It is a lightweight pretrained vision model with good ImageNet features and an explicit
accuracy–latency trade-off. We measured parameters, checkpoint size and milliseconds per
image for every candidate rather than assuming the smaller model was good enough.

**6. What is temperature scaling and why do you need it?**
Deep networks are often overconfident. We divide logits by a temperature T fitted on
validation data to minimise negative log-likelihood, so reported confidence better matches
observed accuracy. We measure this with ECE ([quote before/after] in the slide).

**7. What is abstention and how did you choose the threshold?**
If the maximum calibrated probability is below τ we refuse to recommend a bin and ask for
manual verification. τ is the highest coverage whose validation accuracy on answered images
stays above a target; the target is relaxed from 0.90 to 0.85/0.80/0.75 only if no threshold
qualifies, so the mechanism still works for a weaker model. The test set was not used to
choose it. Read the resulting threshold, coverage and answered-accuracy from
`test.abstention_on_test` in `summary.json`.

**8. What is Grad-CAM used for here?**
It highlights the image region that influenced the prediction. We use it to inspect the
dominant error pairs measured in the confusion matrix (plastic→glass, glass→metal,
metal→glass in our run) and show it in the app as evidence for the user. It is an inspection
aid, not proof of causal reasoning.

**9. What failed / what was hard?**
The from-scratch CNN overfits the small dataset. Counter-intuitively, augmentation *hurt*
both families here (E2 vs E1, E4 vs E3) because TrashNet images are clean studio photos;
fine-tuning the last MobileNet block also degraded validation accuracy (E5 vs E3), so the
frozen pretrained model was selected. A first short run was under-trained, so we increased
epochs. Saving/reloading a Lambda preprocessing layer failed, so we replaced it with a
Rescaling layer. Accuracy drops on RealWaste photos; we report that drop honestly.

**10. What are the limitations?**
Single-object, closed-set classification; clean-background training data; small residual
class; local bin rules differ; Grad-CAM is approximate. All are stated in the report.

**11. What is the "new innovation"?**
An applied systems contribution: calibrated confidence + abstention + Grad-CAM evidence +
measured efficiency, each tied to a real failure mode and evaluated. Not a new algorithm —
say this plainly if asked.

**12. How is the work divided among four members?**
Data pipeline and EDA; baseline CNN and training loop; advanced model, calibration and
robustness; evaluation, XAI, Flask demo and report. Each stream has artefacts in the repo
and weekly update entries.

## Live demo script (say this while clicking)

1. "Here is the upload page; I'll use a bundled sample first so the demo is deterministic."
2. Click the **cardboard** (or **paper**) sample. "The app predicts the material with high
   confidence and recommends the matching bin. The right-hand image is the Grad-CAM heatmap."
   Read the exact category and confidence from the screen — do not memorise a number.
3. Click the **glass** or **plastic** sample. "Here the confidence is below the threshold, so
   the app refuses to recommend a bin and asks for verification — this is the safety
   behaviour we designed." (In our local run these two samples were flagged; confirm live.)
4. "If you want to reproduce this: `py -3.13 src/app.py` after `py -3.13 run_all.py`."

## If something goes wrong during the demo

- If the model misclassifies an item, do **not** panic: say "this is exactly the case our
  abstention mechanism is for" and show the confidence/flag.
- If the app fails to start, explain that the model checkpoint must be generated first and
  show the saved prediction figures and `summary.json` instead.
- Never edit results live. Quote the frozen files.

## Files to keep open in tabs

- `experiments/results/summary.json` — all numbers
- `ppt/slides.pptx` — the deck
- `report/report.tex` — the report
- `http://127.0.0.1:5000` — the live demo
