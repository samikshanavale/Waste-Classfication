# Week 7 — Explainability / Efficiency / Deployment

- **Date:** [FILL]
- **Work completed:** Implemented Grad-CAM, generated an explanation gallery of correct and incorrect predictions, and integrated the final model into a Flask application with bin mapping, uncertainty warning and heatmap overlay.
- **Evidence:** `src/explain.py`, `experiments/figures/gradcam_examples.png`, `src/app.py`, `src/templates/index.html`, `src/static/samples/`.
- **Experiments & Results:** Qualitative analysis of paper/cardboard and glass/plastic confusion. Latency and parameter counts are recorded in `summary.json`.
- **Problems / Failure Analysis:** Some heatmaps focus on printed labels rather than the object; this is discussed in the report as a limitation of Grad-CAM and a reason to keep human verification.
- **Technical Decisions:** Use Grad-CAM as an inspection aid, not as a claim of causal reasoning; show the uncertainty warning whenever confidence is below the validation-chosen threshold.
- **References used this week:** Selvaraju et al. (2017) Grad-CAM.
- **Individual contributions:** Member 4 [FILL] (lead demo and XAI), others [FILL].
- **Next week plan:** Freeze results, compile the LaTeX report and the 7-slide presentation, rehearse the live demo.
