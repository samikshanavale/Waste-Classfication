# Week 2 — Data Acquisition & Understanding

- **Date:** [FILL]
- **Work completed:** Downloaded TrashNet (2,527 images), inspected class counts, created a stratified 70/15/15 split (1,768/379/380), wrote the dataset card with licence and citations, and produced sample visualisations.
- **Evidence:** `data/DATASET_CARD.md`, `notebooks/01_eda.ipynb`, `data/processed/split_manifest.csv`, `src/data_pipeline.py`.
- **Experiments & Results:** Class counts: cardboard 403, glass 501, metal 410, paper 594, plastic 482, trash 137. The residual class is clearly under-represented.
- **Problems / Failure Analysis:** Raw TrashNet archive is ~3.5 GB; used the 43 MB resized release instead. Clean white backgrounds will not represent real disposal points — RealWaste added as a robustness set.
- **Technical Decisions:** Fixed seed 42, stratified split, sealed test set; model-specific scaling placed inside the model so the Flask app can pass raw pixels.
- **References used this week:** Yang & Thung (2016) TrashNet (dataset and licence); Single et al. (2023) RealWaste (robustness set).
- **Individual contributions:** Member 1 [FILL], Member 2 [FILL], Member 3 [FILL], Member 4 [FILL].
- **Next week plan:** Implement the custom CNN baseline E1 and record training/validation curves and metrics.
