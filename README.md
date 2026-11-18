# Image-Based Waste Classification and Smart Bin Recommendation System

**Course:** Deep Learning (R5IT4109L) — 8-week advanced project
**Team:** Om Walunj (241081901), Samiksha Navale (241081903), Harshada Khanjode (241081909), + 4th member TBD
**Project topic (as proposed):** image-based waste identification with confidence-aware disposal guidance.

> **Distinctiveness statement:** this project is a *decision-support system for correct bin
> disposal*, not a generic waste-sorting demo. Its contribution is a **calibrated,
> uncertainty-aware recommendation**: when the model is not confident it abstains and asks
> for manual verification instead of giving a potentially wrong disposal instruction. The
> decision is accompanied by Grad-CAM evidence and an explicit accuracy–latency trade-off,
> both of which are evaluated rather than merely displayed.

---

## 1. Problem and motivation

Incorrect segregation contaminates recyclable streams and increases the cost of manual
sorting at material-recovery facilities. Most people can name "plastic" or "paper" but are
uncertain about specific items (a crisp packet, a greasy pizza box, a chipped mug). A
phone camera or a bin-side camera can remove that uncertainty — **provided the system knows
when it does not know**. A confident wrong answer is worse than no answer.

## 2. What the system does

```
Uploaded image → Deep CNN / MobileNetV2 → category probabilities
   → temperature-scaled confidence → (confidence ≥ τ ? bin recommendation : "verify manually")
   → Grad-CAM evidence image
```

| Category | Recommended stream |
|---|---|
| cardboard / paper | Paper & cardboard recycling (keep dry) |
| glass | Glass recycling (not ceramics) |
| metal | Metal cans recycling |
| plastic | Dry recyclables / plastic stream |
| trash | Residual / general waste |

The full mapping and advice strings live in `src/data_pipeline.py:BIN_MAP` and
`experiments/results/summary.json`.

## 3. Repository layout

| Path | Contents |
|---|---|
| `src/data_pipeline.py` | download, stratified split, tf.data pipelines, bin mapping |
| `src/models.py` | custom CNN, MobileNetV2 transfer learning, logit access for calibration |
| `src/train.py` | training loop, early stopping, LR schedule, checkpoints |
| `src/evaluate.py` | metrics, temperature scaling, ECE, abstention, plots |
| `src/explain.py` | Grad-CAM heatmaps for evidence and failure analysis |
| `src/app.py` + `src/templates/` | Flask demo |
| `run_all.py` | single command reproducing all experiments and artefacts |
| `notebooks/01_eda.ipynb` | dataset inspection |
| `notebooks/02_train_all.ipynb` | Colab one-click training |
| `experiments/` | logs, model checkpoints, results JSON, figures |
| `report/` | LaTeX report and bibliography |
| `ppt/` | 7-slide presentation (Markdown + generator for `.pptx`) |
| `weekly_updates/` | the 8 required weekly update templates |
| `viva/` | viva defence sheet and demo script |

## 4. Setup and reproduction

### Option A — local (Python 3.13)

```powershell
py -3.13 -m pip install -r requirements.txt
py -3.13 run_all.py
py -3.13 src/app.py        # open http://127.0.0.1:5000
```

### Option B — Google Colab (recommended for a GPU)

1. Zip this folder and upload it to Colab, or clone your private repository.
2. Open `notebooks/02_train_all.ipynb`, set runtime to **T4 GPU**, then **Runtime → Run all**.
3. Download `experiments/` (or keep everything in the repo).

All randomness is fixed with seed 42. `run_all.py --help` shows knobs for image size,
batch size and epoch counts. Defaults are 12/12/6 epochs; the CPU run takes roughly 20
minutes. If training has already completed and only the analysis artefacts need to be
rebuilt (for example after a code fix), use `py -3.13 run_all.py --finalize-only`, which
reloads the saved best checkpoints instead of retraining. The Colab notebook uses a GPU.

## 5. Experimental design

| ID | Model | Question answered |
|---|---|---|
| E1 | Custom CNN, no augmentation | Baseline: can a simple model work at all? |
| E2 | Custom CNN + augmentation | Does augmentation improve generalisation? |
| E3 | MobileNetV2, frozen ImageNet features | Does transfer learning beat the baseline? |
| E4 | MobileNetV2 + augmentation | Controlled augmentation ablation for the advanced model |
| E5 | E4 with last block fine-tuned (lr 1e-4) | Does careful fine-tuning add value? |

The final deployable configuration is the validation-selected E3 checkpoint with
temperature scaling and abstention applied; it is not a sixth training experiment.

Selection uses **validation accuracy only**. The held-out test set is evaluated once at
the end. Reported metrics: accuracy, macro/weighted precision, recall, F1, confusion matrix,
ECE, risk–coverage, latency per image, parameter count and model size. A cross-dataset
robustness evaluation on RealWaste and a Grad-CAM failure analysis are included in the report.

### Compiling the LaTeX report
Upload the whole `waste-project` folder to [Overleaf](https://www.overleaf.com) as a ZIP
(or use a local TeX distribution) and compile `report/report.tex`. The file
`report/tables.tex` is generated from the experiment results, so the report always quotes
the actual numbers. Figures are read from `experiments/figures/`.

## 6. Results

Actual numbers are generated by `run_all.py` into `experiments/results/summary.json` and
rendered into `report/tables.tex` and the slides by `ppt/make_slides.py`. Do not invent
numbers for the viva — quote the values from those files.

## 7. Demo

`src/app.py` serves a web page where a user uploads an image (or clicks a sample) and sees:
predicted category, calibrated confidence, recommended bin **or** an uncertainty warning,
top-3 class probabilities, and a Grad-CAM overlay. Run it after training so
`experiments/models/final_model.keras` and `experiments/results/summary.json` exist.

## 8. Licensing and data ethics

Only public academic datasets are used (TrashNet MIT; RealWaste CC BY-NC-SA 4.0). See
`data/DATASET_CARD.md` for citations. No personal or institutional data is committed.
Raw images and model checkpoints are ignored by Git; download instructions are documented.

## 9. AI-assistance disclosure (required by the course guidelines)

AI coding assistants were used to help draft code, presentation content and report text.
All authors reviewed, executed and can explain the submitted code and results. The scientific
claims come from the experiments generated in this repository.
