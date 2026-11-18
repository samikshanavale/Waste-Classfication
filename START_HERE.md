# START HERE

Everything is in this folder. Read this file first.

## 1. What has been built

- `run_all.py` downloads TrashNet, creates a 70/15/15 split, trains E1–E5, calibrates the
  final model, evaluates once on the test set, checks robustness on RealWaste, and writes
  every figure and `experiments/results/summary.json`.
- `src/app.py` is the live Flask demo (upload → category → confidence → bin or "verify
  manually" → Grad-CAM).
- `ppt/slides.pptx` is the 7-slide viva deck; `ppt/slides.md` contains the same content
  with speaking notes.
- `report/report.tex` + `report/references.bib` are the LaTeX report. Compile on Overleaf.
- `viva/cheat_sheet.md` is your defence script.
- `weekly_updates/` contains the eight required weekly update templates.

## 2. Commands (Windows, Python 3.13)

```powershell
cd "C:\SEM 7\DL theory\Mini Project\waste-project"
py -3.13 -m pip install -r requirements.txt
py -3.13 run_all.py                 # training + all results (already run here)
py -3.13 run_all.py --finalize-only # rebuild results from saved checkpoints, no retraining
py -3.13 postprocess.py             # refresh robustness, tables and slides
py -3.13 src/app.py                 # live demo at http://127.0.0.1:5000
```

For a GPU and more epochs, use `notebooks/02_train_all.ipynb` in Google Colab.

## 3. Before the viva

1. Open `experiments/results/summary.json` — every number you quote is in there.
2. Open the demo and test a normal item and an ambiguous one.
3. Read `viva/cheat_sheet.md` once, then put it away.
4. Fill the `[FILL]` placeholders in `weekly_updates/` and the 4th member's name in
   `README.md` / `report/report.tex` / `ppt/make_slides.py` (then rerun
   `py -3.13 ppt/make_slides.py`), and add your guide's name.
5. Compile `report/report.tex` on Overleaf (upload the whole folder as a ZIP).
6. Publish to your **private** repository:

```powershell
git add -A
git commit -m "Fill group details and weekly evidence"
git tag -f v1.0-final
git remote add origin https://github.com/<your-id>/<private-repo>.git
git push -u origin main
git push -f origin v1.0-final
```

The local repository is already initialised, committed and tagged `v1.0-final`; only the
remote URL and your group-specific details are missing.
