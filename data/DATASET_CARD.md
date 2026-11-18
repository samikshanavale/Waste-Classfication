# Dataset card

## Primary training dataset — TrashNet (resized)

- **Source:** Gary Thung & Mindy Yang, Stanford CS229 (2016).
  <https://github.com/garythung/trashnet> · download used by `src/data_pipeline.py`:
  <https://raw.githubusercontent.com/garythung/trashnet/master/data/dataset-resized.zip>
- **Licence:** MIT (repository licence). Academic use permitted; cite the repository.
- **Size:** 2,527 images, 512×384 pixels, 6 classes.
  `cardboard` 403 · `glass` 501 · `metal` 410 · `paper` 594 · `plastic` 482 · `trash` 137.
- **Acquisition:** objects photographed on a white posterboard with sunlight/room light using
  consumer phones. This is a *clean, controlled* setting.
- **Split used here:** stratified 70/15/15 with seed 42 → 1,768 / 379 / 380 images.
  The test split is sealed until final evaluation, and held out from all tuning.
- **Role in this project:** training, validation and held-out evaluation; matches the five
  recyclable categories in the proposal plus a residual `trash` class.

**Citation**

```bibtex
@misc{yang2016trashnet,
  author       = {Yang, Mindy and Thung, Gary},
  title        = {Classification of trash for recyclability status},
  year         = {2016},
  howpublished = {Stanford CS229 Project Report},
  url          = {https://github.com/garythung/trashnet}
}
```

## Robustness dataset — RealWaste

- **Source:** Sam Single, Saeid Iranmanesh, Raad Raad (University of Wollongong, 2023).
  UCI ML Repository dataset 908: <https://archive.ics.uci.edu/dataset/908/realwaste>
- **Licence:** CC BY-NC-SA 4.0 (non-commercial, attribution, share-alike).
- **Size:** 4,752 images, 9 material classes, captured at a landfill reception facility;
  realistic illumination, contamination and overlapping materials.
- **Role in this project:** cross-dataset robustness test — images the model has never
  seen from a different visual distribution. Reported as a separate experiment, never mixed
  into the training split.
- **Practical caution:** do not commit the raw images to the repository; record the download
  procedure and licence instead.

**Citation**

```bibtex
@article{single2023realwaste,
  author  = {Single, Sam and Iranmanesh, Saeid and Raad, Raad},
  title   = {RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification Using Deep Learning},
  journal = {Information},
  volume  = {14}, number = {12}, pages = {633}, year = {2023},
  doi     = {10.3390/info14120633}
}
```

## Known limitations relevant to the project

- TrashNet's uniform white background makes high in-distribution accuracy optimistic;
  the RealWaste test is the honest measure of practical robustness.
- `trash` (residual) is under-represented (137 images), so its per-class recall is
  expected to be weaker — this is analysed explicitly in the failure section.
- No personally identifiable information is present in either dataset.
