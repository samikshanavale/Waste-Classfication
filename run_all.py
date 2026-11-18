"""Reproduce every experiment and generate all artefacts used by the demo/report.

Local quick run:      py -3.13 run_all.py
Google Colab run:     open notebooks/02_train_all.ipynb and Run all

The test set is kept untouched until the final evaluation; the best model is
selected on validation accuracy only.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from data_pipeline import CLASS_NAMES, BIN_MAP, prepare_all  # noqa: E402
from models import build_custom_cnn, build_mobilenetv2, compile_model  # noqa: E402
from train import train_model  # noqa: E402
from evaluate import (  # noqa: E402
    abstention_curve, apply_temperature, classification_metrics,
    confusion_matrix, expected_calibration_error, fit_temperature,
    logits_array, measure_latency_ms, model_complexity, per_class_report,
    plot_calibration, plot_confusion_matrix, plot_history, plot_risk_coverage,
    predict_arrays, save_json,
)
from explain import build_explanation_grid  # noqa: E402
from robustness import download_subset, evaluate_robustness  # noqa: E402
from tensorflow import keras  # noqa: E402

# Human-readable experiment labels. Checkpoint-internal model names are not
# informative here (E4 and E5 both load as "mobilenetv2_frozen_aug"), so the
# result tables use these explicit descriptions of what each run actually did.
RUN_LABELS = {
    "E1": "Custom CNN (baseline)",
    "E2": "Custom CNN + augmentation",
    "E3": "MobileNetV2 transfer (frozen)",
    "E4": "MobileNetV2 transfer + augmentation",
    "E5": "MobileNetV2 fine-tuned (last block)",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs-cnn", type=int, default=12)
    parser.add_argument("--epochs-mt", type=int, default=12)
    parser.add_argument("--epochs-finetune", type=int, default=6)
    parser.add_argument("--skip-cnn", action="store_true")
    parser.add_argument("--finalize-only", action="store_true",
                        help="Skip training and rebuild results/figures from the saved "
                             "best checkpoints in experiments/models/.")
    args = parser.parse_args()

    results_dir = ROOT / "experiments" / "results"
    figures_dir = ROOT / "experiments" / "figures"
    models_dir = ROOT / "experiments" / "models"
    for directory in (results_dir, figures_dir, models_dir):
        directory.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds = prepare_all(
        raw_dir=ROOT / "data" / "raw", processed_dir=ROOT / "data" / "processed",
        img_size=args.img_size, batch_size=args.batch_size)

    experiments = []

    def record(run_id, model, history, model_path):
        val_probs, val_y = predict_arrays(model, val_ds)
        metrics = classification_metrics(val_y, val_probs.argmax(axis=1))
        complexity = model_complexity(model, model_path)
        experiments.append({
            "id": run_id, "name": RUN_LABELS.get(run_id, model.name),
            "val_accuracy": round(metrics["accuracy"], 4),
            "val_f1_macro": round(metrics["f1_macro"], 4),
            "params": complexity["params_total"],
            "size_mb": complexity["size_mb"],
            "latency_ms": measure_latency_ms(model, test_ds),
            "model_path": str(model_path.relative_to(ROOT)),
        })
        plot_history(history, figures_dir / f"curves_{run_id}.png")
        print(f"  {run_id}: val accuracy={metrics['accuracy']:.4f}, "
              f"F1={metrics['f1_macro']:.4f}, latency={experiments[-1]['latency_ms']} ms")

    if args.finalize_only:
        # Rebuild the experiment table from the saved best checkpoints.  This is
        # used when training has already completed (e.g. the Grad-CAM fix was
        # applied after the run) so we never silently retrain or change numbers.
        for run_id in ("E1", "E2", "E3", "E4", "E5"):
            path = models_dir / f"{run_id}.keras"
            if not path.exists():
                raise FileNotFoundError(f"Missing checkpoint {path}; run training first.")
            model = keras.models.load_model(path)
            val_probs, val_y = predict_arrays(model, val_ds)
            metrics = classification_metrics(val_y, val_probs.argmax(axis=1))
            complexity = model_complexity(model, path)
            row = {
                "id": run_id, "name": RUN_LABELS.get(run_id, model.name),
                "val_accuracy": round(metrics["accuracy"], 4),
                "val_f1_macro": round(metrics["f1_macro"], 4),
                "params": complexity["params_total"],
                "size_mb": complexity["size_mb"],
                "latency_ms": measure_latency_ms(model, test_ds),
                "model_path": str(path.relative_to(ROOT)),
            }
            experiments.append(row)
            print(f"  {run_id}: val accuracy={row['val_accuracy']:.4f}, "
                  f"F1={row['val_f1_macro']:.4f}, latency={row['latency_ms']} ms")
    else:
        # ---------------- E1 / E2: custom CNN baseline, with and without augmentation
        if not args.skip_cnn:
            for run_id, augment in (("E1", False), ("E2", True)):
                model = build_custom_cnn(len(CLASS_NAMES), args.img_size, augmentation=augment)
                history, model = train_model(model, train_ds, val_ds, args.epochs_cnn, 1e-3,
                                             run_id, ROOT / "experiments")
                record(run_id, model, history, models_dir / f"{run_id}.keras")

        # ---------------- E3 / E4: MobileNetV2 transfer learning
        frozen_path = None
        for run_id, augment in (("E3", False), ("E4", True)):
            model = build_mobilenetv2(len(CLASS_NAMES), args.img_size, augmentation=augment)
            history, model = train_model(model, train_ds, val_ds, args.epochs_mt, 1e-3,
                                         run_id, ROOT / "experiments")
            path = models_dir / f"{run_id}.keras"
            record(run_id, model, history, path)
            if run_id == "E4":
                frozen_path = path

        # ---------------- E5: fine-tune the last block of the best frozen model
        if frozen_path and args.epochs_finetune > 0:
            model = keras.models.load_model(frozen_path)
            base = next(layer for layer in model.layers if isinstance(layer, keras.Model)
                        and "mobilenet" in layer.name.lower())
            base.trainable = True
            for layer in base.layers[:-30]:
                layer.trainable = False
            for layer in base.layers[-30:]:
                if isinstance(layer, keras.layers.BatchNormalization):
                    layer.trainable = False
            history, model = train_model(model, train_ds, val_ds, args.epochs_finetune, 1e-4,
                                         "E5", ROOT / "experiments")
            record("E5", model, history, models_dir / "E5.keras")

    # ---------------- select on validation only, then touch the test set once
    best = max(experiments, key=lambda r: r["val_accuracy"])
    final_model = keras.models.load_model(ROOT / best["model_path"])
    final_path = models_dir / "final_model.keras"
    shutil.copy2(ROOT / best["model_path"], final_path)

    val_probs, val_y = predict_arrays(final_model, val_ds)
    val_logits = logits_array(final_model, val_ds)
    temperature = fit_temperature(val_logits, val_y)
    abstention = abstention_curve(apply_temperature(val_logits, temperature), val_y)

    test_probs_raw, test_y = predict_arrays(final_model, test_ds)
    test_logits = logits_array(final_model, test_ds)
    test_probs = apply_temperature(test_logits, temperature)

    raw_metrics = classification_metrics(test_y, test_probs_raw.argmax(axis=1))
    final_metrics = classification_metrics(test_y, test_probs.argmax(axis=1))
    cm = confusion_matrix(test_y, test_probs.argmax(axis=1))
    # Ordered error pairs straight from the confusion matrix, so the failure
    # analysis in the report and slides is evidence rather than an assumption.
    off_diagonal = [(CLASS_NAMES[i], CLASS_NAMES[j], int(cm[i, j]))
                    for i in range(len(CLASS_NAMES))
                    for j in range(len(CLASS_NAMES)) if i != j]
    top_confusions = sorted(off_diagonal, key=lambda row: row[2], reverse=True)[:3]
    ece_before = expected_calibration_error(test_probs_raw, test_y)
    ece_after = expected_calibration_error(test_probs, test_y)

    threshold = abstention["chosen_threshold"]
    keep = test_probs.max(axis=1) >= threshold
    abst_test = {
        "threshold": threshold,
        "coverage": round(float(keep.mean()), 4),
        "accuracy_on_answered": round(float((test_probs.argmax(axis=1)[keep] == test_y[keep]).mean()), 4),
        "flagged_for_review": int((~keep).sum()),
    }

    plot_confusion_matrix(cm, CLASS_NAMES, figures_dir / "confusion_matrix_final.png")
    plot_calibration(test_probs_raw, test_probs, test_y, figures_dir / "calibration.png")
    plot_risk_coverage(abstention, figures_dir / "risk_coverage.png")
    n_examples = build_explanation_grid(final_model, test_ds, CLASS_NAMES,
                                        figures_dir / "gradcam_examples.png", args.img_size)

    per_class = per_class_report(test_y, test_probs.argmax(axis=1), CLASS_NAMES)
    print("\nEvaluating cross-dataset robustness on a RealWaste subset ...")
    robustness_dir = ROOT / "data" / "robustness"
    download_subset(robustness_dir, per_class=40)
    robustness = evaluate_robustness(
        final_model, robustness_dir, args.img_size, args.batch_size, temperature,
        figures_dir / "robustness_comparison.png", per_class)
    print(f"  RealWaste accuracy: {robustness['accuracy']:.4f} "
          f"(in-distribution: {final_metrics['accuracy']:.4f})")

    summary = {
        "dataset": {"name": "TrashNet (resized)", "classes": CLASS_NAMES,
                    "images": 2527, "split": "70/15/15 stratified, seed 42",
                    "robustness_set": "RealWaste (UCI, 4752 images, 9 classes)"},
        "experiments": experiments,
        "selected_model": {"id": best["id"],
                           "name": RUN_LABELS.get(best["id"], final_model.name),
                           "checkpoint_name": final_model.name,
                           "selection_criterion": "validation accuracy"},
        "temperature": temperature,
        "test": {"raw_softmax": raw_metrics, "temperature_scaled": final_metrics,
                 "ece_before": round(ece_before, 4), "ece_after": round(ece_after, 4),
                 "per_class": per_class, "top_confusions": top_confusions,
                 "abstention": abstention, "abstention_on_test": abst_test,
                 "latency_ms": measure_latency_ms(final_model, test_ds),
                 "complexity": model_complexity(final_model, final_path)},
        "robustness": robustness,
        "bin_mapping": {k: {"bin": v[0], "advice": v[1]} for k, v in BIN_MAP.items()},
        "gradcam_examples": n_examples,
        "settings": {"img_size": args.img_size, "batch_size": args.batch_size,
                     "epochs_cnn": args.epochs_cnn, "epochs_mt": args.epochs_mt,
                     "epochs_finetune": args.epochs_finetune},
    }
    save_json(summary, results_dir / "summary.json")
    save_json({"experiments": experiments}, results_dir / "experiment_table.json")

    # Sample images for the Flask demo (one per class, from the untouched test set)
    sample_dir = ROOT / "src" / "static" / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for class_name in CLASS_NAMES:
        candidates = sorted((ROOT / "data" / "processed" / "test" / class_name).glob("*"))
        if candidates:
            shutil.copy2(candidates[0], sample_dir / f"{class_name}.jpg")

    # Keep the report and slides in lockstep with the regenerated summary.
    for script in ("report/make_tables.py", "ppt/make_slides.py"):
        subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, check=True)

    print("\n=== Final results (untouched test set) ===")
    print(json.dumps(summary["test"], indent=2))
    print(f"\nArtefacts written to {results_dir} and {figures_dir}")


if __name__ == "__main__":
    main()
