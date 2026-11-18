"""Add the cross-dataset robustness result and regenerate report/slides artefacts.

Used after run_all.py has produced the final model and summary.json. This script
also makes it possible to re-render the report and presentation without retraining.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from data_pipeline import CLASS_NAMES, make_datasets  # noqa: E402
from evaluate import (abstention_curve, apply_temperature, classification_metrics,  # noqa: E402
                      expected_calibration_error, fit_temperature, logits_array,
                      predict_arrays)
from robustness import download_subset, evaluate_robustness  # noqa: E402
from tensorflow import keras  # noqa: E402


def main():
    summary_path = ROOT / "experiments" / "results" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    model = keras.models.load_model(ROOT / "experiments" / "models" / "final_model.keras")

    # Recompute calibration/abstention with the latest analysis code so the
    # generated report and slides always reflect the improved threshold rule.
    img_size = int(summary.get("settings", {}).get("img_size", 160))
    batch_size = int(summary.get("settings", {}).get("batch_size", 32))
    _, val_ds, test_ds = make_datasets(ROOT / "data" / "processed", img_size, batch_size)
    val_logits = logits_array(model, val_ds)
    _, val_y = predict_arrays(model, val_ds)
    temperature = fit_temperature(val_logits, val_y)
    abstention = abstention_curve(apply_temperature(val_logits, temperature), val_y)

    test_logits = logits_array(model, test_ds)
    test_probs_raw, test_y = predict_arrays(model, test_ds)
    test_probs = apply_temperature(test_logits, temperature)
    threshold = abstention["chosen_threshold"]
    keep = test_probs.max(axis=1) >= threshold
    abst_test = {
        "threshold": threshold,
        "coverage": round(float(keep.mean()), 4),
        "accuracy_on_answered": round(float((test_probs.argmax(axis=1)[keep] == test_y[keep]).mean()), 4),
        "flagged_for_review": int((~keep).sum()),
    }
    summary["temperature"] = temperature
    summary["test"]["temperature_scaled"] = classification_metrics(test_y, test_probs.argmax(axis=1))
    summary["test"]["ece_before"] = round(expected_calibration_error(test_probs_raw, test_y), 4)
    summary["test"]["ece_after"] = round(expected_calibration_error(test_probs, test_y), 4)
    summary["test"]["abstention"] = abstention
    summary["test"]["abstention_on_test"] = abst_test
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    robustness_dir = ROOT / "data" / "robustness"
    print("Evaluating cross-dataset robustness on RealWaste subset ...")
    download_subset(robustness_dir, per_class=40)
    robustness = evaluate_robustness(
        model, robustness_dir,
        summary.get("settings", {}).get("img_size", 160),
        summary.get("settings", {}).get("batch_size", 32),
        float(summary.get("temperature", 1.0)),
        ROOT / "experiments" / "figures" / "robustness_comparison.png",
        summary.get("test", {}).get("per_class"),
    )
    summary["robustness"] = robustness
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  RealWaste accuracy: {robustness['accuracy']:.4f}")

    for script in ("report/make_tables.py", "ppt/make_slides.py"):
        subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
