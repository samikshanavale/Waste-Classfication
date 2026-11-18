"""Build the 7-slide viva presentation from the generated results.

    py -3.13 ppt/make_slides.py

Writes ppt/slides.pptx and ppt/slides.md (with speaker notes).
"""
from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "experiments" / "results" / "summary.json"
OUT = ROOT / "ppt"

GREEN = RGBColor(23, 107, 69)
DARK = RGBColor(24, 51, 42)
GREY = RGBColor(90, 110, 100)


def load_summary() -> dict:
    if SUMMARY.exists():
        return json.loads(SUMMARY.read_text(encoding="utf-8"))
    return {"experiments": [], "selected_model": {"name": "pending"},
            "test": {"temperature_scaled": {}, "ece_before": None, "ece_after": None,
                     "abstention_on_test": {}, "latency_ms": None, "complexity": {}},
            "dataset": {"classes": []}, "settings": {}}


def fmt(value, digits=3):
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "—"


def add_slide(prs, title: str, subtitle: str = ""):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = RGBColor(244, 247, 245)
    box = slide.shapes.add_textbox(Inches(0.6), Inches(0.35), Inches(12), Inches(0.9))
    tf = box.text_frame
    tf.text = title
    tf.paragraphs[0].runs[0].font.size = Pt(30)
    tf.paragraphs[0].runs[0].font.bold = True
    tf.paragraphs[0].runs[0].font.color.rgb = GREEN
    if subtitle:
        sub = slide.shapes.add_textbox(Inches(0.62), Inches(1.12), Inches(12), Inches(0.45))
        sub.text_frame.text = subtitle
        sub.text_frame.paragraphs[0].runs[0].font.size = Pt(14)
        sub.text_frame.paragraphs[0].runs[0].font.color.rgb = GREY
    return slide


def add_bullets(slide, lines, left=0.7, top=1.75, width=11.8, height=4.8, size=18):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.level = 0
        p.space_after = Pt(10)
        p.font.size = Pt(size)
        p.font.color.rgb = DARK
    return box


def add_notes(slide, text: str):
    slide.notes_slide.notes_text_frame.text = text


def add_table(slide, headers, rows, left=0.7, top=2.0, width=11.8, height=3.6):
    shape = slide.shapes.add_table(len(rows) + 1, len(headers),
                                   Inches(left), Inches(top), Inches(width), Inches(height))
    table = shape.table
    for j, header in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = header
        for p in cell.text_frame.paragraphs:
            for r in p.runs:
                r.font.bold = True
                r.font.size = Pt(12)
                r.font.color.rgb = RGBColor(255, 255, 255)
        cell.fill.solid()
        cell.fill.fore_color.rgb = GREEN
    for i, row in enumerate(rows, start=1):
        for j, value in enumerate(row):
            cell = table.cell(i, j)
            cell.text = str(value)
            for p in cell.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(11)
    return table


def build():
    data = load_summary()
    exps = data.get("experiments", [])
    test = data.get("test", {})
    scaled = test.get("temperature_scaled", {})
    abst = test.get("abstention_on_test", {})
    selected = data.get("selected_model", {}).get("name", "—")
    complexity = test.get("complexity", {})

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)

    # ---------------------------------------------------------------- slide 1
    s = add_slide(prs, "Image-Based Waste Classification\nand Smart Bin Recommendation System",
                  "Deep Learning (R5IT4109L) — Advanced Mini Project")
    add_bullets(s, [
        "Team: Om Walunj (241081901), Samiksha Navale (241081903),",
        "Harshada Khanjode (241081909) and group member 4",
        "Guide: ____________________",
        "Problem → model → calibrated confidence → bin recommendation",
    ], top=2.1, size=20)
    add_notes(s, "Open with the real problem: people mis-sort waste because they are unsure "
                 "which bin an item belongs to. Our system answers that question and, crucially, "
                 "knows when to abstain. Mention the approved title exactly as written.")

    # ---------------------------------------------------------------- slide 2
    s = add_slide(prs, "1. Problem & Real-Life Scenario",
                  "Why this matters outside the classroom")
    add_bullets(s, [
        "Contaminated recyclable streams increase manual sorting cost and reduce recycling yield.",
        "Users recognise a material but are unsure about the correct disposal stream.",
        "Real scenario: a hostel/canteen disposal point or municipal collection app.",
        "A resident photographs an item → the system names the material and the correct bin.",
        "Key requirement: a wrong, overconfident instruction is worse than saying “please check”.",
        "Therefore confidence estimation and abstention are part of the problem, not decoration.",
    ])
    add_notes(s, "Tell the story of a used bottle vs a greasy pizza box. The first is an easy "
                 "plastic/recyclable case; the second is ambiguous and should be flagged. "
                 "This motivates the uncertainty component.")

    # ---------------------------------------------------------------- slide 3
    s = add_slide(prs, "2. Abstract & End-to-End Pipeline", "One sentence objective, one clear workflow")
    add_bullets(s, [
        "Objective: classify a waste image into a material category and recommend the correct "
        "disposal bin, abstaining when uncertain.",
        "",
        "Image upload → preprocessing & augmentation → deep model (CNN / MobileNetV2)",
        "→ class probabilities → temperature scaling & confidence check",
        "→ bin recommendation OR manual-verification flag → Grad-CAM evidence.",
        "",
        "Evaluation: accuracy, macro F1, confusion matrix, ECE, risk–coverage, latency.",
    ])
    add_notes(s, "Emphasise that the pipeline is complete: data → formulation → architecture → "
                 "training → validation → testing → analysis → deployment. This is the exact "
                 "chain the course guidelines require.")

    # ---------------------------------------------------------------- slide 4
    s = add_slide(prs, "3. Implementation Done", "Dataset, models, training and web application")
    add_bullets(s, [
        "Data: TrashNet — 2,527 images, 6 classes; stratified 70/15/15 split (1,768 / 379 / 380).",
        "RealWaste (4,752 realistic images) reserved for a cross-dataset robustness test.",
        "Baseline: custom CNN — 3 conv blocks + BN + GAP + dropout (trained with/without augmentation).",
        "Advanced: MobileNetV2 transfer learning (frozen features, then last block fine-tuned at lr 1e-4).",
        "Calibration: temperature fitted on validation logits; threshold chosen on validation risk–coverage.",
        "Application: Flask web app — upload → category → calibrated confidence → bin advice + Grad-CAM.",
    ], size=17)
    add_notes(s, "Show the repository structure if asked. Point out the logits layer is kept "
                 "explicit so calibration is possible; a plain softmax-only model could not be calibrated.")

    # ---------------------------------------------------------------- slide 5
    s = add_slide(prs, "4. New Innovation in This Project", "Three additions driven by real requirements")
    add_bullets(s, [
        "1. Uncertainty-aware bin recommendation: temperature-scaled confidence + abstention. "
        "Low-confidence images are routed to manual verification instead of a wrong bin. "
        f"(ECE {fmt(test.get('ece_before'))} → {fmt(test.get('ece_after'))}.)",
        "2. Explainability for trust: Grad-CAM highlights the region that drove a decision and "
        "is used to inspect the dominant error pairs measured in the confusion matrix — not "
        "just shown as a picture.",
        "3. Deployment realism: latency, parameter count and model size are measured for each "
        f"candidate (final model: {fmt(test.get('latency_ms'), 1)} ms/image, "
        f"{complexity.get('params_total', '—')} params), justifying the lightweight architecture.",
        "Together these turn a classifier into a responsible decision-support system.",
    ], size=17)
    add_notes(s, "Be honest: the innovation is applied/system-level, not a new learning algorithm. "
                 "The novelty is that uncertainty, explanation and efficiency are each justified "
                 "by a concrete failure mode of waste disposal and are evaluated with evidence.")

    # ---------------------------------------------------------------- slide 6
    rows = []
    for exp in exps:
        rows.append((exp["id"], exp["name"].replace("_", " "), fmt(exp["val_accuracy"]),
                     fmt(exp["val_f1_macro"]), exp.get("params", "—"), fmt(exp["latency_ms"], 1)))
    s = add_slide(prs, "5. Results & Failure Analysis",
                  f"Selected model: {selected} (chosen on validation accuracy)")
    if rows:
        add_table(s, ["ID", "Model", "Val acc", "Val F1", "Params", "ms/img"], rows,
                  top=1.75, height=min(3.2, 0.42 * (len(rows) + 1)))
    y = 5.15 if rows else 1.9
    top_confusions = test.get("top_confusions", [])
    confusion_text = ", ".join(f"{a}→{b} ({n})" for a, b, n in top_confusions) \
        if top_confusions else "see confusion matrix"
    add_bullets(s, [
        f"Test accuracy {fmt(scaled.get('accuracy'))}, macro F1 {fmt(scaled.get('f1_macro'))}; "
        f"ECE {fmt(test.get('ece_before'))} → {fmt(test.get('ece_after'))} after calibration.",
        f"Abstention at τ={fmt(abst.get('threshold'),2)}: {fmt(abst.get('coverage'))} coverage, "
        f"{fmt(abst.get('accuracy_on_answered'))} accuracy on answered images.",
        f"Largest confusions (true→pred, count): {confusion_text}.",
    ], top=y, height=1.8, size=14)
    fig = ROOT / "experiments" / "figures" / "confusion_matrix_final.png"
    if fig.exists():
        s.shapes.add_picture(str(fig), Inches(9.2), Inches(1.75), height=Inches(2.9))
    add_notes(s, "Never quote a number that is not in results/summary.json. Explain the "
                 "abstention trade-off: coverage falls, but accuracy on answered items rises. "
                 "Quote the top confusion pairs printed on the slide; they come straight from "
                  "the test confusion matrix.")

    # ---------------------------------------------------------------- slide 7
    s = add_slide(prs, "6. Live Demo & Real-Life Impact", "Working prototype you can try now")
    add_bullets(s, [
        "Demo flow: upload an image (or click a sample) → prediction + confidence → bin advice "
        "or uncertainty warning → Grad-CAM evidence.",
        "Live use: bin-side camera or resident's phone at a canteen/hostel waste station.",
        "Example: a plastic bottle → “Plastic → Dry recyclable bin”; a crumpled, contaminated "
        "item → “Uncertain — please verify manually”.",
        "Limitations: single-object clean-background training data, small residual class, "
        "closed-set categories; RealWaste robustness test quantifies the drop.",
        "Future work: multi-object detection, active learning on flagged images, on-device "
        "quantised deployment.",
    ], size=17)
    add_notes(s, "Run src/app.py live. If the demo misclassifies an item, use it as evidence "
                 "for the abstention mechanism rather than hiding it. Close with the topic-lock "
                 "and distinctiveness statement if asked.")

    prs.save(OUT / "slides.pptx")
    write_markdown(data)
    print(f"Wrote {OUT / 'slides.pptx'} and {OUT / 'slides.md'}")


def write_markdown(data: dict):
    test = data.get("test", {})
    scaled = test.get("temperature_scaled", {})
    abst = test.get("abstention_on_test", {})
    lines = [
        "# 7-slide viva presentation (speaker notes included)", "",
        "## Slide 1 — Title", data.get("selected_model", {}).get("name", "—"), "",
        "## Slide 2 — Problem & real-life scenario",
        "- Contamination, user uncertainty, hostel/canteen example, need to abstain.", "",
        "## Slide 3 — Abstract & pipeline",
        "- Objective and full data → model → confidence → bin workflow.", "",
        "## Slide 4 — Implementation",
        "- TrashNet 70/15/15, RealWaste robustness, CNN baseline, MobileNetV2, calibration, Flask.", "",
        "## Slide 5 — New innovation",
        "- Uncertainty + abstention, Grad-CAM evidence, measured efficiency.", "",
        "## Slide 6 — Results",
        f"- Test accuracy: {fmt(scaled.get('accuracy'))}; macro F1: {fmt(scaled.get('f1_macro'))}; "
        f"ECE: {fmt(test.get('ece_before'))} → {fmt(test.get('ece_after'))}.",
        f"- Abstention at τ={fmt(abst.get('threshold'), 2)}: coverage {fmt(abst.get('coverage'))}, "
        f"accuracy on answered {fmt(abst.get('accuracy_on_answered'))}, "
        f"flagged {abst.get('flagged_for_review', '—')}.", "",
        "## Slide 7 — Demo & impact",
        "- Live Flask demo, real-life scenario, limitations, future work.", "",
        "See `slides.pptx` for the formatted deck and `viva/cheat_sheet.md` for the defence script.",
    ]
    (OUT / "slides.md").write_text("\n".join(str(x) for x in lines), encoding="utf-8")


if __name__ == "__main__":
    build()
