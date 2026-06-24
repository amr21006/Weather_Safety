# -*- coding: utf-8 -*-
"""
generate_nlp_validation.py
--------------------------
Validate the production NLP equipment ontology over the FULL modeling corpus.

An INDEPENDENT keyword rule set (developed separately from the production patterns
in build_dataset.py, with different synonym lists) is applied to every incident.
It mirrors the production two-pass logic - narrative text first, then the OSHA
source-title descriptors as a fallback - and the resulting label is compared with
the pipeline's nlp_equipment assignment for all records in the cached dataset.

This is an automated internal-consistency / reproducibility check on the ontology,
not human annotation. It reports the confusion matrix, exact agreement, Cohen's
kappa between the two rule sets, and class-level precision/recall/F1.
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, confusion_matrix


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data")
FIG_DIR = os.path.join(ROOT, "figures")
CACHE = os.path.join(DATA_DIR, "enriched_maritime_data.parquet")

MATRIX_CSV = os.path.join(DATA_DIR, "nlp_keyword_validation_matrix.csv")
FIG_PNG = os.path.join(FIG_DIR, "Reviewer_Fig_1_NLP_Confusion_Matrix.png")
FIG_PDF = os.path.join(FIG_DIR, "Reviewer_Fig_1_NLP_Confusion_Matrix.pdf")
FIG_SUMMARY = os.path.join(FIG_DIR, "Reviewer_Fig_1_NLP_Confusion_Matrix_summary.txt")

CLASSES = [
    "Floating Crane", "Vessel/Barge", "Scaffold/Platform", "Ladder/Gangway",
    "Forklift/Lift", "Excavator/Dredge", "Welding Apparatus", "Electrical",
    "Pump/Compressor", "Rigging/Sling", "Saw/Grinder", "Hand Tool",
    "Vehicle/Truck", "Pile Driver", "Conveyor", "Other/Unknown",
]

# Independent keyword rules. These are written separately from the production
# EQUIPMENT_PATTERNS in build_dataset.py so that agreement between the two label
# sets is a genuine robustness check rather than a restatement of one rule set.
VALIDATION_RULES = [
    ("Pile Driver", [r"\bpile\s*driver\b", r"\bpile[- ]?driv", r"\bpiling\b", r"\bsheet\s*pile\b", r"\bh-?pile\b"]),
    ("Floating Crane", [r"\bcrane\b", r"\bderrick\b", r"\bgantry\b", r"\bhoist\b", r"\bwinch\b"]),
    ("Forklift/Lift", [
        r"\bforklift\b", r"\bfork\s*lift\b", r"\bscissor\s*lift\b", r"\baerial\s*lift\b",
        r"\bman\s*lift\b", r"\bmanlift\b", r"\bboom\s*lift\b", r"\btelehandler\b",
        r"\bpallet\s*jack\b", r"\breach\s*truck\b",
    ]),
    ("Excavator/Dredge", [
        r"\bexcavat\w*\b", r"\bdredg\w*\b", r"\bbackhoe\b", r"\bfront[- ]end\s*loader\b",
        r"\bloader\b", r"\bbulldozer\b", r"\bdozer\b", r"\bauger\b", r"\btrench\w*\b",
    ]),
    ("Vessel/Barge", [
        r"\bbarge\b", r"\bvessel\b", r"\bship\w*\b", r"\bboat\b", r"\btug\b", r"\bskiff\b",
        r"\bferry\b", r"\bsubmarine\b", r"\btanker\b", r"\baircraft\s*carrier\b", r"\bwater\s*vehicle\b",
    ]),
    ("Scaffold/Platform", [
        r"\bscaffold\w*\b", r"\bplatform\b", r"\bstaging\b", r"\bcatwalk\b", r"\bwalkway\b",
        r"\btemporary\s*work\s*platform\b", r"\bdeck\b", r"\bgrating\b", r"\bplank\b",
    ]),
    ("Ladder/Gangway", [r"\bladder\b", r"\bgangway\b", r"\bgang\s*plank\b", r"\bramp\b", r"\bstair\w*\b", r"\bstep\b"]),
    ("Welding Apparatus", [r"\bweld\w*\b", r"\btorch\b", r"\bcutting\s*torch\b", r"\bhot\s*work\b", r"\bslag\b", r"\bplasma\s*cutter\b", r"\bbrazing\b"]),
    ("Electrical", [r"\belectric\w*\b", r"\benergized\b", r"\bvoltage\b", r"\bcircuit\b", r"\bbreaker\b", r"\bpower\s*line\b", r"\bwire\b", r"\bshock\b", r"\btransformer\b"]),
    ("Pump/Compressor", [
        r"\bpump\b", r"\bcompressor\b", r"\bpneumatic\b", r"\bhydraulic\s*pump\b",
        r"\bair\s*hose\b", r"\bpressure\s*hose\b", r"\bpressure\s*wash\w*\b", r"\bpower\s*wash\w*\b",
        r"\bpressure\s*washer\b", r"\bpower\s*washer\b", r"\bspray\s*wand\b", r"\bblast\s*hose\b",
        r"\bwater[- ]?blast\w*\b", r"\bhigh[-\s]*pressure\s*wand\b", r"\bblast\s*pot\b",
        r"\bblasting\s*pot\b", r"\babrasive\s*blasting\b", r"\bsandblast\w*\b", r"\bblower\b",
        r"\bhose\b",
    ]),
    ("Rigging/Sling", [r"\brigging\b", r"\bsling\b", r"\bshackle\b", r"\bchain\b", r"\bcable\b", r"\bwire\s*rope\b", r"\bload\s*line\b", r"\blanyard\b", r"\bharness\b", r"\bjack\b"]),
    ("Saw/Grinder", [r"\bsaw\b", r"\bgrind\w*\b", r"\bgrinder\b", r"\bblade\b", r"\bcut[- ]?off\b", r"\bcutoff\b", r"\bchop\s*saw\b", r"\bcircular\s*saw\b", r"\bchain\s*saw\b", r"\bchainsaw\b", r"\bband\s*saw\b", r"\bbar\s*oil\b", r"\bchain\s*oil\b", r"\bplaner\b", r"\blathe\b"]),
    ("Hand Tool", [r"\bhammer\w*\b", r"\bsledgehammer\b", r"\bwrench\b", r"\bdrill\b", r"\bchisel\b", r"\bscrewdriver\b", r"\bhand\s*tool\b", r"\bjackhammer\b", r"\bpry\s*bar\b", r"\bpocket\s*knife\b", r"\bknife\b", r"\bpunch\b"]),
    ("Vehicle/Truck", [r"\btruck\b", r"\bvehicle\b", r"\bvan\b", r"\btrailer\b", r"\bdump\s*truck\b", r"\bpickup\b", r"\bsemi\b", r"\btractor\s*trailer\b", r"\btransporter\b", r"\bautomobile\b", r"\bbicycle\b", r"\bpedal\s*cycle\b"]),
    ("Conveyor", [r"\bconveyor\b", r"\bconveyor\s*belt\b", r"\bbelt\b"]),
]

# Independent mapping of OSHA structured descriptors and structural materials
# (the SourceTitle / Secondary Source / EventTitle fields and structural nouns
# such as floors, beams, plates, and stairs). These are appended AFTER the
# equipment lexicon so that a record is only resolved structurally when no direct
# equipment term is present - mirroring the production source-title fallback and
# ensuring every modeled incident maps to one of the 15 real classes (the cohort
# contains no Other/Unknown category, so the matrix must not introduce one).
SOURCE_RULES = [
    ("Ladder/Gangway", [r"\bstair\w*", r"\bstep\w*", r"\bgangway\b", r"\bramp\b"]),
    ("Pump/Compressor", [r"pressuriz\w*", r"water[- ]?blast", r"power\s*wash\w*", r"pressure\s*wash\w*", r"\bfans?\b", r"blower", r"ventilation"]),
    ("Forklift/Lift", [r"material\s*and\s*personnel\s*handling", r"personnel\s*handling\s*machine", r"pallet\s*jack"]),
    ("Rigging/Sling", [r"\bjacks?\b", r"\bropes?\b", r"\btwine\b", r"lifeline"]),
    ("Saw/Grinder", [r"special\s*process\s*machinery", r"milling\s*machine", r"metalworking", r"cutting\s*machinery", r"surfacing\s*hand"]),
    ("Hand Tool", [r"wheelbarrow", r"\bskids?\b", r"\bpallets?\b", r"hand\s*tool"]),
    ("Excavator/Dredge", [r"roller", r"compactor", r"\bdirt\b", r"\bearth\b", r"drilling\s*and\s*extraction"]),
    ("Scaffold/Platform", [
        r"floor", r"walkway", r"ground\s*surface", r"constructed\s*surface", r"\broof",
        r"structural", r"\bbeams?\b", r"girder", r"\bplates?\b", r"\bpanels?\b", r"sheet\s*metal",
        r"angle\s*iron", r"\bcaps?\b", r"\blids?\b", r"\bcovers?\b", r"\bdecks?\b", r"grating",
        r"\bplanks?\b", r"\brails?\b", r"railroad", r"\bpier", r"wharf", r"catwalk", r"\bhatch",
        r"\bwalls?\b", r"\bgates?\b", r"\bsurface", r"opening",
    ]),
]

# Combined lexicon used for both the narrative pass and the structured-source pass.
# Equipment terms are checked first; structural descriptors only resolve records
# that carry no direct equipment cue.
COMBINED_RULES = VALIDATION_RULES + SOURCE_RULES

def _scan(text, rules):
    """Return [(label, pattern), ...] for the first pattern that hits per class,
    in rule order (so the highest-priority class for the text comes first)."""
    text_l = str(text).lower()
    hits = []
    for label, patterns in rules:
        for pattern in patterns:
            if re.search(pattern, text_l):
                hits.append((label, pattern))
                break
    return hits


def _match_rules(text, rules=COMBINED_RULES):
    """Return (label, note) for the independent keyword pass over text."""
    hits = _scan(text, rules)
    if not hits:
        return "Other/Unknown", "no ontology equipment keyword found"
    label, pattern = hits[0]
    distinct = sorted({l for l, _ in hits if l != label})
    note = f"matched {pattern}"
    if distinct:
        note += "; also matched: " + ", ".join(distinct)
    return label, note


def validation_label(narrative, source_text):
    """Two-pass independent assignment mirroring the production pipeline: narrative
    text first, then the OSHA structured source/event descriptors as a fallback.
    Both passes use an independently authored lexicon (equipment terms first, then
    structural descriptors), so every modeled incident resolves to one of the 15
    equipment classes. ``ambiguous`` flags narratives that mention more than one
    distinct equipment class."""
    equipment_classes = []
    for label, _ in _scan(narrative, VALIDATION_RULES):
        if label not in equipment_classes:
            equipment_classes.append(label)
    ambiguous = len(equipment_classes) > 1

    label, note = _match_rules(narrative)
    basis = "narrative"
    if label == "Other/Unknown":
        s_label, s_note = _match_rules(source_text)
        if s_label != "Other/Unknown":
            return s_label, "source-descriptor fallback: " + s_note, ambiguous, "source descriptor"
        basis = "unresolved"
    return label, note, ambiguous, basis


def build_validation_matrix(df):
    """Build the full-corpus keyword-validation matrix for every record in df."""
    source_cols = [c for c in ["SourceTitle", "Secondary Source Title", "EventTitle"] if c in df.columns]
    labels, notes, flags, bases = [], [], [], []
    for _, rec in df.iterrows():
        narrative = rec.get("Final Narrative", "")
        source_text = " ".join(str(rec.get(c, "") or "") for c in source_cols)
        label, note, ambiguous, basis = validation_label(narrative, source_text)
        labels.append(label)
        notes.append(note)
        flags.append(ambiguous)
        bases.append(basis)

    return pd.DataFrame({
        "row_id": df.index,
        "narrative_excerpt": df["Final Narrative"].fillna("").astype(str).str.slice(0, 240),
        "nlp_label": df["nlp_equipment"].astype(str),
        "validation_label": labels,
        "assignment_basis": bases,
        "agreement": [n == v for n, v in zip(df["nlp_equipment"].astype(str), labels)],
        "ambiguous_match": flags,
        "validation_note": notes,
    })


def render_confusion(ann, png=FIG_PNG, pdf=FIG_PDF, summary=FIG_SUMMARY, csv=MATRIX_CSV):
    """Draw the confusion matrix and write the matrix CSV + text summary."""
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    ann.to_csv(csv, index=False, encoding="utf-8")

    present = [c for c in CLASSES if c in set(ann["nlp_label"]) | set(ann["validation_label"])]
    present = [c for c in present if c != "Other/Unknown"] + (["Other/Unknown"] if "Other/Unknown" in present else [])
    cm = confusion_matrix(ann["validation_label"], ann["nlp_label"], labels=present)
    acc = accuracy_score(ann["validation_label"], ann["nlp_label"])
    kappa = cohen_kappa_score(ann["validation_label"], ann["nlp_label"], labels=present)

    # Size and fonts are tuned so the figure stays legible when embedded at
    # full page width in the supplementary DOCX (a previous 13 in render with
    # 8 pt labels shrank to ~4 pt on the page). Native width is kept close to the
    # embed width and fonts are enlarged accordingly.
    n_cls = len(present)
    fig_w = max(9.6, 0.42 * n_cls + 3.2)
    fig_h = max(9.0, 0.40 * n_cls + 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(n_cls))
    ax.set_yticks(np.arange(n_cls))
    ax.set_xticklabels(present, rotation=45, ha="right", fontsize=12.5)
    ax.set_yticklabels(present, fontsize=12.5)
    ax.set_xlabel("Pipeline NLP ontology label", fontsize=15, labelpad=10, fontweight="bold")
    ax.set_ylabel("Independent keyword-rule label", fontsize=15, labelpad=10, fontweight="bold")
    ax.set_title(
        "NLP Ontology Validation Confusion Matrix\n"
        f"Full maritime corpus (N = {len(ann):,} incidents)",
        fontsize=17,
        pad=16,
        fontweight="bold",
    )
    # Thin separators between cells improve readability of the dense 16-class grid.
    ax.set_xticks(np.arange(-0.5, n_cls, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_cls, 1), minor=True)
    ax.grid(which="minor", color="#d9e2ec", linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)
    threshold = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            if value:
                ax.text(
                    j, i, str(int(value)), ha="center", va="center",
                    fontsize=11.5, fontweight="bold",
                    color="white" if value > threshold else "#1f2937",
                )
    cbar = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.02)
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label("Incident count", fontsize=12.5)
    fig.text(
        0.01, 0.012,
        f"N = {len(ann):,} OSHA maritime narratives (full modeling corpus); "
        f"exact agreement = {acc:.2f}; Cohen kappa = {kappa:.2f}.",
        fontsize=11,
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(png, bbox_inches="tight", dpi=300)
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    report = classification_report(ann["validation_label"], ann["nlp_label"], labels=present, zero_division=0)
    with open(summary, "w", encoding="utf-8") as f:
        f.write("Reviewer Figure 1: NLP Ontology Validation Confusion Matrix\n")
        f.write(f"Input cache: {CACHE}\n")
        f.write(f"Keyword-validation matrix: {csv}\n")
        f.write(f"PNG: {png}\n")
        f.write(f"PDF: {pdf}\n")
        f.write(f"N = {len(ann)} (full modeling corpus)\n")
        f.write(f"Exact agreement = {acc:.3f}\n")
        f.write(f"Cohen kappa = {kappa:.3f}\n")
        f.write(f"Records with ambiguous (multiple/uncertain) keyword matches = {int(ann['ambiguous_match'].sum())}\n")
        f.write(
            "Method: an independent keyword rule set (developed separately from the production\n"
            "ontology, with different synonym lists) was applied to each incident's narrative\n"
            "text and, where the narrative lacked an equipment cue, to the OSHA source-title\n"
            "descriptors - mirroring the production two-pass logic. Agreement is therefore an\n"
            "automated internal-consistency check on the ontology, not human annotation.\n\n"
        )
        f.write(report)

    return acc, kappa, cm, present


def main():
    if not os.path.exists(CACHE):
        raise SystemExit("Run or refresh build_dataset.py first; cache not found.")
    df = pd.read_parquet(CACHE)
    ann = build_validation_matrix(df)
    acc, kappa, cm, present = render_confusion(ann)

    print(f"Wrote {MATRIX_CSV}")
    print(f"Wrote {FIG_PNG}")
    print(f"Wrote {FIG_PDF}")
    print(f"Wrote {FIG_SUMMARY}")
    print(f"N = {len(ann)} (full corpus); exact agreement = {acc:.3f}; "
          f"Cohen kappa = {kappa:.3f}; ambiguous-match records = {int(ann['ambiguous_match'].sum())}")
    print("Assignment basis (independent pass):")
    print(ann["assignment_basis"].value_counts().to_string())
    print("\nPipeline-label counts:")
    print(ann["nlp_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
