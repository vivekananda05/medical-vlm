# analysis_module.py

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from config import cfg
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

from bert_score import score as bert_score
from radgraph import F1RadGraph

from utils import compute_generation_metrics, parse_radgraph_result
# compute_generation_metrics + parse_radgraph_result from utils.py :contentReference[oaicite:0]{index=0}


# ------------------- Helpers -------------------

def _ensure_nltk():
    """Download required NLTK resources if missing."""
    try:
        nltk.download('punkt_tab', quiet=True)
    except Exception:
        # older versions don't have punkt_tab
        nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)


# Some common chest pathologies – used for a simple hallucination heuristic.
PATHOLOGY_TERMS = [
    "pneumonia", "consolidation", "atelectasis", "effusion", "pleural effusion",
    "pneumothorax", "cardiomegaly", "edema", "fracture", "nodule", "mass",
    "opacity", "infiltrate", "fibrosis", "emphysema"
]


# ------------------- Core Metric Computation -------------------

def compute_full_metrics(preds, refs, out_dir, prefix="test"):
    """
    Compute a rich set of metrics and a per-sample analysis.

    Returns:
        metrics_dict: global metrics
        per_sample_stats: list of dicts per example
    """
    os.makedirs(out_dir, exist_ok=True)
    N = len(preds)

    # --- BLEU-4 & ROUGE-L (corpus level) ---
    base_metrics = compute_generation_metrics(preds, refs)  # BLEU-4, ROUGE-L :contentReference[oaicite:1]{index=1}

    # --- METEOR (per-sample + mean) ---
    _ensure_nltk()
    meteor_scores = []
    lexical_overlap = []
    extra_findings_all = []
    hallucinated_flags = []

    for p, r in zip(preds, refs):
        p_tokens = word_tokenize(p.lower())
        r_tokens = word_tokenize(r.lower())

        # METEOR
        meteor_scores.append(meteor_score([r_tokens], p_tokens))

        # Lexical overlap ratio
        set_p = set(p_tokens)
        set_r = set(r_tokens)
        inter = set_p.intersection(set_r)
        overlap = len(inter) / (len(set_r) + 1e-8)
        lexical_overlap.append(overlap)

        # Very simple hallucination heuristic: pathology terms present in pred but not ref
        extra_terms = []
        for term in PATHOLOGY_TERMS:
            if term in p.lower() and term not in r.lower():
                extra_terms.append(term)

        hallucinated = len(extra_terms) > 0
        hallucinated_flags.append(hallucinated)
        extra_findings_all.append(extra_terms)

    meteor_avg = float(np.mean(meteor_scores)) if N > 0 else 0.0
    hallucination_rate = float(np.mean(hallucinated_flags)) if N > 0 else 0.0

    # --- RadGraph F1 (clinical relation graph) ---
    print("Computing RadGraph F1 Score...")
    radgraph_metric = F1RadGraph(reward_level="partial")
    radgraph_result = radgraph_metric(preds, refs)
    rg_metrics = parse_radgraph_result(radgraph_result)
    rg_f1 = float(rg_metrics.get("radgraph_f1", 0.0))

    # --- ClinicalBERTScore ---
    print("Computing ClinicalBERTScore...")
    P, R, F1 = bert_score(
        preds,
        refs,
        model_type="emilyalsentzer/Bio_ClinicalBERT",
        num_layers=6,
        lang="en",
        verbose=False,
        rescale_with_baseline=True
    )
    bert_f1 = float(F1.mean().item())

    # --- Optional: CIDEr (if available) ---
    cider_score = None
    try:
        from collections import defaultdict
        from pycocoevalcap.cider.cider import Cider

        # COCO-style input: {id: ["sentence1", ...]}
        gts = defaultdict(list)
        res = defaultdict(list)
        for i, (g, r) in enumerate(zip(refs, preds)):
            gts[i].append(g)
            res[i].append(r)

        cider_scorer = Cider()
        cider_score, _ = cider_scorer.compute_score(gts, res)
        cider_score = float(cider_score)
    except Exception:
        # pycocoevalcap is optional; skip if not installed
        cider_score = None

    # -------- Build per-sample stats --------
    per_sample_stats = []
    for i, (p, r) in enumerate(zip(preds, refs)):
        per_sample_stats.append({
            "id": i,
            "pred": p,
            "ref": r,
            "meteor": float(meteor_scores[i]),
            "lexical_overlap": float(lexical_overlap[i]),
            "hallucinated": bool(hallucinated_flags[i]),
            "extra_pathology_terms": extra_findings_all[i],
            "len_pred_tokens": len(word_tokenize(p)),
            "len_ref_tokens": len(word_tokenize(r)),
        })

    # -------- Global metrics summary --------
    metrics_dict = {
        "BLEU-4": float(base_metrics["BLEU-4"]),
        "ROUGE-L": float(base_metrics["ROUGE-L"]),
        "METEOR": meteor_avg,
        "ClinicalBERTScore_F1": bert_f1,
        "RadGraph_F1": rg_f1,
        "CIDEr": cider_score,
        "Hallucination_Rate": hallucination_rate,
    }

    # Save JSONs
    metrics_path = os.path.join(out_dir, f"{prefix}_metrics_{cfg.run_num}.json")
    samples_path = os.path.join(out_dir, f"{prefix}_per_sample_stats_{cfg.run_num}.json")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)

    with open(samples_path, "w", encoding="utf-8") as f:
        json.dump(per_sample_stats, f, indent=2)

    print(f"[analysis_module] Saved global metrics → {metrics_path}")
    print(f"[analysis_module] Saved per-sample stats → {samples_path}")

    return metrics_dict, per_sample_stats


# ------------------- Dashboard Plots -------------------

def plot_evaluation_dashboard(metrics, per_sample, out_dir, prefix="test"):
    """
    Create a set of plots to visually summarize performance:
      - Bar chart of global metrics
      - Histogram of per-sample METEOR
      - Histogram of lexical overlap
      - Hallucination vs non-hallucination count
    """
    os.makedirs(out_dir, exist_ok=True)

    # ---- 1. Global metrics bar chart ----
    metric_names = []
    metric_values = []
    for k in ["BLEU-4", "ROUGE-L", "METEOR", "ClinicalBERTScore_F1", "RadGraph_F1", "CIDEr", "Hallucination_Rate"]:
        if metrics.get(k) is not None:
            metric_names.append(k)
            metric_values.append(metrics[k])

    plt.figure(figsize=(8, 4))
    plt.bar(metric_names, metric_values)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Score")
    plt.title(f"{prefix.upper()} – Global Metrics Summary")
    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"{prefix}_global_metrics.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[analysis_module] Saved global metrics plot → {fig_path}")

    # Extract per-sample arrays
    meteor_vals = [s["meteor"] for s in per_sample]
    overlap_vals = [s["lexical_overlap"] for s in per_sample]
    halluc_flags = [s["hallucinated"] for s in per_sample]

    # ---- 2. METEOR histogram ----
    plt.figure(figsize=(6, 4))
    plt.hist(meteor_vals, bins=20)
    plt.xlabel("METEOR Score")
    plt.ylabel("Count")
    plt.title(f"{prefix.upper()} – Per-sample METEOR Distribution")
    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"{prefix}_meteor_hist.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[analysis_module] Saved METEOR histogram → {fig_path}")

    # ---- 3. Lexical overlap histogram ----
    plt.figure(figsize=(6, 4))
    plt.hist(overlap_vals, bins=20)
    plt.xlabel("Lexical Overlap (pred ∩ ref / ref)")
    plt.ylabel("Count")
    plt.title(f"{prefix.upper()} – Lexical Overlap Distribution")
    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"{prefix}_overlap_hist.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[analysis_module] Saved lexical overlap histogram → {fig_path}")

    # ---- 4. Hallucination rate bar ----
    n_hall = sum(halluc_flags)
    n_non = len(halluc_flags) - n_hall

    plt.figure(figsize=(4, 4))
    plt.bar(["Non-hallucinated", "Hallucinated"], [n_non, n_hall])
    plt.ylabel("Number of Reports")
    plt.title(f"{prefix.upper()} – Hallucination Analysis")
    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"{prefix}_hallucination_bar.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[analysis_module] Saved hallucination bar plot → {fig_path}")
