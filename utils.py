import os
import torch
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge import Rouge

def save_checkpoint(state_dict, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state_dict, path)

def load_checkpoint(model, path, map_location=None):
    model.load_state_dict(torch.load(path, map_location=map_location, weights_only=True))

def plot_multi_losses(loss_dict, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    for name, (train_vals, val_vals) in loss_dict.items():
        plt.figure(figsize=(7, 5))
        plt.plot(train_vals, label=f"Train {name}", marker="o")
        plt.plot(val_vals, label=f"Val {name}", marker="s")
        plt.title(f"{name} Loss per Epoch")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        path = os.path.join(save_dir, f"{name.lower()}_loss_curve.png")
        plt.savefig(path)
        plt.close()
        print(f" Saved {name} loss plot at {path}")

def compute_generation_metrics(preds, refs):
    smoothie = SmoothingFunction().method4
    ref_tokens = [[r.split()] for r in refs]
    pred_tokens = [p.split() for p in preds]
    bleu4 = corpus_bleu(ref_tokens, pred_tokens, smoothing_function=smoothie)
    rouge = Rouge()
    rouge_l = rouge.get_scores(preds, refs, avg=True)["rouge-l"]["f"]
    return {"BLEU-4": bleu4, "ROUGE-L": rouge_l}


def parse_radgraph_result(radgraph_result):
    """
    Safely extract RadGraph metrics regardless of return format.
    Supports both dict and tuple outputs.
    Returns: dict with keys {f1, precision, recall}
    """
    rg_f1 = rg_prec = rg_rec = 0.0

    if isinstance(radgraph_result, dict):
        rg_f1 = radgraph_result.get("radgraph_f1", 0.0)
        rg_prec = radgraph_result.get("precision", 0.0)
        rg_rec = radgraph_result.get("recall", 0.0)

    elif isinstance(radgraph_result, (list, tuple)):
        # Filter out numeric values (some versions return nested objects)
        vals = [v for v in radgraph_result if isinstance(v, (float, int))]
        if len(vals) >= 3:
            rg_f1, rg_prec, rg_rec = vals[:3]
        elif len(vals) == 2:
            rg_f1, rg_prec = vals
        elif len(vals) == 1:
            rg_f1 = vals[0]

    else:
        print("[Warning] Unexpected RadGraph output type:", type(radgraph_result))

    return {"radgraph_f1": rg_f1, "precision": rg_prec, "recall": rg_rec}
