import os
import torch
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge import Rouge

def save_checkpoint(state_dict, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state_dict, path)

def load_checkpoint(model, path, map_location=None):
    model.load_state_dict(torch.load(path, map_location=map_location))

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
