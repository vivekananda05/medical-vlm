# test.py

import os
import torch
from tqdm import tqdm
from config import cfg
from dataset import get_dataloader
from model import VisionEncoderWrapper, TextEncoderWrapper, AlignModule, VisualPrefixForLM
from utils import load_checkpoint
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import numpy as np

# New: import analysis module
from analysis_module import compute_full_metrics, plot_evaluation_dashboard  # new

os.environ["HF_TOKEN"] = cfg.HF_TOKEN


@torch.no_grad()
def evaluate(save_samples=20):
    device = cfg.device
    print(f"  Evaluating on device: {device}")

    # -------------------- Load Text & Vision Modules --------------------
    text_wrapper = TextEncoderWrapper(cfg, freeze_backbone=True, device=device)
    tokenizer = text_wrapper.tokenizer
    lm = text_wrapper.model.to(device)
    lm.eval()

    vision = VisionEncoderWrapper(cfg, freeze_backbone=True).to(device)
    align_module = AlignModule(cfg, lm_model=lm, hidden=cfg.proj_hidden, init_with_lm_head=True).to(device)
    prefix_adapter = VisualPrefixForLM(cfg, lm_model=lm, prefix_len=cfg.prefix_length).to(device)

    # -------------------- Load Trained Checkpoints ----------------------
    load_checkpoint(align_module, cfg.save_dir_align, map_location=device)   # :contentReference[oaicite:2]{index=2}
    load_checkpoint(prefix_adapter, cfg.save_dir_prefix, map_location=device)
    print("  Loaded trained weights for AlignModule and PrefixAdapter")

    for p in lm.parameters():
        p.requires_grad = False
    vision.eval(); align_module.eval(); prefix_adapter.eval()

    # Normalize LM embeddings to avoid scale mismatch
    lm_token_emb = torch.nn.functional.normalize(
        lm.get_input_embeddings().weight.detach().to(device), dim=-1
    )

    # -------------------- Load Test Data -------------------------------
    test_loader = get_dataloader(
        cfg.data_root, split="test", tokenizer=tokenizer,
        batch_size=cfg.batch_size, image_size=cfg.image_size, max_text_len=cfg.max_text_len
    )  # :contentReference[oaicite:3]{index=3}
    print(f"  Loaded Test Split: {len(test_loader)} batches")

    preds, refs = [], []
    saved_preds, saved_refs, saved_imgs = [], [], []

    # -------------------- Evaluation Loop -------------------------------
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating on Test Set", ncols=100)):
        imgs = batch["image"].to(device)
        texts = batch["text"]

        # Encode image
        vis_feat = vision(imgs)
        _, F_aligned = align_module(vis_feat, lm_token_embeddings=lm_token_emb)
        F_aligned = F.normalize(F_aligned, dim=-1)

        # Convert to prefix embeddings
        prefix_emb = prefix_adapter.to_prefix(F_aligned).view(
            F_aligned.size(0), prefix_adapter.prefix_len, lm.config.hidden_size
        )

        # Safe prompt initialization
        bos_token_id = tokenizer.bos_token_id or tokenizer.pad_token_id or 0
        prompt_ids = torch.full(
            (F_aligned.size(0), 1), bos_token_id, dtype=torch.long, device=device
        )

        # Combine prefix + prompt embeddings
        prompt_emb = lm.get_input_embeddings()(prompt_ids)
        input_embs = torch.cat([prefix_emb, prompt_emb], dim=1)
        attn_mask = torch.ones(input_embs.size()[:2], device=device)

        # Text generation
        gen_tokens = lm.generate(
            inputs_embeds=input_embs,
            attention_mask=attn_mask,
            max_new_tokens=60,
            do_sample=True,
            temperature=1.0,
            top_p=0.9,
            top_k=5,
            pad_token_id=lm.config.eos_token_id,
        )

        # Decode predictions
        for i in range(len(gen_tokens)):
            pred = tokenizer.decode(gen_tokens[i], skip_special_tokens=True)
            preds.append(pred)
            refs.append(texts[i])
            if len(saved_preds) < save_samples:
                saved_preds.append(pred)
                saved_refs.append(texts[i])
                saved_imgs.append(imgs[i].cpu())

    # -------------------- Full Metric + Analysis Module -----------------
    print("\nRunning full metric analysis + dashboard creation...")
    metrics, per_sample = compute_full_metrics(
        preds, refs, out_dir=cfg.result_root, prefix="test"
    )

    # Nice console summary
    print("\n  Evaluation Metrics on Test Split")
    print("------------------------------------")
    for k, v in metrics.items():
        if v is not None:
            print(f"{k:22s}: {v:.4f}" if isinstance(v, float) else f"{k:22s}: {v}")

    print("\nCreating evaluation dashboard plots...")
    plot_evaluation_dashboard(
        metrics, per_sample, out_dir=cfg.plot_dir, prefix="test"
    )  # plots in cfg.plot_dir :contentReference[oaicite:4]{index=4}

    # -------------------- Also Save Metrics as TXT (for continuity) -----
    with open(cfg.metrics_path, "w", encoding="utf-8") as f:
        f.write("=== Evaluation Metrics (Summary) ===\n")
        for k, v in metrics.items():
            f.write(f"{k:22s}: {v:.6f}\n" if isinstance(v, float) else f"{k:22s}: {v}\n")
    print(f"   Saved metrics summary TXT to {cfg.metrics_path}")

    # -------------------- Save Predictions (Qualitative) ----------------
    os.makedirs(os.path.dirname(cfg.results_path), exist_ok=True)
    with open(cfg.results_path, "w", encoding="utf-8") as f:
        for i, (p, r) in enumerate(zip(saved_preds, saved_refs)):
            f.write(f"=== SAMPLE {i+1} ===\n[Generated]: {p.strip()}\n[Reference]: {r.strip()}\n\n")

    print(f" Saved {len(saved_preds)} qualitative samples to {cfg.results_path}")

    # -------------------- Save Image + Text Overlay Samples -------------
    os.makedirs(cfg.sample_root, exist_ok=True)
    for i, (img, gen_text, ref_text) in enumerate(zip(saved_imgs, saved_preds, saved_refs)):
        img_pil = to_pil_image((img - img.min()) / (img.max() - img.min()))
        plt.figure(figsize=(6, 6))
        plt.imshow(img_pil, cmap='gray')
        plt.axis("off")

        plt.title(f"[Generated]\n{gen_text}\n\n[Reference]\n{ref_text}", fontsize=8, wrap=True)
        plt.tight_layout()
        save_path = os.path.join(cfg.sample_root, f"sample_{i+1}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()

    print(f"   Saved {len(saved_imgs)} image samples to {cfg.sample_root}")


if __name__ == "__main__":
    evaluate(save_samples=cfg.samples)
