import os
import torch
from tqdm import tqdm
from config import cfg
from dataset import get_dataloader
from model import VisionEncoderWrapper, TextEncoderWrapper, AlignModule, VisualPrefixForLM
from utils import load_checkpoint, compute_generation_metrics
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
os.environ["HF_TOKEN"] = cfg.HF_TOKEN
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from bert_score import score as bert_score
from radgraph import F1RadGraph
import numpy as np


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
    load_checkpoint(align_module, cfg.save_dir_align, map_location=device)
    load_checkpoint(prefix_adapter, cfg.save_dir_prefix, map_location=device)
    print("  Loaded trained weights for AlignModule and PrefixAdapter")

    for p in lm.parameters():
        p.requires_grad = False
    vision.eval(); align_module.eval(); prefix_adapter.eval()

    # Normalize LM embeddings to avoid scale mismatch
    lm_token_emb = torch.nn.functional.normalize(
        lm.get_input_embeddings().weight.detach().to(device), dim=-1
    )
    
    # lm_token_emb = lm.get_input_embeddings().weight.detach().to(device)  

    # -------------------- Load Test Data -------------------------------
    test_loader = get_dataloader(
        cfg.data_root, split="test", tokenizer=tokenizer,
        batch_size=cfg.batch_size, image_size=cfg.image_size, max_text_len=cfg.max_text_len
    )
    print(f"  Loaded Test Split: {len(test_loader)} batches")

    preds, refs, saved_preds, saved_refs, saved_imgs = [], [], [], [], []

    # -------------------- Evaluation Loop -------------------------------


    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating on Test Set", ncols=100)):
    
        imgs = batch["image"].to(device)
        texts = batch["text"]

    # Encode image
        vis_feat = vision(imgs)
        _, F_aligned = align_module(vis_feat, lm_token_embeddings=lm_token_emb)
        F_aligned = F.normalize(F_aligned, dim=-1)
        #text_pooled = F.normalize(text_pooled, dim=-1)
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
            temperature=1, 
            top_p=0.9,
            top_k=5,
            pad_token_id=lm.config.eos_token_id,
            #repetition_penalty=1.2,
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


    
    # -------------------- Compute Metrics -------------------------------
    metrics = compute_generation_metrics(preds, refs)
    print("\n  Evaluation Metrics on Test Split")
    print("------------------------------------")
    print(f"BLEU-4 Score  : {metrics['BLEU-4']:.4f}")
    print(f"ROUGE-L Score : {metrics['ROUGE-L']:.4f}")

    # -------------------- METEOR -------------------------------
    print("\nComputing METEOR Score...")
    # meteor_scores = [meteor_score([r], p) for p, r in zip(preds, refs)]
    # meteor_avg = np.mean(meteor_scores)
    # print(f"METEOR Score  : {meteor_avg:.4f}")
    nltk.download('punkt_tab', quiet=True)
    nltk.download('wordnet', quiet=True)
    meteor_scores = []
    for p, r in zip(preds, refs):
        # Tokenize both hypothesis (prediction) and reference
        p_tokens = word_tokenize(p.lower())
        r_tokens = word_tokenize(r.lower())
        meteor_scores.append(meteor_score([r_tokens], p_tokens))

    meteor_avg = np.mean(meteor_scores)
    print(f"METEOR Score  : {meteor_avg:.4f}")
    
    # -------------------- RadGraph F1 -------------------------------
    # print("Computing RadGraph F1 Score...")
    # radgraph_metric = F1RadGraph(reward_level="partial")
    # rg_f1 = radgraph_metric(preds, refs)['radgraph_f1']
    # print(f"RadGraph F1 Score : {rg_f1:.4f}")

    # -------------------- ClinicalBERTScore -------------------------------
    #print("Computing ClinicalBERTScore...")
    # P, R, F1 = bert_score(preds, refs, model_type="emilyalsentzer/Bio_ClinicalBERT", lang="en", verbose=False)
    # P, R, F1 = bert_score(
    #     preds,
    #     refs,
    #     model_type="emilyalsentzer/Bio_ClinicalBERT",
    #     num_layers=12,          # Explicitly specify the layer count
    #     lang="en",
    #     verbose=False,
    #     rescale_with_baseline=True
    #     )

    # bert_f1 = F1.mean().item()
    # print(f"ClinicalBERTScore (F1): {bert_f1:.4f}")

   
    # -------------------- Combine and Display -------------------------------
    print("\n======== Final Evaluation Summary ========")
    print(f"BLEU-4            : {metrics['BLEU-4']:.4f}")
    print(f"ROUGE-L           : {metrics['ROUGE-L']:.4f}")
    print(f"METEOR            : {meteor_avg:.4f}")
    # print(f"ClinicalBERTScore : {bert_f1:.4f}")
    # print(f"RadGraph F1       : {rg_f1:.4f}")
    print("==========================================")

    # -------------------- Save Predictions -------------------------------
    result_root = os.path.join(cfg.project_root, "results")
    os.makedirs(result_root, exist_ok=True)
    results_path = os.path.join(result_root, "test_predictions_4.txt")
    sample_root = os.path.join(result_root, "samples_4")
    os.makedirs(sample_root, exist_ok=True)
    with open(cfg.results_path, "w", encoding="utf-8") as f:
        for i, (p, r) in enumerate(zip(saved_preds, saved_refs)):
            f.write(f"=== SAMPLE {i+1} ===\n[Generated]: {p.strip()}\n[Reference]: {r.strip()}\n\n")

    print(f" Saved {len(saved_preds)} qualitative samples to {results_path}")

    # Save metrics to file
    metrics_path = os.path.join(result_root, "evaluation_metrics_4.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("=== Evaluation Metrics ===\n")
        f.write(f"BLEU-4 Score        : {metrics['BLEU-4']:.4f}\n")
        f.write(f"ROUGE-L Score       : {metrics['ROUGE-L']:.4f}\n")
        f.write(f"METEOR Score        : {meteor_avg:.4f}\n")
        # f.write(f"ClinicalBERTScore F1: {bert_f1:.4f}\n")
        # f.write(f"RadGraph F1 Score   : {rg_f1:.4f}\n")
    print(f"   Saved all metrics to {metrics_path}")

   
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

if __name__ == "__main__":
    evaluate(save_samples=cfg.samples)
