import os
import torch
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from config import cfg
from dataset import get_dataloader
from model import VisionEncoderWrapper, TextEncoderWrapper, AlignModule, VisualPrefixForLM
from utils import save_checkpoint, plot_multi_losses, compute_generation_metrics
import matplotlib.pyplot as plt
import torch.nn.functional as F

os.environ["HF_TOKEN"] = cfg.HF_TOKEN
# ---------------------------------------------------------------------------
# Contrastive loss (InfoNCE)
# ---------------------------------------------------------------------------
def contrastive_loss(vision_feats, text_feats, temperature=0.07):
    """Symmetric InfoNCE contrastive loss between normalized vision/text features."""
    v = F.normalize(vision_feats, dim=-1)
    t = F.normalize(text_feats, dim=-1)
    logits = (v @ t.t()) / temperature
    labels = torch.arange(v.size(0), device=v.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)
    return (loss_i + loss_t) / 2


def cosine_similarity_loss(vision_feats, text_feats):
    """Compute cosine similarity loss between paired image/text embeddings."""
    v = F.normalize(vision_feats, dim=-1)
    t = F.normalize(text_feats, dim=-1)
    cosine_sim = (v * t).sum(dim=-1)  # elementwise dot product per pair
    loss = 1 - cosine_sim.mean()      # we want cosine_sim → 1
    return loss

# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------
def train():
    device = cfg.device
    print(f"  Training on device: {device}")

    # -----------------------------------------------------------------------
    # Model setup
    # -----------------------------------------------------------------------
    text_wrapper = TextEncoderWrapper(cfg, freeze_backbone=True, device=device)
    tokenizer = text_wrapper.tokenizer
    lm = text_wrapper.model.to(device)
    lm.eval()

    vision = VisionEncoderWrapper(cfg, freeze_backbone=True).to(device)
    align_module = AlignModule(cfg, lm_model=lm, hidden=cfg.proj_hidden, init_with_lm_head=True).to(device)
    prefix_adapter = VisualPrefixForLM(cfg, lm_model=lm, prefix_len=cfg.prefix_length).to(device)

    #lm_token_emb = lm.get_input_embeddings().weight.detach().to(device)
    lm_token_emb = F.normalize(lm.get_input_embeddings().weight.detach().to(device), dim=-1)

    for p in lm.parameters():
        p.requires_grad = False
    
    # Unfreeze AlignModule (trainable projection)
    for p in align_module.parameters():
        p.requires_grad = True

    # Unfreeze VisualPrefixForLM (trainable multimodal adapter)
    for p in prefix_adapter.parameters():
        p.requires_grad = True

    params = list(align_module.parameters()) + list(prefix_adapter.parameters())
    #optimizer = optim.AdamW(params, lr=cfg.lr, weight_decay=1e-4)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad,
            list(align_module.parameters()) +
            list(prefix_adapter.parameters())),
            lr=cfg.lr,
            weight_decay=1e-4
            )

    # -----------------------------------------------------------------------
    # Data setup
    # -----------------------------------------------------------------------
    train_loader = get_dataloader(cfg.data_root, split="train", tokenizer=tokenizer, batch_size=cfg.batch_size, image_size=cfg.image_size, max_text_len=cfg.max_text_len)
    val_loader = get_dataloader(cfg.data_root, split="val", tokenizer=tokenizer, batch_size=cfg.batch_size, image_size=cfg.image_size, max_text_len=cfg.max_text_len)
    print(f"  Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")

    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id or -100, label_smoothing=0.1)

    # Track losses
    loss_history = {
        "Total": ([], []),
        "Align": ([], []),
        "Contrastive": ([], []),
        "LM": ([], []),
        "Cosine": ([], []),
    }

    # -----------------------------------------------------------------------
    # Best model tracking
    # -----------------------------------------------------------------------
    best_val_loss = float("inf")
    os.makedirs("checkpoints", exist_ok=True)

    # -----------------------------------------------------------------------
    # Training Loop
    # -----------------------------------------------------------------------
    for epoch in range(cfg.epochs):
        align_module.train()
        prefix_adapter.train()

        total_train, total_val = 0, 0
        align_train, align_val = 0, 0
        contr_train, contr_val = 0, 0
        cos_train, cos_val = 0, 0
        lm_train, lm_val = 0, 0

        # ----------------------------- TRAIN -------------------------------
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} [Train]"):
            imgs = batch["image"].to(device)
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            with torch.no_grad():
                text_pooled = text_wrapper.pooled_text_embedding(ids, mask)

            vis_feat = vision(imgs)
            _, F_aligned = align_module(vis_feat, lm_token_embeddings=lm_token_emb)

            F_aligned_norm = F.normalize(F_aligned, dim=-1)
            text_pooled_norm = F.normalize(text_pooled, dim=-1)
            align_l = mse_loss(F_aligned_norm, text_pooled_norm)
            contr_l = contrastive_loss(F_aligned, text_pooled, cfg.temperature)
            cos_l   = cosine_similarity_loss(F_aligned, text_pooled)

            outputs = prefix_adapter(ids, mask, F_aligned)
            logits = outputs.logits
            prefix_len = prefix_adapter.prefix_len
            lm_logits_for_loss = logits[:, prefix_len:-1, :]
            target = ids[:, 1:]
            lm_l = ce_loss(lm_logits_for_loss.reshape(-1, lm_logits_for_loss.size(-1)), target.reshape(-1))

            total_loss = cfg.lambda_align*align_l +  cfg.lambda_contr*contr_l + cfg.lambda_lm*lm_l + cfg.lambda_cos*cos_l

            optimizer.zero_grad()
            total_loss.backward()
            clip_grad_norm_(params, 1.0)
            optimizer.step()
            
            torch.cuda.empty_cache()

            total_train += total_loss.item()
            align_train += align_l.item()
            contr_train += contr_l.item()
            lm_train += lm_l.item()
            cos_train += cos_l.item()

        # ----------------------------- VALIDATION --------------------------
        align_module.eval()
        prefix_adapter.eval()
        val_preds, val_refs = [], []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} [Val]")):
                imgs = batch["image"].to(device)
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                texts = batch["text"]

                text_pooled = text_wrapper.pooled_text_embedding(ids, mask)
                vis_feat = vision(imgs)
                _, F_aligned = align_module(vis_feat, lm_token_embeddings=lm_token_emb)
                
                        
                # if batch_idx == 0 and epoch % 2 == 0:
                #     img_norm = F.normalize(F_aligned, dim=-1)
                #     txt_norm = F.normalize(text_pooled, dim=-1)
                #     sim_matrix = img_norm @ txt_norm.T  

                #     plt.figure(figsize=(6,5))
                #     plt.imshow(sim_matrix.detach().cpu().numpy(), cmap='coolwarm', vmin=-1, vmax=1)
                #     plt.colorbar(label='Cosine Similarity')
                #     plt.title(f"Image-Text Alignment Matrix (Epoch {epoch+1})")
                #     plt.xlabel("Text Index")
                #     plt.ylabel("Image Index")
                #     plt.tight_layout()
                #     save_path = os.path.join(cfg.plot_dir, f"img_text_alignment_epoch{epoch+1}.png")
                #     plt.savefig(save_path, dpi=150)
                #     plt.close()
                    
                F_aligned_norm = F.normalize(F_aligned, dim=-1)
                text_pooled_norm = F.normalize(text_pooled, dim=-1)
                align_l = mse_loss(F_aligned_norm, text_pooled_norm)
                contr_l = contrastive_loss(F_aligned, text_pooled, cfg.temperature)
                cos_l   = cosine_similarity_loss(F_aligned, text_pooled)

                outputs = prefix_adapter(ids, mask, F_aligned)
                logits = outputs.logits
                prefix_len = prefix_adapter.prefix_len
                lm_logits_for_loss = logits[:, prefix_len:-1, :]
                target = ids[:, 1:]
                lm_l = ce_loss(lm_logits_for_loss.reshape(-1, lm_logits_for_loss.size(-1)), target.reshape(-1))

                total_loss = cfg.lambda_align*align_l +  cfg.lambda_contr*contr_l + cfg.lambda_lm*lm_l + cfg.lambda_cos*cos_l

                torch.cuda.empty_cache()

                total_val += total_loss.item()
                align_val += align_l.item()
                contr_val += contr_l.item()
                lm_val += lm_l.item()
                cos_val += cos_l.item()

                if len(val_preds) < 100:
                    pred = tokenizer.decode(torch.argmax(logits[0, prefix_len:], dim=-1), skip_special_tokens=True)
                    val_preds.append(pred)
                    val_refs.append(texts[0])

        # ----------------------------- METRICS -----------------------------
        Ntrain, Nval = len(train_loader), len(val_loader)
        avg_val_loss = total_val / Nval
        loss_history["Total"][0].append(total_train / Ntrain)
        loss_history["Total"][1].append(avg_val_loss)
        loss_history["Align"][0].append(align_train / Ntrain)
        loss_history["Align"][1].append(align_val / Nval)
        loss_history["Contrastive"][0].append(contr_train / Ntrain)
        loss_history["Contrastive"][1].append(contr_val / Nval)
        loss_history["LM"][0].append(lm_train / Ntrain)
        loss_history["LM"][1].append(lm_val / Nval)
        loss_history["Cosine"][0].append(cos_train/ Ntrain)
        loss_history["Cosine"][1].append(cos_val/ Nval)

        print(f"\n  Epoch {epoch+1}/{cfg.epochs} Summary:")
        print(f"Train Total: {loss_history['Total'][0][-1]:.4f} | Val Total: {avg_val_loss:.4f}")
        print(f"Train Align: {align_train/Ntrain:.4f} | Train Contrastive: {contr_train/Ntrain:.4f} | Train LM: {lm_train/Ntrain:.4f} | Train Cosine: {cos_train/Ntrain:.4f}")
        print(f"Val Align: {align_val/Nval:.4f} | Val Contrastive: {contr_val/Nval:.4f} | Val LM: {lm_val/Nval:.4f} | Val Cosine: {cos_val/Nval:.4f}")
        
        # BLEU/ROUGE validation metrics
        if val_preds:
            metrics = compute_generation_metrics(val_preds, val_refs)
            print(f" BLEU-4: {metrics['BLEU-4']:.4f} | ROUGE-L: {metrics['ROUGE-L']:.4f}")

        # ----------------------------- SAVE MODEL --------------------------

        #  Save the best model based on lowest validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"\n  New best model found at epoch {epoch+1}! Saving best models...")
            save_checkpoint(align_module.state_dict(), cfg.save_dir_align)
            save_checkpoint(prefix_adapter.state_dict(), cfg.save_dir_prefix)

    # -----------------------------------------------------------------------
    # After all epochs → Plot loss curves
    # -----------------------------------------------------------------------
    plot_multi_losses(loss_history, save_dir=cfg.plot_dir)
    print("\n  Training complete. Loss plots saved under /plots/ and models under /checkpoints/.")
    print(f"  Best model achieved validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    train()
