import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import os 

# class VisionEncoderWrapper(nn.Module):
#     """Frozen DINO-ViT + small projection."""
#     def __init__(self, cfg, freeze_backbone=True):
#         super().__init__()
#         self.backbone = timm.create_model(cfg.vision_model_name, pretrained=True, num_classes=0, global_pool="avg")
#         self.vision_proj = nn.Linear(self.backbone.num_features, cfg.vision_emb_dim)
        
#         if freeze_backbone:
#             for p in self.backbone.parameters():
#                 p.requires_grad = False

#     def forward(self, x):
#         return self.vision_proj(self.backbone(x))

class VisionEncoderWrapper(nn.Module):
    """MobileNetV3 (or any timm model) + projection."""
    def __init__(self, cfg, freeze_backbone=True):
        super().__init__()

        # Load backbone dynamically
        self.backbone = timm.create_model(
            cfg.vision_model_name,
            pretrained=True,
            num_classes=0,
            global_pool="avg"
        )

        # Dynamically infer real output dimension (e.g., 1280 for MobileNetV3)
        with torch.no_grad():
            dummy = torch.randn(1, 3, cfg.image_size, cfg.image_size)
            out = self.backbone(dummy)
            backbone_out_dim = out.shape[1]
        # backbone_out_dim = self.backbone.num_features
        print(f"[VisionEncoderWrapper] Using {cfg.vision_model_name} → actual output dim = {backbone_out_dim}")

        # Projection layer (input = backbone_out_dim)
        self.vision_proj = nn.Linear(backbone_out_dim, cfg.vision_emb_dim)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        feat = self.backbone(x)
        return self.vision_proj(feat)



class TextEncoderWrapper(nn.Module):
    """LLM wrapper for tokenization + embedding pooling."""
    def __init__(self, cfg, freeze_backbone=True, device="cpu"):
        super().__init__()
        
         # --- Authenticate HuggingFace ---
        token = os.getenv("HF_TOKEN")
        if not token:
            raise ValueError("Please set your Hugging Face token as HF_TOKEN environment variable.")

        # Log in programmatically
        # try:
        #     login(token=token, add_to_git_credential=True)
        # except Exception as e:
        #     print(f"Hugging Face login skipped or failed: {e}")

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name, use_fast=True, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(cfg.text_model_name, device_map=None, token=token).to(device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '<PAD>'})
            
            self.model.resize_token_embeddings(len(self.tokenizer))
            print(" Added missing pad_token '<PAD>' to tokenizer.")
        if freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad = False

    def pooled_text_embedding(self, input_ids, attention_mask):
        outputs = self.model.base_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return pooled


class AlignModule(nn.Module):
    """AlignVLM convex projection alignment."""
    def __init__(self, cfg, lm_model, hidden=None, init_with_lm_head=True):
        super().__init__()
        hidden = hidden or cfg.proj_hidden
        self.vocab_size = lm_model.config.vocab_size
        self.W1 = nn.Linear(cfg.vision_emb_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.W2 = nn.Linear(hidden, self.vocab_size)
        self.ln2 = nn.LayerNorm(self.vocab_size)

        torch.nn.init.xavier_uniform_(self.W1.weight)
        torch.nn.init.xavier_uniform_(self.W2.weight)

        if self.W1.bias is not None:
            torch.nn.init.zeros_(self.W1.bias)
        if self.W2.bias is not None:
            torch.nn.init.zeros_(self.W2.bias)

        if init_with_lm_head and hasattr(lm_model, "lm_head"):
            try:
                w = lm_model.lm_head.weight.data.clone()
                if self.W2.weight.shape[1] == w.shape[1]:
                    self.W2.weight.data.copy_(w)
            except Exception:
                pass

    def forward(self, vision_feat, lm_token_embeddings=None):
        x = self.W1(vision_feat)
        x = self.ln1(x)
        x = F.gelu(x)
        x = self.W2(x)
        x = self.ln2(x)
        P = F.softmax(x, dim=-1)
        F_aligned = None
        if lm_token_embeddings is not None:
            F_aligned = P @ lm_token_embeddings
        return P, F_aligned


class VisualPrefixForLM(nn.Module):
    """Convert aligned features into LM prefix embeddings."""
    def __init__(self, cfg, lm_model, prefix_len=None):
        super().__init__()
        self.prefix_len = prefix_len or cfg.prefix_length
        self.embed_dim = lm_model.config.hidden_size
        # self.to_prefix = nn.Sequential(
        #     nn.Linear(self.embed_dim, self.embed_dim * self.prefix_len),
        #     nn.Tanh()
        # )
        self.to_prefix = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.ReLU(),
            nn.Linear(self.embed_dim * 2, self.embed_dim * self.prefix_len),
            #nn.Tanh()
            )
        self.norm = nn.LayerNorm(self.embed_dim)    
        self.lm = lm_model

    def forward(self, input_ids, attention_mask, aligned_vectors):
        B = aligned_vectors.size(0)
        prefix_emb = self.to_prefix(aligned_vectors).view(B, self.prefix_len, self.embed_dim)
        prefix_emb = self.norm(prefix_emb + 0.1 * torch.randn_like(prefix_emb))
        input_emb = self.lm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([prefix_emb, input_emb], dim=1)
        prefix_mask = torch.ones((B, self.prefix_len), device=attention_mask.device)
        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        return self.lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True)


