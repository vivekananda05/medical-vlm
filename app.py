import os
import torch
from PIL import Image
import streamlit as st
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from config import cfg
from dataset import get_dataloader  # (optional for tokenizer)
from model import VisionEncoderWrapper, TextEncoderWrapper, AlignModule, VisualPrefixForLM
from utils import load_checkpoint

# -----------------------------------------------
#  Setup Environment
# -----------------------------------------------
os.environ["HF_TOKEN"] = cfg.HF_TOKEN
device = cfg.device

# -----------------------------------------------
#  Load Model Components
# -----------------------------------------------
@st.cache_resource
def load_models():
    text_wrapper = TextEncoderWrapper(cfg, freeze_backbone=True, device=device)
    tokenizer = text_wrapper.tokenizer
    lm = text_wrapper.model.to(device).eval()

    vision = VisionEncoderWrapper(cfg, freeze_backbone=True).to(device).eval()
    align_module = AlignModule(cfg, lm_model=lm, hidden=cfg.proj_hidden, init_with_lm_head=True).to(device).eval()
    prefix_adapter = VisualPrefixForLM(cfg, lm_model=lm, prefix_len=cfg.prefix_length).to(device).eval()

    load_checkpoint(align_module, cfg.save_dir_align, map_location=device)
    load_checkpoint(prefix_adapter, cfg.save_dir_prefix, map_location=device)

    return vision, align_module, prefix_adapter, lm, tokenizer

vision, align_module, prefix_adapter, lm, tokenizer = load_models()

st.title("🩺 Medical Image Report Generator")
st.write("Upload a medical image (e.g., X-ray, MRI, CT scan) to generate an AI-based diagnostic report.")

# -----------------------------------------------
#  Upload Section
# -----------------------------------------------
uploaded_file = st.file_uploader("Upload a medical image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    #st.image(image, caption="Uploaded Medical Image", use_container_width=True)
     # 🔹 Smaller preview in the UI
    preview = image.resize((400, 400))
    st.image(
        preview,
        caption="Uploaded Medical Image",
    )

    # -----------------------------------------------
    #  Preprocess and Encode
    # -----------------------------------------------
    transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        # Vision encoding
        vis_feat = vision(img_tensor)
        #lm_token_emb = lm.get_input_embeddings().weight.detach().to(device)
        lm_token_emb = F.normalize(lm.get_input_embeddings().weight.detach().to(device), dim=-1)
        _, F_aligned = align_module(vis_feat, lm_token_embeddings=lm_token_emb)
        F_aligned = torch.nn.functional.normalize(F_aligned, dim=-1)

        # Convert to prefix embeddings
        prefix_emb = prefix_adapter.to_prefix(F_aligned).view(
            F_aligned.size(0), prefix_adapter.prefix_len, lm.config.hidden_size
        )

        # Prepare LM prompt
        bos_token_id = tokenizer.bos_token_id or tokenizer.pad_token_id or 0
        prompt_ids = torch.full((1, 1), bos_token_id, dtype=torch.long, device=device)
        prompt_emb = lm.get_input_embeddings()(prompt_ids)
        input_embs = torch.cat([prefix_emb, prompt_emb], dim=1)
        attn_mask = torch.ones(input_embs.size()[:2], device=device)

        # Generate text
        st.write(" Generating report... please wait.")
        gen_tokens = lm.generate(
            inputs_embeds=input_embs,
            attention_mask=attn_mask,
            max_new_tokens=80,
            do_sample=True,
            temperature=1.0,
            top_p=0.9,
            top_k=5,
            pad_token_id=lm.config.eos_token_id,
        )

        # Decode text
        gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
        st.success(" Report Generated!")

        # Display Result
        st.subheader("Generated Medical Report:")
        st.write(gen_text)

        # Save option
        # if st.button(" Save Report"):
        #     save_dir = os.path.join(cfg.project_root, "gui_reports")
        #     os.makedirs(save_dir, exist_ok=True)
        #     base_name = os.path.splitext(os.path.basename(uploaded_file.name))[0]
        #     with open(os.path.join(save_dir, f"{base_name}_report.txt"), "w") as f:
        #         f.write(gen_text)
        #     st.info(f"Report saved to: {save_dir}/{base_name}_report.txt")
