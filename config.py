from dataclasses import dataclass
import torch
import os

@dataclass
class Config:
    # dataset
    data_root: str = "/mnt/DATA1/vivekananda/DATA/iux_data"
    image_size: int = 224
    max_text_len: int = 256
    
    HF_TOKEN = "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"
    # models
    vision_model_name: str = "vit_base_patch16_224"    # "vit_base_patch16_224"  "mobilenetv3_large_100"
    text_model_name: str = "distilgpt2"                 # "distilgpt2"   "meta-llama/Llama-3.2-1B"
    vision_emb_dim: int = 768
    proj_hidden: int = 768
    prefix_length: int = 10

    # training
    batch_size: int = 16
    lr: float = 5e-5
    epochs: int = 20
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    lambda_align = 1.0
    lambda_contr = 0.5
    lambda_lm = 0.5
    temperature = 0.1
    project_root = "/mnt/DATA1/vivekananda/medical_alignvlm"
    checkpoints = os.path.join(project_root, "checkpoints")
    os.makedirs(checkpoints, exist_ok=True)
    save_dir_align = os.path.join(checkpoints, "best_align_module_5.pt")
    save_dir_prefix = os.path.join(checkpoints, "best_prefix_adapter_5.pt")
    plot_dir = os.path.join(project_root, "plots_5")
    os.makedirs(plot_dir, exist_ok=True)

    #test
    result_root = os.path.join(project_root, "results")
    results_path = os.path.join(result_root, "test_predictions_5.txt")
    sample_root = os.path.join(result_root, "samples_5")
    metrics_path = os.path.join(result_root, "evaluation_metrics_5.txt")
    samples = 5
cfg = Config()
