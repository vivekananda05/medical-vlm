import os
import random
import torch
import matplotlib.pyplot as plt
#from torchvision.utils import make_grid
#from transformers import AutoTokenizer
from PIL import Image
from dataset import IUXRayReportDataset
# Assuming the IUXRayReportDataset class is already defined above

# ---------------------------------------------------------
# Visualization function
# ---------------------------------------------------------
def visualize_iuxray_dataset(dataset, num_samples=4):
    """Visualize image–report pairs safely."""
    total = len(dataset)
    if total == 0:
        raise ValueError("Dataset is empty! Check reports_all.tsv and image paths.")
    num_samples = min(num_samples, total)

    indices = random.sample(range(total), num_samples)
    samples = [dataset[i] for i in indices]


    fig, axes = plt.subplots(1, num_samples, figsize=(4*num_samples, 4))
    if num_samples == 1:
        axes = [axes]

    for i, sample in enumerate(samples):
        image = sample["image"].permute(1, 2, 0)
        image = image * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
        image = torch.clamp(image, 0, 1)

        axes[i].imshow(image)
        axes[i].axis("off")
        axes[i].set_title(f"Report:\n{sample['text'][:200]}...", fontsize=10, wrap=True)

    plt.tight_layout()
    plt.savefig("/mnt/DATA1/vivekananda/medical_alignvlm/visualize.png")


# ---------------------------------------------------------
# Example usage
# ---------------------------------------------------------
if __name__ == "__main__":
    # Path to IU X-Ray dataset folder (must contain images/ and reports_all.tsv)
    root = "/mnt/DATA1/vivekananda/DATA/iux_data"
    
    reports_file = os.path.join(root, "reports_clean.tsv")

    with open(reports_file, "r", encoding="utf-8") as f:
       lines = [line.strip() for line in f.readlines()]

    print("Total lines:", len(lines))
    print("First 3 lines:")
    for l in lines[:3]:
       print(l)


    # Create dataset instance
    dataset = IUXRayReportDataset(root=root, image_size=224, max_text_len=256)

    print(f"Total samples: {len(dataset)}")

    # Visualize random 4 samples
    visualize_iuxray_dataset(dataset, num_samples=2)
