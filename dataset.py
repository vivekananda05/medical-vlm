import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class IUXRayReportDataset(Dataset):
    """IU X-Ray dataset for a specific split (train/val/test)."""
    def __init__(self, root, split="train", image_size=224, max_text_len=256, tokenizer=None):
        """
        Args:
            root (str): Root directory containing 'train', 'val', 'test' folders.
            split (str): One of ['train', 'val', 'test'].
            image_size (int): Image resize dimension.
            max_text_len (int): Max length for text tokenization.
            tokenizer: Optional tokenizer for text processing.
        """
        self.split = split
        self.image_dir = os.path.join(root, split, "images")
        self.report_file = os.path.join(root, split, "reports.tsv")

        if not os.path.exists(self.report_file):
            raise FileNotFoundError(f"Missing reports.tsv in {self.report_file}")

        # Load image-report pairs
        self.samples = []
        with open(self.report_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    img_name, report = parts
                    self.samples.append((img_name, report))

        # Image transformation
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, report = self.samples[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        if self.tokenizer is not None:
            tok = self.tokenizer(report, truncation=True, padding="max_length",
                                 max_length=self.max_text_len, return_tensors="pt")
            input_ids = tok["input_ids"].squeeze(0)
            attention_mask = tok["attention_mask"].squeeze(0)
        else:
            input_ids, attention_mask = None, None

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "text": report
        }


def get_dataloader(root, split="train", tokenizer=None, batch_size=16, image_size=224, max_text_len=256):
    
    assert split in ["train", "val", "test"], "split must be one of ['train', 'val', 'test']"

    dataset = IUXRayReportDataset(root, split, image_size, max_text_len, tokenizer)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,  
        num_workers=4,
        pin_memory=True
    )

    print(f"Loaded {split} dataset -> {len(dataset)} samples | Batch size: {batch_size}")
    return dataloader


