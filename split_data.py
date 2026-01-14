import os
import shutil
import pandas as pd

def split_and_save_dataset(root, image_dir, ratios=(0.8, 0.1, 0.1), seed=42):

    reports_file = os.path.join(root, "reports_clean.tsv")
    if not os.path.exists(reports_file):
        raise FileNotFoundError(f"Expected reports_clean.tsv in {root}")

    # Load the TSV file (no header, just image name and report)
    df = pd.read_csv(reports_file, sep="\t", header=None, names=["image", "report"])
    df = df[df["report"].notna() & (df["report"].str.strip() != "")]
    total = len(df)
    print(f" Total valid samples: {total}")

    # Shuffle dataset for random split
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    n_train = int(ratios[0] * total)
    n_val = int(ratios[1] * total)

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]

    print(f"📊 Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Define destination folders
    splits = {"train": train_df, "val": val_df, "test": test_df}
    for split, split_df in splits.items():
        split_dir = os.path.join(root, split)
        image_dest = os.path.join(split_dir, "images")
        os.makedirs(image_dest, exist_ok=True)

        # Copy images from separate directory
        missing = 0
        for img_name in split_df["image"]:
            src = os.path.join(image_dir, img_name)
            dst = os.path.join(image_dest, img_name)
            if os.path.exists(src):
                shutil.copy2(src, dst)
            else:
                missing += 1

        # Save reports.tsv file for this split
        report_path = os.path.join(split_dir, "reports.tsv")
        split_df.to_csv(report_path, sep="\t", index=False, header=False)

        print(f"📁 Saved {len(split_df)} samples to {split_dir}/ (missing: {missing})")

    print("✅ Dataset successfully split and saved to train/val/test folders.")


if __name__ == "__main__":
    # Example usage
    root_dir = "/mnt/DATA1/vivekananda/DATA/iux_data"        # Folder containing reports_all.tsv
    image_dir = "/mnt/DATA1/vivekananda/DATA/iux_data/NLMCXR_png"  # Folder containing all images
    split_and_save_dataset(root_dir, image_dir)
