"""
PressureVision AI - Backbone Feature Extraction and Caching
============================================================
Extracts 768-D feature representations from ConvNeXt-Tiny and Swin-Tiny
backbones for train, val, and test splits (both clean and clinical augmentations).
"""

import os
import sys

# Ensure root directory is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from model import PressureVisionAI
from inference import get_device, IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE

sys.stdout.reconfigure(encoding="utf-8")


def get_transforms():
    # 1. Deterministic clean inference/eval transform
    clean_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    # 2. Clinically safe augmentations (anatomically neutral, preserves diagnostic color/edges)
    # Aug 1: Horizontal flip
    hflip_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    # Aug 2: Mild clockwise rotation (+8 deg) with neutral border fill
    rot_pos_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomRotation(degrees=(8, 8), interpolation=InterpolationMode.BILINEAR, fill=(128, 128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    # Aug 3: Mild counter-clockwise rotation (-8 deg) with neutral border fill
    rot_neg_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomRotation(degrees=(-8, -8), interpolation=InterpolationMode.BILINEAR, fill=(128, 128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    # Aug 4: Subtle clinical lighting/contrast variation (avoids shifting erythema to necrosis)
    jitter_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return {
        "clean": clean_transform,
        "hflip": hflip_transform,
        "rot_pos": rot_pos_transform,
        "rot_neg": rot_neg_transform,
        "jitter": jitter_transform
    }


def extract_split(model, df, transform, device, split_name, batch_size=16):
    model.eval()
    convnext_feats = []
    swin_feats = []
    labels = []
    paths = []

    print(f"Extracting {split_name} ({len(df)} samples, batch size {batch_size})...")
    t0 = time.time()

    images_batch = []
    labels_batch = []
    paths_batch = []

    with torch.no_grad():
        for idx, row in df.iterrows():
            img_path = row["path"]
            label = int(row["label"])
            try:
                img = Image.open(img_path).convert("RGB")
                tensor = transform(img)
                images_batch.append(tensor)
                labels_batch.append(label)
                paths_batch.append(img_path)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue

            if len(images_batch) == batch_size or idx == len(df) - 1:
                batch_tensors = torch.stack(images_batch).to(device)
                fc = model.convnext(batch_tensors)
                fs = model.swin(batch_tensors)

                convnext_feats.append(fc.cpu())
                swin_feats.append(fs.cpu())
                labels.extend(labels_batch)
                paths.extend(paths_batch)

                images_batch.clear()
                labels_batch.clear()
                paths_batch.clear()

    elapsed = time.time() - t0
    all_fc = torch.cat(convnext_feats, dim=0)
    all_fs = torch.cat(swin_feats, dim=0)
    all_labels = torch.tensor(labels, dtype=torch.long)

    print(f"  Finished {split_name} in {elapsed:.1f}s: ConvNeXt {all_fc.shape}, Swin {all_fs.shape}")
    return {
        "convnext": all_fc,
        "swin": all_fs,
        "labels": all_labels,
        "paths": paths
    }


def main():
    os.makedirs("experiments/features", exist_ok=True)
    device, dev_info = get_device()
    print(f"Using device: {dev_info['device_str']}")

    # Load trained model to get fine-tuned backbones
    print("Loading baseline model backbones from models/pressurevision_exp3_best.pth...")
    ckpt = torch.load("models/pressurevision_exp3_best.pth", map_location=device)
    model = PressureVisionAI()
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    train_df = pd.read_csv("experiments/splits/train_split.csv")
    val_df = pd.read_csv("experiments/splits/val_split.csv")
    test_df = pd.read_csv("experiments/splits/test_split.csv")

    t_dict = get_transforms()

    # 1. Clean validation and test features (always deterministic)
    if not os.path.exists("experiments/features/val_features.pt"):
        val_data = extract_split(model, val_df, t_dict["clean"], device, "Validation (clean)")
        torch.save(val_data, "experiments/features/val_features.pt")
    else:
        print("experiments/features/val_features.pt already exists, skipping.")

    if not os.path.exists("experiments/features/test_features.pt"):
        test_data = extract_split(model, test_df, t_dict["clean"], device, "Test (clean)")
        torch.save(test_data, "experiments/features/test_features.pt")
    else:
        print("experiments/features/test_features.pt already exists, skipping.")

    # 2. Train clean features
    if not os.path.exists("experiments/features/train_clean_features.pt"):
        train_clean = extract_split(model, train_df, t_dict["clean"], device, "Train (clean)")
        torch.save(train_clean, "experiments/features/train_clean_features.pt")
    else:
        print("experiments/features/train_clean_features.pt already exists, skipping.")

    # 3. Train augmented views for Experiment B
    for aug_key in ["hflip", "rot_pos", "rot_neg", "jitter"]:
        target_file = f"experiments/features/train_{aug_key}_features.pt"
        if not os.path.exists(target_file):
            aug_data = extract_split(model, train_df, t_dict[aug_key], device, f"Train ({aug_key})")
            torch.save(aug_data, target_file)
        else:
            print(f"{target_file} already exists, skipping.")

    print("\nFeature extraction completed successfully! Saved all tensors in experiments/features/")


if __name__ == "__main__":
    main()
