"""
PressureVision AI - Standalone Local Grad-CAM Runner
=====================================================
Usage:
    python run_gradcam.py
    python run_gradcam.py --image "Dataset/archive/pressure_ulcers_expanded/Stage_III/healing-pressure-ulcer-0006.jpg"
    python run_gradcam.py --image "path/to/wound.jpg" --target_class "Stage_III" --output "gradcam_result.png"
"""

import os
import sys
import argparse
import time
import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless and CLI execution
import matplotlib.pyplot as plt

from inference import (
    load_model,
    generate_gradcam,
    DISPLAY_CLASS_NAMES,
    RAW_CLASS_NAMES,
    CLASS_DESCRIPTIONS
)


def find_default_sample() -> str:
    candidates = [
        r"Dataset\archive\pressure_ulcers_expanded\Stage_III\healing-pressure-ulcer-0006.jpg",
        r"Dataset\archive\pressure_ulcers_expanded\Stage_II\32.jpg",
        r"Dataset\archive\pressure_ulcers_expanded\Stage_IV\pressure-ulcer-0062.jpg",
        r"Dataset\archive\pressure_ulcers_expanded\SDTI\12.jpg",
        r"Dataset\archive\pressure_ulcers_expanded\Unstageable\1.jpg",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    # Search in Dataset directory
    if os.path.isdir("Dataset"):
        for root, dirs, files in os.walk("Dataset"):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    return os.path.join(root, f)
    raise FileNotFoundError("No default test image found in Dataset. Please specify --image <path>.")


def main():
    parser = argparse.ArgumentParser(description="PressureVision AI - Local Grad-CAM Visual Explainability CLI")
    parser.add_argument("--image", type=str, default=None, help="Path to input ulcer RGB image")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--target_class", type=str, default=None, help="Target class name or index to explain (default: predicted class)")
    parser.add_argument("--opacity", type=float, default=0.45, help="Heatmap blend opacity [0.0 - 1.0] (default: 0.45)")
    parser.add_argument("--colormap", type=str, default="turbo", help="Colormap name (default: turbo)")
    parser.add_argument("--output", type=str, default="gradcam_local_result.png", help="Path to save 3-panel visualization")
    args = parser.parse_args()

    print("=" * 70)
    print("  PRESSUREVISION AI -- LOCAL GRAD-CAM EXPLAINABILITY ENGINE")
    print("=" * 70)

    # 1. Image path resolution
    image_path = args.image
    if not image_path:
        image_path = find_default_sample()
        print(f"[*] No --image specified. Using default sample: {image_path}")

    if not os.path.isfile(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)

    # 2. Model loading
    print("[*] Loading trained hybrid model...")
    model, device, device_info, err = load_model(args.checkpoint)
    if err or model is None:
        print(f"[ERROR] Failed to load model: {err}")
        sys.exit(1)

    print(f"[+] Loaded successfully on device: {device_info['device_str']}")
    print(f"[+] Target layer: ConvNeXt Stage 3 (model.convnext.stages[-1])")

    # 3. Load image
    orig_img = Image.open(image_path).convert("RGB")
    w, h = orig_img.size
    print(f"[+] Loaded input image: {image_path} ({w} x {h} px)")

    # 4. Generate Grad-CAM
    print("[*] Computing forward pass and gradient backpropagation...")
    t0 = time.time()
    result = generate_gradcam(
        model=model,
        image=orig_img,
        device=device,
        target_class=args.target_class,
        overlay_opacity=args.opacity,
        colormap_name=args.colormap
    )
    elapsed_ms = (time.time() - t0) * 1000.0

    print("-" * 70)
    print(f"Predicted Class : {result['predicted_class']} ({result['confidence_percentage']:.2f}% confidence)")
    print(f"Explaining Class: {result['target_class']} (Index {result['target_index']})")
    print(f"Inference Time  : {elapsed_ms:.1f} ms")
    print(f"Activation Shape: {result['activation_shape']}")
    print(f"Gradient Shape  : {result['gradient_shape']}")
    print("-" * 70)

    # 5. Display 7-Class Probability Table
    print("\nComplete 7-Class Probability Spectrum:")
    print("-" * 60)
    print(f"{'Class Name':<16} | {'Probability':<12} | {'Explained':<10} | {'Predicted'}")
    print("-" * 60)
    for c_name in DISPLAY_CLASS_NAMES:
        prob = result["probabilities"][c_name]
        is_target = "[YES]" if c_name == result["target_class"] else ""
        is_pred = "[PRED]" if c_name == result["predicted_class"] else ""
        print(f"{c_name:<16} | {prob * 100.0:6.2f}%      | {is_target:<10} | {is_pred}")
    print("-" * 60)

    # 6. Save 3-panel visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#0B0F19")
    
    # Panel 1: Original Image
    axes[0].imshow(orig_img)
    axes[0].set_title(f"Original Input\n({w} x {h} px)", color="#38BDF8", fontsize=12, fontweight="bold", pad=10)
    axes[0].axis("off")

    # Panel 2: Colored Heatmap
    axes[1].imshow(result["colored_heatmap_pil"])
    axes[1].set_title("ConvNeXt Grad-CAM Heatmap\n(Activation [0, 1])", color="#38BDF8", fontsize=12, fontweight="bold", pad=10)
    axes[1].axis("off")

    # Panel 3: Blended Overlay
    axes[2].imshow(result["overlay_pil"])
    axes[2].set_title(
        f"Grad-CAM Overlay (a={args.opacity})\nExplaining: {result['target_class']}",
        color="#F8FAFC",
        fontsize=12,
        fontweight="bold",
        pad=10
    )
    axes[2].axis("off")

    plt.suptitle(
        f"PressureVision AI -- Prediction: {result['predicted_class']} ({result['confidence_percentage']:.1f}%) | Latency: {elapsed_ms:.1f}ms",
        color="#F8FAFC",
        fontsize=14,
        fontweight="bold",
        y=1.02
    )
    plt.tight_layout()

    out_path = os.path.abspath(args.output)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    print(f"\n[+] 3-Panel visualization saved to: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
