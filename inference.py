"""
PressureVision AI - Inference Pipeline
======================================
Inference engine for Pressure Ulcer Stage Classification:
- Automatic hardware device detection (CUDA GPU / CPU)
- Single-instance model loading with state-dict integrity verification
- Strict deterministic inference preprocessing (No training augmentations)
- Real softmax probability distribution across 7 clinical staging classes
"""

import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from typing import Dict, Any, Tuple, Optional, List

from model import PressureVisionAI


# Standard 7-Class definitions (Exact mapping and UI friendly display)
RAW_CLASS_NAMES: List[str] = [
    "Invalid",
    "SDTI",
    "Stage_I",
    "Stage_II",
    "Stage_III",
    "Stage_IV",
    "Unstageable"
]

DISPLAY_CLASS_NAMES: List[str] = [
    "Invalid",
    "SDTI",
    "Stage I",
    "Stage II",
    "Stage III",
    "Stage IV",
    "Unstageable"
]

CLASS_DESCRIPTIONS: Dict[str, str] = {
    "Invalid": "Non-pressure ulcer artifact, background, clothing, or invalid image.",
    "SDTI": "Suspected Deep Tissue Injury: Intact or non-intact skin with localized area of persistent non-blanchable deep red, maroon, or purple discoloration.",
    "Stage I": "Stage I: Intact skin with a localized area of non-blanchable erythema (redness).",
    "Stage II": "Stage II: Partial-thickness loss of skin with exposed dermis. Viable wound bed without visible adipose.",
    "Stage III": "Stage III: Full-thickness skin loss with visible adipose (subcutaneous fat) and granulation tissue.",
    "Stage IV": "Stage IV: Full-thickness skin and tissue loss with exposed or directly palpable fascia, muscle, tendon, ligament, or bone.",
    "Unstageable": "Unstageable: Full-thickness skin and tissue loss in which the extent of tissue damage cannot be confirmed because it is obscured by slough or eschar."
}

# ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = (224, 224)


def get_device() -> Tuple[torch.device, Dict[str, str]]:
    """
    Detects and returns the optimal computing device (CUDA GPU or CPU)
    along with hardware metadata.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        device_info = {
            "device": "CUDA",
            "device_str": f"CUDA ({gpu_name})",
            "gpu_name": gpu_name,
            "device_count": str(torch.cuda.device_count())
        }
    else:
        device = torch.device("cpu")
        device_info = {
            "device": "CPU",
            "device_str": "CPU",
            "gpu_name": "N/A",
            "device_count": "1"
        }
    return device, device_info


def find_checkpoint_path() -> Optional[str]:
    """
    Searches standard project paths for the trained checkpoint file.
    """
    candidates = [
        os.path.join(os.path.dirname(__file__), "models", "pressurevision_exp3_best.pth"),
        os.path.join("models", "pressurevision_exp3_best.pth"),
        os.path.join(os.path.dirname(__file__), "PressureVisionAI_Experiment3_BestModel", "pressurevision_exp3_best.pth"),
        os.path.join("PressureVisionAI_Experiment3_BestModel", "pressurevision_exp3_best.pth"),
        os.path.abspath("models/pressurevision_exp3_best.pth"),
        os.path.abspath("PressureVisionAI_Experiment3_BestModel/pressurevision_exp3_best.pth")
    ]
    
    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)
            
    return None


def get_inference_transform() -> transforms.Compose:
    """
    Strict inference preprocessing pipeline:
    1. Resize image to 224x224 (Bilinear)
    2. Convert PIL Image to PyTorch Tensor [0.0, 1.0]
    3. Normalize using standard ImageNet mean and std
    """
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def load_model(checkpoint_path: Optional[str] = None) -> Tuple[Optional[PressureVisionAI], torch.device, Dict[str, str], Optional[str]]:
    """
    Instantiates the PressureVisionAI architecture, loads the trained checkpoint
    with strict state dictionary verification, moves to the optimal device,
    and sets eval mode.
    
    Returns:
        (model, device, device_info, error_message)
    """
    device, device_info = get_device()
    
    if checkpoint_path is None:
        checkpoint_path = find_checkpoint_path()
        
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        err_msg = (
            "Model checkpoint not found. Please ensure 'pressurevision_exp3_best.pth' "
            "is placed inside the 'models/' directory."
        )
        return None, device, device_info, err_msg

    try:
        # Load checkpoint dictionary
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract model state dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Instantiate architecture
        model = PressureVisionAI(
            num_classes=7,
            fusion_dim=512,
            gate_hidden_dim=256,
            dropout=0.4
        )
        
        # Strict state dictionary loading to ensure 100% architectural compatibility
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        
        return model, device, device_info, None

    except Exception as e:
        err_msg = f"Failed to load model checkpoint: {str(e)}"
        return None, device, device_info, err_msg


def predict_image(
    model: PressureVisionAI,
    image: Image.Image,
    device: torch.device
) -> Dict[str, Any]:
    """
    Executes live model inference on an input RGB image:
    1. Validates and converts image to RGB
    2. Applies exact inference transform
    3. Adds batch dimension (1, 3, 224, 224) and moves to device
    4. Executes forward pass in torch.no_grad()
    5. Applies Softmax to produce real class probabilities
    6. Formats predicted class, confidence, and full probability distribution.
    
    Returns:
        Dictionary containing predicted class, confidence, raw probabilities,
        and gate attention weights.
    """
    if model is None:
        raise ValueError("Model is not loaded. Cannot perform inference.")
        
    # 1. Convert PIL image to RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    original_size = image.size  # (width, height)
    
    # 2. Apply deterministic inference preprocessing
    transform = get_inference_transform()
    tensor = transform(image).unsqueeze(0).to(device)  # Shape: (1, 3, 224, 224)
    
    # 3. Model execution
    model.eval()
    with torch.no_grad():
        fused_features, alpha, proj_convnext, proj_swin = model.forward_features(tensor)
        logits = model.classifier(fused_features)  # Shape: (1, 7)
        
        # Softmax probabilities
        probabilities = torch.softmax(logits, dim=1).squeeze(0)  # Shape: (7,)
        
        # Prediction
        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_idx].item()
        
        # Gate weight stats (average alpha weight for ConvNeXt)
        alpha_convnext_pct = alpha.mean().item() * 100.0
        alpha_swin_pct = 100.0 - alpha_convnext_pct

    # Format probability mapping
    prob_dict: Dict[str, float] = {}
    prob_list: List[Dict[str, Any]] = []
    
    for i, (raw_name, display_name) in enumerate(zip(RAW_CLASS_NAMES, DISPLAY_CLASS_NAMES)):
        p = probabilities[i].item()
        prob_dict[display_name] = p
        prob_list.append({
            "index": i,
            "raw_name": raw_name,
            "display_name": display_name,
            "probability": p,
            "percentage": p * 100.0,
            "is_predicted": (i == predicted_idx),
            "description": CLASS_DESCRIPTIONS.get(raw_name, "")
        })

    return {
        "predicted_class": DISPLAY_CLASS_NAMES[predicted_idx],
        "predicted_class_raw": RAW_CLASS_NAMES[predicted_idx],
        "predicted_index": predicted_idx,
        "confidence": confidence,
        "confidence_percentage": confidence * 100.0,
        "probabilities": prob_dict,
        "probabilities_list": prob_list,
        "original_dimensions": original_size,
        "gate_weights": {
            "convnext_weight_percent": alpha_convnext_pct,
            "swin_weight_percent": alpha_swin_pct
        }
    }
