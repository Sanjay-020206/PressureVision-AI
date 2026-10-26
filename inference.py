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
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

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


def find_checkpoint_path() -> str:
    """
    Locates trained checkpoint: checks local models directory first,
    then fallback directory, and falls back to downloading from Hugging Face Hub.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))

    candidates = [
        os.path.join(base_dir, "models", "pressurevision_exp3_best.pth"),
        os.path.join(base_dir, "PressureVisionAI_Experiment3_BestModel", "pressurevision_exp3_best.pth")
    ]

    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)

    repo_id = os.getenv(
        "HF_MODEL_REPO_ID",
        "Sanjay020206/PressureVision-AI"
    )

    try:
        return hf_hub_download(
            repo_id=repo_id,
            filename="pressurevision_exp3_best.pth",
            repo_type="model"
        )
    except Exception as e:
        raise RuntimeError(
            f"Unable to download the PressureVision AI checkpoint "
            f"from Hugging Face repository {repo_id}: {e}"
        ) from e


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
        try:
            checkpoint_path = find_checkpoint_path()
        except Exception as e:
            return None, device, device_info, str(e)
        
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        err_msg = (
            "Model checkpoint not found. Please ensure 'pressurevision_exp3_best.pth' "
            "is placed inside the 'models/' directory or accessible via Hugging Face."
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


# =============================================================================
# Grad-CAM Visual Explainability Engine
# =============================================================================

class PressureVisionGradCAM:
    """
    Grad-CAM engine for the PressureVision AI hybrid classifier.
    Computes gradient-weighted class activation maps from the final spatial stage
    of the ConvNeXt branch (model.convnext.stages[-1]) for the selected class logit
    of the complete hybrid model.
    """
    def __init__(
        self,
        model: PressureVisionAI,
        target_layer: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
        transform: Optional[transforms.Compose] = None
    ):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)
        self.model.eval()
        
        # Target layer: final spatial feature representation of ConvNeXt-Tiny (Stage 3)
        self.target_layer = target_layer or self.model.convnext.stages[-1]
        self.transform = transform or get_inference_transform()
        
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._hooks: List[Any] = []

    def _save_activation(self, module: torch.nn.Module, input: Any, output: torch.Tensor):
        self._activations = output

    def _save_gradient(self, grad: torch.Tensor):
        self._gradients = grad

    def _register_hooks(self):
        self._cleanup_hooks()
        handle = self.target_layer.register_forward_hook(self._save_activation)
        self._hooks.append(handle)

    def _cleanup_hooks(self):
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks.clear()

    def generate(
        self,
        image: Image.Image,
        target_class: Optional[Union[int, str]] = None,
        overlay_opacity: float = 0.45,
        colormap_name: str = "turbo"
    ) -> Dict[str, Any]:
        """
        Calculates Grad-CAM for a given PIL image.
        
        Args:
            image: Input PIL Image in RGB format.
            target_class: Class name or integer index to explain. If None, explains predicted class.
            overlay_opacity: Heatmap overlay blending weight (0.0 to 1.0).
            colormap_name: Colormap name ('turbo', 'jet', 'viridis', etc.).
            
        Returns:
            Dictionary containing prediction data, raw heatmap, colored heatmap PIL, and overlay PIL.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        orig_w, orig_h = image.size
        orig_np = np.array(image)

        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        self._register_hooks()
        try:
            self.model.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(True):
                logits = self.model(input_tensor)
                probabilities = F.softmax(logits, dim=1).squeeze(0)
                pred_idx = torch.argmax(probabilities).item()

                if target_class is None:
                    target_idx = pred_idx
                elif isinstance(target_class, str):
                    if target_class in RAW_CLASS_NAMES:
                        target_idx = RAW_CLASS_NAMES.index(target_class)
                    elif target_class in DISPLAY_CLASS_NAMES:
                        target_idx = DISPLAY_CLASS_NAMES.index(target_class)
                    else:
                        raise ValueError(f"Unknown class '{target_class}'. Must be one of {DISPLAY_CLASS_NAMES}")
                else:
                    target_idx = int(target_class)
                    if target_idx < 0 or target_idx >= len(RAW_CLASS_NAMES):
                        raise IndexError(f"Target index {target_idx} out of range [0, 6]")

                act = self._activations
                if act is None:
                    raise RuntimeError("Failed to capture activations from target layer.")

                act.register_hook(self._save_gradient)

                # Differentiate unnormalized class logit score
                class_score = logits[0, target_idx]
                class_score.backward(retain_graph=False)

            grads = self._gradients
            if grads is None:
                raise RuntimeError("Failed to capture gradients for target layer.")

            # Spatial Global Average Pooling of gradients (alpha importance weights)
            # Shape of act and grads: [1, 768, 7, 7]
            weights = torch.mean(grads, dim=(2, 3), keepdim=True)  # [1, 768, 1, 1]

            # Weighted sum of feature maps
            cam = torch.sum(weights * act, dim=1, keepdim=True)  # [1, 1, 7, 7]

            # Apply ReLU to retain only positive contributions
            cam = F.relu(cam)

            cam_2d = cam.squeeze().detach().cpu().numpy()

            # Normalize to [0, 1]
            c_min, c_max = float(np.min(cam_2d)), float(np.max(cam_2d))
            if c_max - c_min > 1e-8:
                norm_cam = (cam_2d - c_min) / (c_max - c_min)
            else:
                norm_cam = np.zeros_like(cam_2d)

            # Resize to original image size
            heatmap_pil = Image.fromarray((norm_cam * 255.0).astype(np.uint8)).resize(
                (orig_w, orig_h),
                resample=Image.BILINEAR
            )
            heatmap_resized = np.array(heatmap_pil, dtype=np.float32) / 255.0

            # Colormap
            try:
                cmap = plt.colormaps.get_cmap(colormap_name)
            except AttributeError:
                cmap = cm.get_cmap(colormap_name)

            colored_heatmap = cmap(heatmap_resized)[:, :, :3]
            colored_heatmap_uint8 = np.uint8(np.clip(colored_heatmap * 255.0, 0, 255))

            # Blend with original image
            alpha = float(np.clip(overlay_opacity, 0.0, 1.0))
            overlay_uint8 = np.uint8(np.clip((1.0 - alpha) * orig_np + alpha * colored_heatmap_uint8, 0, 255))

            colored_heatmap_pil = Image.fromarray(colored_heatmap_uint8)
            overlay_pil = Image.fromarray(overlay_uint8)

            prob_list = probabilities.detach().cpu().numpy().tolist()

            return {
                "predicted_class": DISPLAY_CLASS_NAMES[pred_idx],
                "predicted_class_raw": RAW_CLASS_NAMES[pred_idx],
                "predicted_index": pred_idx,
                "confidence": float(prob_list[pred_idx]),
                "confidence_percentage": float(prob_list[pred_idx]) * 100.0,
                "target_class": DISPLAY_CLASS_NAMES[target_idx],
                "target_class_raw": RAW_CLASS_NAMES[target_idx],
                "target_index": target_idx,
                "probabilities": {DISPLAY_CLASS_NAMES[i]: float(p) for i, p in enumerate(prob_list)},
                "heatmap_raw": norm_cam,
                "heatmap_resized": heatmap_resized,
                "colored_heatmap_pil": colored_heatmap_pil,
                "overlay_pil": overlay_pil,
                "activation_shape": list(act.shape),
                "gradient_shape": list(grads.shape)
            }

        finally:
            self._cleanup_hooks()
            self._activations = None
            self._gradients = None
            self.model.zero_grad(set_to_none=True)


def generate_gradcam(
    model: PressureVisionAI,
    image: Image.Image,
    device: Optional[torch.device] = None,
    target_class: Optional[Union[int, str]] = None,
    overlay_opacity: float = 0.45,
    colormap_name: str = "turbo"
) -> Dict[str, Any]:
    """
    Convenience function to generate a Grad-CAM explanation for an image.
    """
    if device is None:
        device = next(model.parameters()).device
    engine = PressureVisionGradCAM(model=model, device=device)
    return engine.generate(
        image=image,
        target_class=target_class,
        overlay_opacity=overlay_opacity,
        colormap_name=colormap_name
    )

