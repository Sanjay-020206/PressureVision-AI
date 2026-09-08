"""
PressureVision AI - Model Architecture
=======================================
Hybrid Deep Learning Model for Pressure Ulcer Stage Classification:
- ConvNeXt-Tiny (CNN Backbone for local visual feature extraction)
- Swin-Tiny (Hierarchical Vision Transformer for long-range spatial context)
- Adaptive Gated Feature Fusion (Learnable gate balancing representation channels)
- Regularized 7-Class Classification Head

Total Model Parameters: 56,657,121
Trained Checkpoint: pressurevision_exp3_best.pth
"""

import torch
import torch.nn as nn
import timm
from typing import Tuple, Dict, Any, Optional


class PressureVisionAI(nn.Module):
    """
    PressureVision AI Hybrid Architecture:
    Combines ConvNeXt-Tiny and Swin-Tiny with an Adaptive Gated Feature Fusion mechanism.
    
    Attributes:
        convnext (nn.Module): ConvNeXt-Tiny backbone (768-dim output).
        swin (nn.Module): Swin-Tiny backbone (768-dim output).
        convnext_projection (nn.Sequential): Linear(768, 512) -> LayerNorm(512).
        swin_projection (nn.Sequential): Linear(768, 512) -> LayerNorm(512).
        gate (nn.Sequential): Linear(1024, 256) -> GELU -> Dropout -> Linear(256, 512) -> Sigmoid.
        classifier (nn.Sequential): LayerNorm(512) -> Dropout -> Linear(512, 256) -> GELU -> Dropout -> Linear(256, 7).
    """

    CLASS_NAMES = [
        "Invalid",
        "SDTI",
        "Stage_I",
        "Stage_II",
        "Stage_III",
        "Stage_IV",
        "Unstageable"
    ]
    
    DISPLAY_NAMES = [
        "Invalid",
        "SDTI",
        "Stage I",
        "Stage II",
        "Stage III",
        "Stage IV",
        "Unstageable"
    ]

    def __init__(
        self,
        num_classes: int = 7,
        fusion_dim: int = 512,
        gate_hidden_dim: int = 256,
        dropout: float = 0.4
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.fusion_dim = fusion_dim
        self.gate_hidden_dim = gate_hidden_dim
        self.dropout_rate = dropout

        # -------------------------------------------------------------
        # 1. Dual Feature Extraction Backbones
        # -------------------------------------------------------------
        # ConvNeXt-Tiny: Captures local textures, wound edges, and color patterns
        self.convnext = timm.create_model(
            "convnext_tiny.fb_in22k_ft_in1k",
            pretrained=False,
            num_classes=0
        )
        
        # Swin-Tiny: Captures broader spatial and contextual dependencies
        self.swin = timm.create_model(
            "swin_tiny_patch4_window7_224.ms_in1k",
            pretrained=False,
            num_classes=0
        )
        
        convnext_dim = 768
        swin_dim = 768

        # -------------------------------------------------------------
        # 2. Linear Projections with Layer Normalization
        # -------------------------------------------------------------
        self.convnext_projection = nn.Sequential(
            nn.Linear(convnext_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
        self.swin_projection = nn.Sequential(
            nn.Linear(swin_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )

        # -------------------------------------------------------------
        # 3. Adaptive Gated Feature Fusion (MLP Gate + Sigmoid)
        # -------------------------------------------------------------
        # Gate operates on concatenated features (512 + 512 = 1024)
        # to produce dynamic channel-wise weights alpha in [0, 1]^512
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, fusion_dim),
            nn.Sigmoid()
        )

        # -------------------------------------------------------------
        # 4. Final Classification Head
        # -------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract pooled representation vectors from both backbones.
        """
        feat_convnext = self.convnext(x)  # Shape: (B, 768)
        feat_swin = self.swin(x)          # Shape: (B, 768)
        return feat_convnext, feat_swin

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through backbones, projection layers, and adaptive fusion gate.
        
        Returns:
            fused_features: (B, 512) combined feature representation.
            alpha: (B, 512) learnable gating weights.
            proj_convnext: (B, 512) projected ConvNeXt features.
            proj_swin: (B, 512) projected Swin features.
        """
        feat_convnext, feat_swin = self.extract_features(x)
        
        proj_convnext = self.convnext_projection(feat_convnext)  # (B, 512)
        proj_swin = self.swin_projection(feat_swin)              # (B, 512)
        
        concat_features = torch.cat([proj_convnext, proj_swin], dim=1)  # (B, 1024)
        alpha = self.gate(concat_features)  # (B, 512)
        
        # Adaptive feature fusion formula:
        # F = alpha * ConvNeXt_features + (1 - alpha) * Swin_features
        fused_features = alpha * proj_convnext + (1.0 - alpha) * proj_swin  # (B, 512)
        
        return fused_features, alpha, proj_convnext, proj_swin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass producing raw unnormalized logits for the 7 classes.
        """
        fused_features, _, _, _ = self.forward_features(x)
        logits = self.classifier(fused_features)  # (B, 7)
        return logits

    def get_convnext_target_layer(self) -> Optional[nn.Module]:
        """
        Convenience hook target for future Grad-CAM integration (ConvNeXt stage 3 block).
        """
        if hasattr(self.convnext, "stages") and len(self.convnext.stages) > 0:
            return self.convnext.stages[-1]
        return None

    def get_swin_target_layer(self) -> Optional[nn.Module]:
        """
        Convenience hook target for future Grad-CAM integration (Swin layers block).
        """
        if hasattr(self.swin, "layers") and len(self.swin.layers) > 0:
            return self.swin.layers[-1]
        return None


def get_model_summary(model: nn.Module) -> Dict[str, Any]:
    """
    Returns parameter count and architectural metadata for PressureVision AI.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "model_name": "PressureVision AI",
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "backbones": ["ConvNeXt-Tiny", "Swin-Tiny"],
        "fusion_type": "Adaptive Gated Feature Fusion",
        "num_classes": 7,
        "input_resolution": (224, 224, 3),
        "official_test_accuracy": "79.53%"
    }
