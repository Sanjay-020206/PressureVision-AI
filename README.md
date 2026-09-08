# PressureVision AI: Pressure Ulcer Stage Classification using a Hybrid Deep Learning Model

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-ff4b4b.svg)](https://streamlit.io/)
[![Timm](https://img.shields.io/badge/timm-0.9+-brightgreen.svg)](https://github.com/huggingface/pytorch-image-models)

---

## 1. Project Overview

**PressureVision AI** is an advanced deep learning decision-support platform engineered for automated, non-invasive classification of pressure ulcer severity from standard photographic RGB wound images. The system harnesses a **dual-backbone hybrid neural architecture** uniting modern depthwise convolutional representations with hierarchical vision transformers.

The application is deployed as a high-performance **Streamlit web application** featuring a dark glassmorphic clinical dashboard that evaluates dermatological photographs in real-time, displaying stage predictions, model confidence scores, dynamic backbone gating weights, and complete 7-class probability spectrums.

---

## 2. Model Architecture

PressureVision AI leverages two complementary neural network backbones to capture both fine-grained surface dermatological features and long-range spatial wound contexts:

```
                      ┌────────────────────────┐
                      │    RGB Wound Image     │
                      │       (224 × 224)      │
                      └───────────┬────────────┘
                                  │
                   ┌──────────────┴──────────────┐
                   ▼                             ▼
       ┌───────────────────────┐     ┌───────────────────────┐
       │     ConvNeXt-Tiny     │     │       Swin-Tiny       │
       │    (CNN Backbone)     │     │  (Hierarchical ViT)   │
       │     Output: 768-d     │     │     Output: 768-d     │
       └───────────┬───────────┘     └───────────┬───────────┘
                   ▼                             ▼
       ┌───────────────────────┐     ┌───────────────────────┐
       │  ConvNeXt Projection  │     │    Swin Projection    │
       │  (Linear + LayerNorm) │     │  (Linear + LayerNorm) │
       │     Output: 512-d     │     │     Output: 512-d     │
       └───────────┬───────────┘     └───────────┬───────────┘
                   │                             │
                   └──────────────┬──────────────┘
                                  ▼
                   ┌─────────────────────────────┐
                   │ Adaptive Gated Fusion (MLP) │
                   │       α ∈ [0, 1]^512        │
                   └──────────────┬──────────────┘
                                  │
                                  ▼
                   ┌─────────────────────────────┐
                   │     Classification Head     │
                   │ (LayerNorm + GELU + Dropout)│
                   └──────────────┬──────────────┘
                                  │
                                  ▼
                   ┌─────────────────────────────┐
                   │    7-Class Output Logits    │
                   └─────────────────────────────┘
```

### Backbone Details:
- **ConvNeXt-Tiny (`convnext_tiny.fb_in22k_ft_in1k`)**:
  - ImageNet-pretrained convolutional architecture with $7\times 7$ depthwise separable convolutions and inverted bottleneck design.
  - Specializes in extracting local texture patterns, skin erythema, granulation edges, and necrotic slough margins.
  - Feature representation dimension: $768$.
- **Swin-Tiny (`swin_tiny_patch4_window7_224.ms_in1k`)**:
  - Hierarchical Vision Transformer with shifted window self-attention ($224\times 224$ resolution).
  - Specializes in global contextual relationships, anatomical surrounding skin patterns, and multi-scale wound geometry.
  - Feature representation dimension: $768$.

---

## 3. Adaptive Feature Fusion

Rather than applying naive concatenation or element-wise addition, PressureVision AI employs an **Adaptive Gated Feature Fusion** mechanism.

1. Both backbone feature vectors are projected into a unified embedding space:
   $$\mathbf{z}_{\text{convnext}} = \text{LayerNorm}(\mathbf{W}_c \mathbf{x}_{\text{convnext}} + \mathbf{b}_c) \in \mathbb{R}^{512}$$
   $$\mathbf{z}_{\text{swin}} = \text{LayerNorm}(\mathbf{W}_s \mathbf{x}_{\text{swin}} + \mathbf{b}_s) \in \mathbb{R}^{512}$$

2. The projected vectors are concatenated $[\mathbf{z}_{\text{convnext}} \,\|\, \mathbf{z}_{\text{swin}}] \in \mathbb{R}^{1024}$ and passed through a learnable MLP gating network:
   $$\boldsymbol{\alpha} = \sigma\left(\mathbf{W}_2 \cdot \text{GELU}(\mathbf{W}_1 [\mathbf{z}_{\text{convnext}} \,\|\, \mathbf{z}_{\text{swin}}] + \mathbf{b}_1) + \mathbf{b}_2\right) \in (0, 1)^{512}$$

3. The final fused representation $\mathbf{F} \in \mathbb{R}^{512}$ is computed via channel-wise adaptive gating:
   $$\mathbf{F} = \boldsymbol{\alpha} \odot \mathbf{z}_{\text{convnext}} + (1 - \boldsymbol{\alpha}) \odot \mathbf{z}_{\text{swin}}$$

This enables the model to dynamically prioritize convolutional texture features or transformer contextual features on a per-channel and per-image basis.

---

## 4. 7-Class Pressure Ulcer Staging System

The model classifies wound images into 7 standard staging classes:

| Class Index | Internal Label | UI Display Name | Clinical Description |
|:---:|:---|:---|:---|
| `0` | `Invalid` | **Invalid** | Non-ulcer artifact, background, clothing, or non-wound photograph. |
| `1` | `SDTI` | **SDTI** | Suspected Deep Tissue Injury: Persistent non-blanchable deep red/purple discoloration. |
| `2` | `Stage_I` | **Stage I** | Intact skin with a localized area of non-blanchable erythema (redness). |
| `3` | `Stage_II` | **Stage II** | Partial-thickness loss of skin with exposed dermis. Viable pink/red wound bed. |
| `4` | `Stage_III` | **Stage III** | Full-thickness skin loss with visible subcutaneous adipose tissue and granulation. |
| `5` | `Stage_IV` | **Stage IV** | Full-thickness skin and tissue loss with exposed fascia, muscle, tendon, or bone. |
| `6` | `Unstageable` | **Unstageable** | Full-thickness tissue loss obscured by slough or eschar. |

---

## 5. Model Parameters & Performance

- **Total Architecture Parameters**: $56,657,121$
- **Trained Model Checkpoint**: `pressurevision_exp3_best.pth` (Selected by lowest validation loss: $0.6792$)
- **Official Untouched Test Set Evaluation**:
  - **Accuracy**: `79.53%` (272 / 342 correct predictions)
  - **Macro Precision**: `81.89%`
  - **Macro Recall**: `81.28%`
  - **Macro F1-Score**: `80.99%`
  - **Weighted F1-Score**: `79.44%`

*Note: All performance figures reflect the project's untouched benchmark test set. No claims of clinical accuracy or regulatory medical validation are made.*

---

## 6. Directory & File Structure

```
PressureVisionAI/
│
├── app.py                     # Main Streamlit web application (UI, dashboard, visualizations)
├── model.py                   # PyTorch PressureVisionAI architecture definition
├── inference.py               # Cached model loader, preprocessing, prediction pipeline
├── requirements.txt           # Minimal exact Python dependencies
├── README.md                  # Comprehensive project documentation
│
├── models/
│   ├── pressurevision_exp3_best.pth      # Trained checkpoint (56.6M params)
│   └── pressurevision_exp3_config.json  # Architecture & training hyperparameters
│
└── Dataset/                   # Benchmark dataset reference and test samples
    └── archive/
        └── pressure_ulcers_expanded/
            ├── Stage_I/
            ├── Stage_II/
            ├── Stage_III/
            ├── Stage_IV/
            ├── SDTI/
            ├── Unstageable/
            └── Invalid/
```

---

## 7. Installation & Environment Setup

### Prerequisites
- Python 3.9, 3.10, or 3.11
- pip package manager

### Step 1: Clone or Navigate to Project Directory
```bash
cd E:\Pressure_ulcer_class
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
python -m venv venv

# On Windows:
.\venv\Scripts\activate

# On Linux / macOS:
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 8. Model Checkpoint Placement

Ensure that the trained model checkpoint is located at:
```
models/pressurevision_exp3_best.pth
```
*(The application also automatically falls back to `PressureVisionAI_Experiment3_BestModel/pressurevision_exp3_best.pth` if present).*

---

## 9. Running the Streamlit Web Application

To start the local web dashboard:

```bash
streamlit run app.py
```

Streamlit will launch a local server accessible via your browser at:
`http://localhost:8501`

---

## 10. Inference Pipeline & Image Preprocessing

1. **Input**: User uploads an RGB photograph (`.jpg`, `.jpeg`, `.png`, `.webp`) or selects a sample.
2. **RGB Conversion**: Automatically converts single-channel (grayscale), RGBA, or CMYK images to standard 3-channel RGB.
3. **Resizing**: Deterministically resized to $(224 \times 224)$ pixels.
4. **Tensor Normalization**: Converted to float tensor and normalized via standard ImageNet parameters:
   - Mean: $[0.485, 0.456, 0.406]$
   - Standard Deviation: $[0.229, 0.224, 0.225]$
5. **Forward Pass**: Evaluated under `torch.no_grad()` in `eval()` mode.
6. **Softmax Output**: Computes true mathematical probability distribution over all 7 staging classes ($\sum P_i = 100\%$).

---

## 11. Hardware Requirements & Device Detection

- **Device Auto-Detection**: The inference engine detects CUDA-enabled NVIDIA GPUs via `torch.cuda.is_available()`. If detected, weights and tensors are processed on GPU memory; otherwise, CPU execution is used seamlessly.
- **Minimum Requirements**:
  - **RAM**: 4 GB RAM minimum (8 GB recommended)
  - **Processor**: Multi-core x86-64 CPU (Intel i5/i7 or AMD Ryzen)
  - **GPU (Optional)**: NVIDIA GPU with CUDA 11.8+ for sub-50ms inference latency.

---

## 12. Explainability & Grad-CAM Future Integration

> **Note**: Grad-CAM interpretability is currently designated as **Coming Soon / Future Implementation**. Simulated or artificial heatmaps are strictly omitted to maintain diagnostic integrity.

### Integration Architecture:
The `model.py` architecture is already prepared with hook access points for future Grad-CAM integration:
- **ConvNeXt Hook**: `model.get_convnext_target_layer()` targets `convnext.stages[-1]` to visualize convolutional feature maps highlighting micro-textures, wound edges, and erythema margins.
- **Swin Hook**: `model.get_swin_target_layer()` targets `swin.layers[-1]` to visualize attention map weights across anatomical regions.

---

## 13. System Limitations

1. **Image Quality Sensitivity**: Extreme under-exposure, severe motion blur, or camera glare can degrade feature extraction accuracy.
2. **2D Photographic Representation**: 2D images do not directly capture depth or palpation data (e.g. assessing bone undermining in Stage IV).
3. **Class Imbalance**: Wound staging datasets historically exhibit variations in class frequency (e.g., SDTI vs Stage II).

---

## 14. Regulatory & Clinical Disclaimer

> ⚠️ **IMPORTANT NOTICE:**
> **PressureVision AI is an artificial intelligence decision-support tool developed for research, academic evaluation, and clinical workflow demonstration.**
> It is **not** a certified medical diagnostic device and is **not** cleared by any health regulatory authority (such as the FDA or CE-IVD). It must **never** be used as a replacement for clinical judgment, physical examination, or formal diagnosis by a certified medical doctor or wound care specialist.
