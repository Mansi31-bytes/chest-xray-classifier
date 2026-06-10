import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import io

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chest X-Ray Pneumonia Classifier",
    page_icon="🫁",
    layout="centered"
)

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE      = 224
THRESHOLD     = 0.94
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
CLASS_NAMES   = ["NORMAL", "PNEUMONIA"]
MODEL_PATH    = "resnet50_best.pth"

# ── Model definition ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(512, 2)
    )
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    )
    model.eval()
    return model

# ── Transform ─────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ── Grad-CAM ──────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model        = model
        self.activations  = None
        self.gradients    = None
        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, "activations", o.detach())
        )
        target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "gradients", go[0].detach())
        )

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        inp = input_tensor.clone().requires_grad_(True)
        logits = self.model(inp)
        logits[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE),
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def overlay_cam(img_gray, cam, alpha=0.45):
    heatmap = cm.jet(cam)[:, :, :3]
    img_rgb = np.stack([img_gray] * 3, axis=2)
    return np.clip((1 - alpha) * img_rgb + alpha * heatmap, 0, 1)


def denormalize(tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img  = (tensor * std + mean).permute(1, 2, 0).numpy()
    return np.clip(img[:, :, 0], 0, 1)


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🫁 Chest X-Ray Pneumonia Classifier")
st.markdown(
    "Upload a chest X-ray and the model will predict **Normal** or **Pneumonia**, "
    "along with a **Grad-CAM heatmap** showing which lung regions influenced the decision."
)

st.info(
    "**Model:** ResNet-50 fine-tuned on 5,216 chest X-rays  |  "
    "**AUC-ROC:** 0.965  |  **Sensitivity:** 98.7%  |  **Specificity:** 76.5%  |  "
    f"**Decision threshold:** {THRESHOLD}"
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.markdown("""
This app classifies chest X-rays as **Normal** or **Pneumonia** using a fine-tuned ResNet-50.

**How it works:**
1. Upload a chest X-ray (JPEG/PNG)
2. The model predicts P(Pneumonia)
3. If P ≥ 0.94 → Pneumonia
4. Grad-CAM shows which regions drove the decision

**Dataset:** Kermany et al. (2018) — Guangzhou Women and Children's Medical Center

**Threshold = 0.94** was chosen to maximise sensitivity (catch more true pneumonia cases) at the cost of some specificity.

---
**Author:** Mansi  
M.Tech Bioengineering, IISc Bangalore
    """)

    st.header("Metrics")
    st.markdown("""
| Metric | Score |
|---|---|
| AUC-ROC | 0.965 |
| Sensitivity | 98.7% |
| Specificity | 76.5% |
| F1 Score | 0.928 |
| Accuracy | 90.4% |
    """)

# ── File uploader ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["jpg", "jpeg", "png"],
    help="Supports JPEG and PNG formats"
)

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")

    # Load model
    with st.spinner("Loading model..."):
        model = load_model()
        gradcam = GradCAM(model, model.layer4[-1])

    # Preprocess
    img_tensor = transform(image).unsqueeze(0)  # (1, 3, 224, 224)
    img_gray   = denormalize(img_tensor.squeeze(0))

    # Inference
    with torch.no_grad():
        logits = model(img_tensor)
        probs  = torch.softmax(logits, dim=1)
        prob_pneumonia = probs[0, 1].item()
        prob_normal    = probs[0, 0].item()

    # Prediction at tuned threshold
    pred_class = 1 if prob_pneumonia >= THRESHOLD else 0
    pred_label = CLASS_NAMES[pred_class]

    # Grad-CAM
    cam     = gradcam.generate(img_tensor, class_idx=pred_class)
    overlay = overlay_cam(img_gray, cam)

    # ── Results layout ────────────────────────────────────────────────────────
    st.markdown("---")

    # Prediction badge
    if pred_class == 1:
        st.error(f"## 🔴 PNEUMONIA DETECTED")
    else:
        st.success(f"## 🟢 NORMAL")

    # Probability bars
    col1, col2 = st.columns(2)
    with col1:
        st.metric("P(Normal)", f"{prob_normal:.3f}", delta=None)
        st.progress(prob_normal)
    with col2:
        st.metric("P(Pneumonia)", f"{prob_pneumonia:.3f}", delta=None)
        st.progress(prob_pneumonia)

    st.caption(f"Decision threshold: {THRESHOLD} — if P(Pneumonia) ≥ {THRESHOLD} → Pneumonia")

    # ── Image + Grad-CAM side by side ─────────────────────────────────────────
    st.markdown("### Grad-CAM Visualization")
    st.caption("Red = high model attention | Blue = low model attention")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img_gray, cmap="gray")
    axes[0].set_title("Original X-Ray", fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title(
        f"Grad-CAM — {pred_label} (P={prob_pneumonia:.3f})",
        fontweight="bold",
        color="#E07B54" if pred_class == 1 else "#4C9EBE"
    )
    axes[1].axis("off")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Clinical note ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.warning(
        "⚠️ **Clinical disclaimer:** This tool is for educational and research purposes only. "
        "It is not a substitute for professional medical diagnosis. "
        "Always consult a qualified radiologist for clinical decisions."
    )

else:
    # Show example when no image uploaded
    st.markdown("---")
    st.markdown("### How to use")
    st.markdown("""
1. Click **Browse files** above
2. Select a chest X-ray image (JPEG or PNG)
3. The model will automatically run and show:
   - Prediction (Normal / Pneumonia)
   - Confidence scores
   - Grad-CAM heatmap
    """)

    st.markdown("### What Grad-CAM shows")
    st.markdown("""
- **Pneumonia cases:** Heatmap concentrates on lower lung fields and perihilar regions where consolidation appears
- **Normal cases:** Heatmap is more diffuse — no single pathological focus
- **False positives** tend to occur on X-rays with pleural haziness or cardiomegaly
    """)
