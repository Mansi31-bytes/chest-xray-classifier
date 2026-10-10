import torch, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/tmp/claude-0/-home-claude/2658320a-9dba-5c3a-8e42-cad6de0b1849/scratchpad/oft_demo"
d = torch.load(f"{ROOT}/results_full/gradcam_samples.pt", weights_only=False)
x = d["sample_x"]; y = d["sample_y"]
cams1, cams_a, cams_b = d["cams_stage1"], d["cams_a"], d["cams_b"]
stab_a, stab_b = d["stability_a"], d["stability_b"]
classes = ["NORMAL", "PNEUMONIA"]

def denorm(img):
    img = img.permute(1, 2, 0).numpy() * 0.5 + 0.5
    return np.clip(img, 0, 1)

n = min(4, x.shape[0])
fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
for i in range(n):
    base = denorm(x[i])[:, :, 0]
    axes[i, 0].imshow(base, cmap="gray")
    axes[i, 0].set_title(f"{classes[y[i].item()]} (input)")
    axes[i, 1].imshow(base, cmap="gray")
    axes[i, 1].imshow(cams1[i], cmap="jet", alpha=0.45)
    axes[i, 1].set_title("stage1-only CAM")
    axes[i, 2].imshow(base, cmap="gray")
    axes[i, 2].imshow(cams_a[i], cmap="jet", alpha=0.45)
    axes[i, 2].set_title(f"Baseline CAM\nstability={stab_a[i]:.2f}")
    axes[i, 3].imshow(base, cmap="gray")
    axes[i, 3].imshow(cams_b[i], cmap="jet", alpha=0.45)
    axes[i, 3].set_title(f"OFT CAM\nstability={stab_b[i]:.2f}")
    for j in range(4):
        axes[i, j].axis("off")

plt.tight_layout()
plt.savefig(f"{ROOT}/figures_full/gradcam_comparison.png", dpi=130)
print("saved figure")
