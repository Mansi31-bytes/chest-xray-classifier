"""
Toy comparison: Mansi's existing two-stage freeze/unfreeze fine-tuning
vs. Orthogonal Fine-Tuning (OFT), on a small real subset of the
chest X-ray (NORMAL vs PNEUMONIA) dataset, with Grad-CAM stability.

IMPORTANT CAVEATS (read before using any numbers from this):
  - This sandbox has no GPU and cannot reach ImageNet-pretrained weights
    (download.pytorch.org is blocked here), so BOTH models are trained
    from random initialization, not from her real pretrained checkpoint.
  - Only a 360-image subset (180/class train, 50/class test) of the full
    ~5,800-image dataset is used, for CPU runtime reasons.
  - This is a methodology demo -- it shows the OFT mechanism is correctly
    implemented and produces a measurably different fine-tuning behavior
    than her original approach. It is NOT a claim about which method is
    better for real diagnostic performance; that would need her full
    pipeline (real pretrained weights, full data, GPU) to say anything
    meaningful.

Both methods share an IDENTICAL stage 1 (freeze backbone, train the FC
head) so stage 2 is the only variable being compared:
  A. BASELINE (her original): unfreeze layer4 + head, fine-tune all
     those weights directly with Adam (lr=1e-4) -- unconstrained update.
  B. OFT: layer4's conv weights stay frozen; only a learned orthogonal
     rotation (Cayley-parameterized, see oft_layer.py) is applied to
     them. Head is still fine-tuned normally.
"""

import os, sys, time, copy, json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

sys.path.insert(0, os.path.dirname(__file__))
from oft_layer import apply_oft_to_layer4
from gradcam import GradCAM, cam_stability

torch.manual_seed(0)

ROOT = "/tmp/claude-0/-home-claude/2658320a-9dba-5c3a-8e42-cad6de0b1849/scratchpad/oft_demo"
DATA = "/tmp/claude-0/-home-claude/2658320a-9dba-5c3a-8e42-cad6de0b1849/scratchpad/ds_full/pak"  # FULL real dataset (5,216 train / 624 test), not the 360-image subset
RESULTS = f"{ROOT}/results_full"
FIGS = f"{ROOT}/figures_full"
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

IMG_SIZE = 96  # smaller than her 224 to keep the full-dataset run CPU-feasible
BATCH = 32
STAGE1_EPOCHS = 6
STAGE2_EPOCHS = 10
DEVICE = "cpu"

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])
test_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

train_ds = datasets.ImageFolder(f"{DATA}/train", transform=train_tf)
test_ds = datasets.ImageFolder(f"{DATA}/test", transform=test_tf)
print("classes:", train_ds.classes, "train:", len(train_ds), "test:", len(test_ds))

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=2)

# class imbalance in the real dataset (PNEUMONIA >> NORMAL) -- weight the loss
# so accuracy isn't just "always predict PNEUMONIA"
import collections
counts = collections.Counter(train_ds.targets)
n_total = sum(counts.values())
class_weights = torch.tensor(
    [n_total / (len(counts) * counts[i]) for i in range(len(counts))], dtype=torch.float32
)
print("class counts:", dict(counts), "-> loss weights:", class_weights.tolist())


def build_model():
    m = models.resnet50(weights=None)  # no internet access to pretrained weights in this sandbox
    m.fc = nn.Sequential(
        nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 2)
    )
    return m


def freeze_all_but_head(m):
    for p in m.parameters():
        p.requires_grad = False
    for p in m.fc.parameters():
        p.requires_grad = True


def evaluate(m):
    m.eval()
    correct, total, loss_sum = 0, 0, 0.0
    crit = nn.CrossEntropyLoss(weight=class_weights)
    tp = fp = tn = fn = 0  # positive class = PNEUMONIA (index 1)
    with torch.no_grad():
        for x, y in test_loader:
            out = m(x)
            loss_sum += crit(out, y).item() * x.size(0)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += x.size(0)
            tp += ((pred == 1) & (y == 1)).sum().item()
            fp += ((pred == 1) & (y == 0)).sum().item()
            tn += ((pred == 0) & (y == 0)).sum().item()
            fn += ((pred == 0) & (y == 1)).sum().item()
    sensitivity = tp / (tp + fn + 1e-8)  # recall on PNEUMONIA
    specificity = tn / (tn + fp + 1e-8)  # recall on NORMAL
    metrics = {"acc": correct / total, "loss": loss_sum / total,
               "sensitivity_pneumonia": sensitivity, "specificity_normal": specificity}
    return metrics["acc"], metrics["loss"], metrics


def train_epochs(m, params, n_epochs, lr, tag):
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-4 if tag == "stage2" else 0.0)
    crit = nn.CrossEntropyLoss(weight=class_weights)
    for ep in range(n_epochs):
        m.train()
        t0 = time.time()
        tot_loss, n = 0.0, 0
        for x, y in train_loader:
            opt.zero_grad()
            out = m(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * x.size(0)
            n += x.size(0)
        acc, tloss, _m = evaluate(m)
        print(f"  [{tag}] epoch {ep+1}/{n_epochs} train_loss={tot_loss/n:.4f} "
              f"test_acc={acc:.3f} test_loss={tloss:.4f} ({time.time()-t0:.1f}s)")
    return m


STAGE1_CKPT = f"{RESULTS}/stage1_state.pt"
MODEL_A_CKPT = f"{RESULTS}/model_a_state.pt"

print("\n=== STAGE 1 (shared): freeze backbone, train head ===")
base_model = build_model()
if os.path.exists(STAGE1_CKPT):
    print(f"resuming: found checkpoint {STAGE1_CKPT}, skipping stage1 training")
    stage1_state = torch.load(STAGE1_CKPT, weights_only=False)
    base_model.load_state_dict(stage1_state)
else:
    freeze_all_but_head(base_model)
    train_epochs(base_model, base_model.fc.parameters(), STAGE1_EPOCHS, 1e-3, "stage1")
    stage1_state = copy.deepcopy(base_model.state_dict())
    torch.save(stage1_state, STAGE1_CKPT)
    print(f"checkpoint saved: {STAGE1_CKPT}")
stage1_acc, stage1_loss, stage1_metrics = evaluate(base_model)
print(f"stage1-only acc={stage1_acc:.3f}")

# Grad-CAM on stage-1-only model (common reference point for both branches)
target_layer_s1 = base_model.layer4[-1]
cam_s1 = GradCAM(base_model, target_layer_s1)
sample_x, sample_y = next(iter(test_loader))
cams_stage1, _ = cam_s1(sample_x)
cam_s1.remove()

results = {"stage1_acc": stage1_acc, "stage1_loss": stage1_loss}

# ---------------- Method A: baseline two-stage (unconstrained fine-tune) ----------------
print("\n=== METHOD A: BASELINE (unfreeze layer4 + head, Adam fine-tune) ===")
model_a = build_model()
model_a.load_state_dict(stage1_state)
for p in model_a.layer4.parameters():
    p.requires_grad = True
for p in model_a.fc.parameters():
    p.requires_grad = True
params_a = list(model_a.layer4.parameters()) + list(model_a.fc.parameters())
if os.path.exists(MODEL_A_CKPT):
    print(f"resuming: found checkpoint {MODEL_A_CKPT}, skipping baseline stage2 training")
    model_a.load_state_dict(torch.load(MODEL_A_CKPT, weights_only=False))
else:
    train_epochs(model_a, params_a, STAGE2_EPOCHS, 1e-4, "stage2")
    torch.save(model_a.state_dict(), MODEL_A_CKPT)
    print(f"checkpoint saved: {MODEL_A_CKPT}")
acc_a, loss_a, metrics_a = evaluate(model_a)
print(f"BASELINE final acc={acc_a:.3f} loss={loss_a:.4f}")

target_layer_a = model_a.layer4[-1]
cam_a_hook = GradCAM(model_a, target_layer_a)
cams_a, _ = cam_a_hook(sample_x)
cam_a_hook.remove()
stability_a = [cam_stability(cams_stage1[i], cams_a[i]) for i in range(len(cams_a))]

results["baseline"] = {
    "final_acc": acc_a, "final_loss": loss_a,
    "sensitivity_pneumonia": metrics_a["sensitivity_pneumonia"],
    "specificity_normal": metrics_a["specificity_normal"],
    "mean_cam_stability_vs_stage1": sum(stability_a) / len(stability_a),
}

# ---------------- Method B: OFT ----------------
print("\n=== METHOD B: ORTHOGONAL FINE-TUNING (layer4 frozen, learn rotation) ===")
model_b = build_model()
model_b.load_state_dict(stage1_state)
for p in model_b.layer4.parameters():
    p.requires_grad = False  # base weights stay frozen, by construction of OFT
oft_params = apply_oft_to_layer4(model_b, block_size=16)
for p in model_b.fc.parameters():
    p.requires_grad = True
params_b = oft_params + list(model_b.fc.parameters())
n_oft_params = sum(p.numel() for p in oft_params)
n_baseline_params = sum(p.numel() for p in params_a)
print(f"trainable params -- OFT rotation: {n_oft_params:,}  |  baseline unfreeze: {n_baseline_params:,}")

train_epochs(model_b, params_b, STAGE2_EPOCHS, 1e-3, "stage2")  # OFT params need higher lr (small init)
acc_b, loss_b, metrics_b = evaluate(model_b)
print(f"OFT final acc={acc_b:.3f} loss={loss_b:.4f}")

# orthogonality sanity check on all wrapped layers
ortho_errs = []
for mod in model_b.layer4.modules():
    if hasattr(mod, "orthogonality_check"):
        ortho_errs.append(mod.orthogonality_check())
print(f"mean ||R^T R - I||_F across OFT layers: {sum(ortho_errs)/len(ortho_errs):.2e} (should be ~0)")

target_layer_b = model_b.layer4[-1]
cam_b_hook = GradCAM(model_b, target_layer_b)
cams_b, _ = cam_b_hook(sample_x)
cam_b_hook.remove()
stability_b = [cam_stability(cams_stage1[i], cams_b[i]) for i in range(len(cams_b))]

results["oft"] = {
    "final_acc": acc_b, "final_loss": loss_b,
    "sensitivity_pneumonia": metrics_b["sensitivity_pneumonia"],
    "specificity_normal": metrics_b["specificity_normal"],
    "mean_cam_stability_vs_stage1": sum(stability_b) / len(stability_b),
    "trainable_params": n_oft_params,
    "mean_orthogonality_error": sum(ortho_errs) / len(ortho_errs),
}
results["baseline"]["trainable_params"] = n_baseline_params
results["stage1_sensitivity_pneumonia"] = stage1_metrics["sensitivity_pneumonia"]
results["stage1_specificity_normal"] = stage1_metrics["specificity_normal"]
results["dataset"] = {"train_n": len(train_ds), "test_n": len(test_ds), "img_size": IMG_SIZE,
                       "stage1_epochs": STAGE1_EPOCHS, "stage2_epochs": STAGE2_EPOCHS}

with open(f"{RESULTS}/comparison.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n=== SUMMARY ===")
print(json.dumps(results, indent=2))

# save sample tensors + cams for the figure-generation step
torch.save({
    "sample_x": sample_x, "sample_y": sample_y,
    "cams_stage1": cams_stage1, "cams_a": cams_a, "cams_b": cams_b,
    "stability_a": stability_a, "stability_b": stability_b,
}, f"{RESULTS}/gradcam_samples.pt")
print("\nSaved results/comparison.json and results/gradcam_samples.pt")
