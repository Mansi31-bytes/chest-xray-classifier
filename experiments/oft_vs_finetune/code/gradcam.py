"""Grad-CAM, ported from Mansi's own chest-xray-classifier repo
(notebooks/04_gradcam.ipynb) -- hooks model.layer4[-1], same as her original."""

import torch
import torch.nn.functional as F
import numpy as np


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.fh = target_layer.register_forward_hook(self._forward_hook)
        self.bh = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        # Grad-CAM needs a gradient to flow back to target_layer's output. If
        # every parameter up to that layer is frozen (as in our stage1-only /
        # OFT-frozen-backbone models), autograd won't build a graph unless the
        # *input* itself requires grad -- so force that here.
        x = x.clone().requires_grad_(True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1)
        score = logits[torch.arange(x.shape[0]), class_idx].sum()
        score.backward(retain_graph=False)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # GAP of grads
        cam = (weights * self.activations).sum(dim=1)  # weighted sum
        cam = F.relu(cam)
        cam = F.interpolate(
            cam.unsqueeze(1), size=x.shape[-2:], mode="bilinear", align_corners=False
        ).squeeze(1)
        cam_min = cam.view(cam.shape[0], -1).min(dim=1)[0].view(-1, 1, 1)
        cam_max = cam.view(cam.shape[0], -1).max(dim=1)[0].view(-1, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam.cpu().numpy(), logits.detach()

    def remove(self):
        self.fh.remove()
        self.bh.remove()


def cam_stability(cam_a: np.ndarray, cam_b: np.ndarray) -> float:
    """Cosine similarity between two flattened CAMs, in [-1, 1].
    1.0 = identical explanation; lower = the explanation moved."""
    a = cam_a.flatten()
    b = cam_b.flatten()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)
