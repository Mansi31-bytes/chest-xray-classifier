"""
Minimal Orthogonal Fine-Tuning (OFT) wrapper for a Conv2d layer.

Core idea (Qiu et al., "Controlling Text-to-Image Diffusion by Orthogonal
Finetuning", 2023 -- the OFT method Prof. Weller has co-authored work on):
instead of additively updating a pretrained weight matrix W (as plain
fine-tuning / LoRA do), OFT learns an orthogonal transform R and sets

    W' = R @ W

R is constrained to O(n) by parameterizing it as a Cayley transform of a
learnable skew-symmetric matrix S (S = A - A^T):

    R = (I - S)(I + S)^{-1}

This guarantees R^T R = I exactly, every training step, by construction --
not by a loss penalty. Orthogonal transforms preserve pairwise angles and
norms between neurons, which is the stated motivation: it keeps the
pretrained weight structure ("hyperspherical energy") intact while still
letting the layer rotate to fit new data, in contrast to unconstrained
fine-tuning which can drift the weights arbitrarily.

To keep this tractable on CPU, we apply a *block-diagonal* R (block size
`block_size`) to the output-channel dimension of a frozen Conv2d, rather
than one dense R over all channels -- this is the same block-diagonal
simplification used in the original OFT paper to control parameter count.
"""

import torch
import torch.nn as nn


class OFTConv2d(nn.Module):
    """Wraps a frozen Conv2d; learns a block-diagonal orthogonal transform
    applied to its output channels. The base conv weight is NOT updated --
    only the orthogonal rotation is."""

    def __init__(self, base_conv: nn.Conv2d, block_size: int = 16):
        super().__init__()
        assert isinstance(base_conv, nn.Conv2d)
        self.base_conv = base_conv
        for p in self.base_conv.parameters():
            p.requires_grad = False

        out_ch = base_conv.out_channels
        self.out_ch = out_ch
        self.block_size = min(block_size, out_ch)
        # number of full blocks + remainder handled by padding to nearest block
        n_blocks = (out_ch + self.block_size - 1) // self.block_size
        self.n_blocks = n_blocks
        self.pad = n_blocks * self.block_size - out_ch

        # learnable skew-symmetric generator per block: store only the
        # strictly-upper-triangular entries, rebuild the skew matrix in forward
        b = self.block_size
        self.tri_idx = torch.triu_indices(b, b, offset=1)
        n_params = self.tri_idx.shape[1]
        self.skew_params = nn.Parameter(torch.zeros(n_blocks, n_params) * 0.0)

    def _build_R(self, device, dtype):
        b = self.block_size
        n = self.n_blocks
        S = torch.zeros(n, b, b, device=device, dtype=dtype)
        idx = self.tri_idx.to(device)
        S[:, idx[0], idx[1]] = self.skew_params.to(dtype)
        S = S - S.transpose(1, 2)  # skew-symmetric: S = A - A^T by construction
        I = torch.eye(b, device=device, dtype=dtype).unsqueeze(0).expand(n, b, b)
        # Cayley transform: R = (I - S)(I + S)^-1, guarantees orthogonality
        R = torch.linalg.solve(I + S, I - S)
        return R  # (n_blocks, b, b)

    def forward(self, x):
        w = self.base_conv.weight  # (out_ch, in_ch, kh, kw), frozen
        device, dtype = w.device, w.dtype
        R = self._build_R(device, dtype)  # (n_blocks, b, b)

        out_ch = self.out_ch
        b = self.block_size
        if self.pad:
            w_pad = torch.cat(
                [w, torch.zeros(self.pad, *w.shape[1:], device=device, dtype=dtype)],
                dim=0,
            )
        else:
            w_pad = w
        w_blocks = w_pad.view(self.n_blocks, b, -1)  # (n_blocks, b, in*kh*kw)
        w_rot = torch.bmm(R, w_blocks)  # rotate each block of output channels
        w_rot = w_rot.view(self.n_blocks * b, *w.shape[1:])[:out_ch]

        return nn.functional.conv2d(
            x,
            w_rot,
            bias=self.base_conv.bias,
            stride=self.base_conv.stride,
            padding=self.base_conv.padding,
            dilation=self.base_conv.dilation,
            groups=self.base_conv.groups,
        )

    def orthogonality_check(self):
        """Returns mean ||R^T R - I||_F across blocks -- should be ~0."""
        with torch.no_grad():
            R = self._build_R(self.skew_params.device, self.skew_params.dtype)
            I = torch.eye(R.shape[-1], device=R.device).unsqueeze(0)
            err = torch.bmm(R.transpose(1, 2), R) - I
            return err.norm(dim=(1, 2)).mean().item()


def apply_oft_to_layer4(model, block_size=16):
    """Replaces every Conv2d inside model.layer4 with an OFTConv2d wrapper.
    Everything else in the model is left untouched (and should already be
    frozen by the caller, matching the two-stage baseline's stage-2 scope)."""
    oft_params = []
    for name, module in model.layer4.named_children():
        for sub_name, sub_module in module.named_children():
            if isinstance(sub_module, nn.Conv2d):
                wrapped = OFTConv2d(sub_module, block_size=block_size)
                setattr(module, sub_name, wrapped)
                oft_params.append(wrapped.skew_params)
    return oft_params
