"""
Lightweight MiLoRA implementation for GPT-2 style layers.

MiLoRA:
    W = U Sigma V^T
    W = W_principal + W_minor

The smallest r singular components initialize B and A:
    B = U_minor * Sigma_minor
    A = V_minor^T
so:
    B @ A = W_minor

The principal component is frozen; A and B are trainable.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAWeight(nn.Module):
    """LoRA/MiLoRA wrapper for a matrix used as x @ W."""

    def __init__(self, weight, bias, rank, method="milora"):
        super().__init__()
        if rank > min(weight.shape):
            raise ValueError(f"rank {rank} is too large for weight {tuple(weight.shape)}")

        W = weight.detach().float()
        self.in_features, self.out_features = W.shape
        self.rank = rank
        self.method = method

        self.register_buffer("original_weight", W.clone())

        if bias is not None:
            self.register_buffer("frozen_bias", bias.detach().float().clone())
        else:
            self.frozen_bias = None

        if method == "milora":
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)

            # Smallest r singular components.
            Um = U[:, -rank:]
            Sm = S[-rank:]
            Vhm = Vh[-rank:, :]

            B_init = Um * Sm.unsqueeze(0)
            A_init = Vhm

            W_minor = B_init @ A_init
            W_principal = W - W_minor

            self.register_buffer("principal_weight", W_principal)
            self.register_buffer("minor_weight", W_minor)

            self.A = nn.Parameter(A_init.clone())
            self.B = nn.Parameter(B_init.clone())

        elif method == "lora":
            self.register_buffer("principal_weight", W.clone())
            self.register_buffer("minor_weight", torch.zeros_like(W))

            self.A = nn.Parameter(
                torch.empty(rank, W.shape[1], dtype=W.dtype)
            )
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

            self.B = nn.Parameter(
                torch.zeros(W.shape[0], rank, dtype=W.dtype)
            )
        else:
            raise ValueError("method must be 'lora' or 'milora'")

    def forward_matrix(self):
        return self.principal_weight + self.B @ self.A

    def forward(self, x):
        W = self.forward_matrix()
        # GPT-2 Conv1D convention: y = x @ W + b
        y = torch.matmul(x, W)
        if self.frozen_bias is not None:
            y = y + self.frozen_bias
        return y

    @torch.no_grad()
    def initialization_error(self):
        reconstructed = self.forward_matrix()
        denom = self.original_weight.norm().clamp_min(1e-12)
        return ((reconstructed - self.original_weight).norm() / denom).item()

    @torch.no_grad()
    def singular_values(self):
        return torch.linalg.svdvals(self.original_weight).cpu()


class MiLoRAConv1D(nn.Module):
    """
    Wrapper for transformers.pytorch_utils.Conv1D used by GPT-2.
    The underlying GPT-2 Conv1D stores weight as [in_features, out_features].
    """

    def __init__(self, base, rank, method="milora"):
        super().__init__()
        self.adapter = LoRAWeight(base.weight, base.bias, rank, method)

    def forward(self, x):
        return self.adapter(x)

    @torch.no_grad()
    def initialization_error(self):
        return self.adapter.initialization_error()


def replace_gpt2_modules(model, rank=4, method="milora"):
    """Replace GPT-2 attention/MLP projection Conv1D modules."""
    from transformers.pytorch_utils import Conv1D

    replaced = []

    for module_name, module in list(model.named_modules()):
        if not isinstance(module, Conv1D):
            continue

        # GPT-2's attention c_attn and c_proj and MLP c_fc/c_proj are all
        # valid matrix transformations. We adapt these layers.
        parent_name, child_name = (
            module_name.rsplit(".", 1)
            if "." in module_name else ("", module_name)
        )
        parent = model.get_submodule(parent_name) if parent_name else model

        wrapped = MiLoRAConv1D(module, rank=rank, method=method)
        setattr(parent, child_name, wrapped)
        replaced.append(module_name)

    if not replaced:
        raise RuntimeError("No GPT-2 Conv1D modules were found.")

    return replaced


def freeze_except_adapters(model):
    for name, parameter in model.named_parameters():
        parameter.requires_grad = (
            name.endswith(".adapter.A") or name.endswith(".adapter.B")
        )


def trainable_parameter_count(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
