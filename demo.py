"""
Standalone mathematical demonstration of the MiLoRA initialization.

Run:
    python demo.py

This does not download an LLM.
"""

import torch
from milora import LoRALinear


def main():
    torch.manual_seed(42)

    layer = torch.nn.Linear(16, 12, bias=False)
    wrapped = LoRALinear(layer, rank=4, method="milora")

    err = wrapped.initialization_error()

    s = wrapped.singular_values()
    print("MiLoRA SVD demonstration")
    print("=" * 50)
    print("Weight shape:", tuple(layer.weight.shape))
    print("Singular values:")
    print(s.numpy())
    print()
    print("Rank used for minor components:", wrapped.rank)
    print("Relative reconstruction error at initialization:", err)

    if err < 1e-5:
        print("\nPASS: W_principal + B@A reconstructs the original W.")
    else:
        print("\nWARNING: reconstruction error is larger than expected.")

    # Compare trainable parameter counts.
    trainable = wrapped.A.numel() + wrapped.B.numel()
    full = layer.weight.numel()
    print("Full weight parameters:", full)
    print("MiLoRA trainable parameters:", trainable)
    print("Reduction:", f"{100 * (1 - trainable / full):.2f}%")

    # Show that standard LoRA starts with zero update.
    lora = LoRALinear(layer, rank=4, method="lora")
    lora_error = lora.initialization_error()
    print("\nStandard LoRA initialization error:", lora_error)
    print("MiLoRA starts from the pretrained matrix; standard LoRA starts")
    print("with a zero low-rank update.")


if __name__ == "__main__":
    main()
