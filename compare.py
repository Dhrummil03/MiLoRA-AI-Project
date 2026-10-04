import json
from pathlib import Path

methods = ["lora", "milora"]

print("MiLoRA vs LoRA")
print("=" * 70)

for method in methods:
    path = Path("outputs") / method / "result.json"
    if not path.exists():
        print(f"{method:8s}: result not found ({path})")
        continue

    with open(path, "r", encoding="utf-8") as f:
        r = json.load(f)

    print(
        f"{method:8s} | "
        f"rank={r['rank']} | "
        f"trainable={r['trainable_parameters']:,} | "
        f"total={r['total_parameters']:,} | "
        f"trainable%={100*r['trainable_parameters']/r['total_parameters']:.4f}% | "
        f"loss={r['final_average_loss']:.4f}"
    )

print("\nInterpretation:")
print("- Both methods train only a small number of adapter parameters.")
print("- MiLoRA initializes those adapters from the minor singular components.")
print("- The small experiment is a demonstration, not a reproduction of the paper's")
print("  large-scale benchmark numbers.")
