"""
Colab/laptop-friendly MiLoRA training demonstration using GPT-2.

Example:
    python train.py --method milora --rank 4 --max_steps 30

The original paper used much larger LLaMA-family models and 8 NVIDIA L40 GPUs.
This script demonstrates the paper's core MiLoRA algorithm at small scale.
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm

from milora import replace_gpt2_modules, freeze_except_adapters, trainable_parameter_count


class InstructionDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=128):
        with open(path, "r", encoding="utf-8") as f:
            self.rows = [json.loads(line) for line in f]

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        text = f"### Instruction:\n{row['instruction']}\n### Response:\n{row['output']}"

        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        ids = enc["input_ids"].squeeze(0)
        mask = enc["attention_mask"].squeeze(0)
        labels = ids.clone()
        labels[mask == 0] = -100

        return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["milora", "lora"], default="milora")
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--data", default="data/train.jsonl")
    ap.add_argument("--save_dir", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.config.pad_token_id = tokenizer.pad_token_id

    replaced = replace_gpt2_modules(model, rank=args.rank, method=args.method)
    freeze_except_adapters(model)
    model.to(device)

    trainable, total = trainable_parameter_count(model)

    print("\nMiLoRA experiment")
    print("=" * 60)
    print("Method:", args.method)
    print("Model:", args.model)
    print("Rank:", args.rank)
    print("Adapted layers:", len(replaced))
    print("Trainable parameters:", f"{trainable:,}")
    print("Total parameters:", f"{total:,}")
    print("Trainable %:", f"{100*trainable/total:.4f}%")

    if args.method == "milora":
        errors = []
        for module in model.modules():
            if hasattr(module, "initialization_error"):
                errors.append(module.initialization_error())
        if errors:
            print("Max initialization reconstruction error:",
                  f"{max(errors):.8f}")

    dataset = InstructionDataset(args.data, tokenizer, args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr
    )

    model.train()
    losses = []
    step = 0

    for epoch in range(args.epochs):
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
            batch = {k: v.to(device) for k, v in batch.items()}

            out = model(**batch)
            loss = out.loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            step += 1
            losses.append(float(loss.detach().cpu()))

            if step % 5 == 0:
                print(f"step={step:03d} loss={losses[-1]:.4f}")

            if args.max_steps > 0 and step >= args.max_steps:
                break

        if args.max_steps > 0 and step >= args.max_steps:
            break

    final_loss = sum(losses[-min(5, len(losses)):]) / min(5, len(losses))

    save_dir = Path(args.save_dir or f"outputs/{args.method}")
    save_dir.mkdir(parents=True, exist_ok=True)

    adapter_state = {
        name: p.detach().cpu()
        for name, p in model.named_parameters()
        if p.requires_grad
    }
    torch.save(adapter_state, save_dir / "adapter.pt")

    result = {
        "method": args.method,
        "model": args.model,
        "rank": args.rank,
        "adapted_layers": len(replaced),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "final_average_loss": final_loss,
        "steps": step,
    }

    with open(save_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\nDONE")
    print("Steps:", step)
    print("Final average loss:", round(final_loss, 4))
    print("Saved to:", save_dir)


if __name__ == "__main__":
    main()
