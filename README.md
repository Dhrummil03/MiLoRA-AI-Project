# MiLoRA – Lightweight Reproduction for a Student AI Project

This project is a **small, laptop-friendly implementation of the core MiLoRA idea** from:

> Wang et al., "MiLoRA: Harnessing Minor Singular Components for Parameter-Efficient LLM Finetuning", NAACL 2025.

Official paper/code:
- https://github.com/sufenlp/MiLoRA
- https://aclanthology.org/2025.naacl-long.248/

## Important scope

The original paper evaluates LLaMA-family models and reports experiments using 8 NVIDIA L40 GPUs. This repository is a **scaled-down educational reproduction**, designed to demonstrate the core algorithm on a consumer GPU such as an RTX 4050.

It implements:
1. Standard LoRA baseline.
2. SVD decomposition of selected pretrained linear layers.
3. Separation into principal and minor singular components.
4. MiLoRA initialization from the **minor singular components**.
5. Parameter-efficient fine-tuning.
6. A small comparison of LoRA vs MiLoRA.
7. Parameter-count and initialization checks.

## Core MiLoRA equation

For a pretrained matrix:

    W = U Sigma V^T

The largest singular components form:

    Wp = Up Sigmap Vp^T

The smallest `r` singular components form:

    Wm = Um Sigmam Vm^T

MiLoRA freezes `Wp` and initializes:

    B = Um Sigmam
    A = Vm^T

so that:

    B A = Wm

Therefore at initialization:

    Wp + B A = W

This preserves the original pretrained layer before training while directing learning toward the minor singular subspace.

## Project structure

```text
MiLoRA_4050_Project/
├── milora.py
├── train.py
├── compare.py
├── demo.py
├── make_dataset.py
├── requirements.txt
├── data/
│   └── train.jsonl
└── README.md
```

## 1. Create environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If CUDA/PyTorch installation needs a GPU-specific wheel, install PyTorch from the official PyTorch selector first, then run:

```powershell
pip install transformers datasets accelerate
```

## 2. Check the algorithm without downloading a model

```powershell
python demo.py
```

This performs an SVD on a random matrix and verifies that the MiLoRA decomposition reconstructs the original matrix.

## 3. Create the tiny training dataset

```powershell
python make_dataset.py
```

A small instruction dataset is generated under `data/train.jsonl`.

## 4. Train MiLoRA

```powershell
python train.py --method milora --model distilgpt2 --rank 4 --epochs 1
```

For a faster CPU-only test:

```powershell
python train.py --method milora --model distilgpt2 --rank 2 --max_steps 20 --device cpu
```

## 5. Train standard LoRA baseline

```powershell
python train.py --method lora --model distilgpt2 --rank 4 --epochs 1
```

## 6. Compare both methods

After both runs:

```powershell
python compare.py
```

The comparison reports trainable parameters and final loss from the saved result files.

## 7. Generate text

```powershell
python train.py --method milora --model distilgpt2 --rank 4 --epochs 1 --save_dir outputs/milora
```

Then:

```powershell
python train.py --method milora --model distilgpt2 --rank 4 --max_steps 0 --load_dir outputs/milora
```

If you only need a project demonstration tomorrow, the most useful commands are:

```powershell
python demo.py
python make_dataset.py
python train.py --method milora --model distilgpt2 --rank 4 --max_steps 30
```

## What to say in the viva

**Q: What is the main contribution?**

MiLoRA changes the initialization of LoRA. Instead of randomly initializing the trainable low-rank update, it obtains the SVD of the pretrained weight matrix and initializes the LoRA factors from the smallest, or minor, singular components.

**Q: Why minor components?**

The paper argues that principal singular components contain important pretrained knowledge, while minor components represent less-optimized/noisy or long-tail information. Updating the minor subspace can therefore reduce interference with important pretrained knowledge.

**Q: Is MiLoRA the same as PiSSA?**

No. PiSSA adapts the principal singular components, while MiLoRA adapts the minor singular components.

**Q: Does MiLoRA guarantee strict orthogonality during training?**

No. The paper deliberately uses a softer approach: the minor subspace is selected through initialization rather than enforcing a hard orthogonality constraint throughout training.

**Q: Why did we not reproduce the exact 7B experiment?**

The original paper used large GPU resources. This project is an educational, laptop-scale implementation of the core algorithm.

## Citation

Wang, Hanqing, et al. "MiLoRA: Harnessing Minor Singular Components for Parameter-Efficient LLM Finetuning." NAACL 2025.


## Google Colab recommended workflow

1. Open Google Colab and create a new notebook.
2. Runtime -> Change runtime type -> GPU.
3. Upload this ZIP or clone your own GitHub repository.
4. Extract into `/content/MiLoRA_4050_Project`.
5. Run the commands in the submission guide:
   - `pip install -r requirements.txt`
   - `python demo.py`
   - `python make_dataset.py`
   - `python train.py --method milora --rank 4 --max_steps 30`
   - `python train.py --method lora --rank 4 --max_steps 30`
   - `python compare.py`
