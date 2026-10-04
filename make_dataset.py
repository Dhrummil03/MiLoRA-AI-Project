from pathlib import Path
import json

examples = [
    ("What is machine learning?", "Machine learning is a branch of AI in which models learn patterns from data to make predictions or decisions."),
    ("What is an algorithm?", "An algorithm is a finite sequence of well-defined steps used to solve a problem or perform a computation."),
    ("What is Python?", "Python is a high-level programming language widely used for software development, automation, data science and artificial intelligence."),
    ("What is deep learning?", "Deep learning is a machine learning approach that uses neural networks with multiple layers to learn representations from data."),
    ("What is an LLM?", "An LLM is a large language model trained on large amounts of text to understand and generate natural language."),
    ("What is fine-tuning?", "Fine-tuning adapts a pretrained model to a particular task or dataset by training it further on task-specific examples."),
    ("What is LoRA?", "LoRA is a parameter-efficient fine-tuning method that learns a low-rank weight update while keeping the pretrained model weights frozen."),
    ("What is SVD?", "Singular Value Decomposition factorizes a matrix into left singular vectors, singular values and right singular vectors."),
    ("Why use GPUs for deep learning?", "GPUs can perform many matrix operations in parallel, making neural-network training much faster than typical CPU execution."),
    ("What is overfitting?", "Overfitting occurs when a model learns the training data too closely and performs poorly on unseen data."),
    ("What is a neural network?", "A neural network is a model made of interconnected computational units that learn parameters from data."),
    ("What is NLP?", "Natural Language Processing is the field of AI concerned with processing, understanding and generating human language."),
    ("What is parameter-efficient fine-tuning?", "Parameter-efficient fine-tuning updates only a small number of trainable parameters while keeping most pretrained parameters frozen."),
    ("What is a transformer?", "A transformer is a neural architecture based primarily on attention mechanisms and is widely used for language modeling."),
    ("Why freeze pretrained weights?", "Freezing pretrained weights reduces memory and computation requirements and helps retain knowledge learned during pretraining."),
    ("What is a singular value?", "A singular value measures the strength of a corresponding singular direction in a matrix."),
    ("What is a low-rank matrix?", "A low-rank matrix can be represented using fewer independent dimensions than its full matrix dimensions."),
    ("What is an attention mechanism?", "Attention lets a model assign different importance to elements of an input when producing a representation."),
    ("What is model inference?", "Inference is the process of using a trained model to generate predictions or outputs for new inputs."),
    ("What is a dataset?", "A dataset is a collection of examples used to train, validate or evaluate a machine-learning model."),
]

out = Path("data")
out.mkdir(exist_ok=True)

with open(out / "train.jsonl", "w", encoding="utf-8") as f:
    for question, answer in examples:
        row = {
            "instruction": question,
            "output": answer
        }
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Wrote {len(examples)} examples to {out / 'train.jsonl'}")
