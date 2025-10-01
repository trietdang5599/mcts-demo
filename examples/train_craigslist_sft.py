"""Supervised fine-tuning on the Craigslist Bargains negotiation dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# Allow running the script without installing the package
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dyna_gym.data_utils import (
    CRAIGSLIST_SPECIAL_TOKENS,
    load_craigslist_split,
    render_dialogue,
)


def build_dataset(data_dir: Path, splits: List[str], include_outcome: bool = True) -> DatasetDict:
    """Load the dataset splits and return a DatasetDict of formatted strings."""
    datasets = {}
    for split in splits:
        examples = load_craigslist_split(data_dir, split)
        texts = [render_dialogue(ex, include_outcome=include_outcome) for ex in examples]
        datasets[split] = Dataset.from_dict({"text": texts})
    return DatasetDict(datasets)


def tokenize_dataset(tokenizer, dataset: DatasetDict, max_length: int):
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            max_length=max_length,
            truncation=True,
        )

    return dataset.map(tokenize, batched=True, remove_columns=["text"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/craigslist_bargains"), help="Path to the dataset directory containing raw/*.json")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model or path to fine-tune")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/craigslist-sft"), help="Directory to store checkpoints")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length for training")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--gradient_accumulation", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=float, default=15, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay coefficient")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 mixed precision if available")
    parser.add_argument("--bf16", action="store_true", help="Enable bf16 mixed precision if available")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Maximum checkpoints to keep")
    parser.add_argument("--eval_split", type=str, default="validation", help="Split used for evaluation")
    parser.add_argument("--train_split", type=str, default="train", help="Split used for training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    data_dir = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_special_tokens(CRAIGSLIST_SPECIAL_TOKENS)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))

    dataset = build_dataset(data_dir, [args.train_split, args.eval_split])
    tokenized = tokenize_dataset(tokenizer, dataset, args.max_length)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to=[]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized[args.train_split],
        eval_dataset=tokenized[args.eval_split],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
