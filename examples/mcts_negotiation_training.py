"""MCTS-guided negotiation fine-tuning pipeline for Craigslist Bargains."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Tuple

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dyna_gym.data_utils import CRAIGSLIST_SPECIAL_TOKENS, load_craigslist_split, render_dialogue
from dyna_gym.data_utils.craigslist import extract_final_price
from dyna_gym.pipelines import uct_for_hf_transformer_pipeline


def build_prompt(example, prompt_turns: int) -> str:
    return render_dialogue(example, include_outcome=False, max_turns=prompt_turns)


def make_reward_fn(example):
    buyer_target = example.buyer_target or example.list_price or 0.0
    seller_target = example.seller_target or example.list_price or buyer_target
    list_price = example.list_price or seller_target or buyer_target or 1.0

    def reward_fn(text: str) -> float:
        lowered = text.lower()
        deal_mentions = lowered.count("deal reached")
        if deal_mentions != 1:
            return -1.0

        price = extract_final_price(text)
        if price is None:
            return -1.0

        price_token = f"${price:,.2f}"
        if text.count(price_token) != 1:
            return -1.0

        within_interval = 1.0 if buyer_target <= price <= seller_target else 0.0
        proximity = 1.0 - min(abs(price - list_price) / max(list_price, 1.0), 1.0)

        length_penalty = 0.0
        turns = lowered.count("buyer") + lowered.count("seller")
        if turns > 16:
            length_penalty = 0.05 * (turns - 16)

        return 0.6 * within_interval + 0.4 * proximity - length_penalty

    return reward_fn


def tokenize_texts(tokenizer, dataset: Dataset, max_length: int):
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    return dataset.map(tokenize, batched=True, remove_columns=["text"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/craigslist_bargains"))
    parser.add_argument("--model_path", type=str, required=True, help="Path to the initial fine-tuned model checkpoint")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/craigslist-mcts"))
    parser.add_argument("--num_samples", type=int, default=128, help="Number of dialogues to generate with MCTS")
    parser.add_argument("--prompt_turns", type=int, default=2, help="Number of ground-truth turns to condition on")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation", type=int, default=16)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--rollouts", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=512)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--reuse_tree", action="store_true", help="Reuse MCTS tree between steps")
    parser.add_argument("--no_train", action="store_true", help="Only generate dialogues, skip further fine-tuning")
    parser.add_argument("--should_print_tree", action="store_true", help="Print search trees during generation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.add_special_tokens(CRAIGSLIST_SPECIAL_TOKENS)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model.resize_token_embeddings(len(tokenizer))

    examples = load_craigslist_split(args.data_dir, "train")
    random.shuffle(examples)
    examples = examples[: args.num_samples]

    generation_args = dict(
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=True,
        temperature=args.temperature,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    uct_args = dict(
        rollouts=args.rollouts,
        gamma=1.0,
        width=args.top_k,
        reuse_tree=args.reuse_tree,
    )

    generator = uct_for_hf_transformer_pipeline(
        model=model,
        tokenizer=tokenizer,
        horizon=args.horizon,
        reward_func=lambda _: 0.0,
        uct_args=uct_args,
        model_generation_args=generation_args,
        should_plot_tree=False,
        should_print_tree=args.should_print_tree,
        reward_func_input_is_state=False,
        decode_skip_special_tokens=False,
    )

    generated_dialogues: List[Tuple[str, str, float, dict]] = []

    for example in examples:
        prompt = build_prompt(example, args.prompt_turns)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].squeeze(0).to(model.device)
        attention_mask = inputs["attention_mask"].squeeze(0).to(model.device)

        reward_fn = make_reward_fn(example)

        outputs = generator(
            input_ids=input_ids,
            attention_mask=attention_mask,
            reward_override=reward_fn,
            skip_special_tokens=False,
        )
        if not outputs["texts"]:
            continue
        best_idx = max(range(len(outputs["rewards"])), key=lambda idx: outputs["rewards"][idx])
        text_with_tokens = outputs["texts_with_special_tokens"][best_idx]
        plain_text = outputs["texts_plain"][best_idx]

        metadata = {
            "scenario_id": example.scenario_id,
            "buyer_target": example.buyer_target,
            "seller_target": example.seller_target,
            "list_price": example.list_price,
            "category": example.category,
            "title": example.title,
        }
        generated_dialogues.append((text_with_tokens, plain_text, float(outputs["rewards"][best_idx]), metadata))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / "mcts_dialogues.jsonl"
    with data_path.open("w", encoding="utf-8") as fh:
        for text_with_tokens, plain_text, reward, meta in generated_dialogues:
            record = {
                "text": text_with_tokens,
                "plain_text": plain_text,
                "reward": reward,
                **meta,
            }
            # count dialogue turns for downstream metrics
            turn_count = text_with_tokens.count("<buyer>") + text_with_tokens.count("<seller>")
            record["turn_count"] = turn_count
            fh.write(json.dumps(record) + "\n")

    if args.no_train or not generated_dialogues:
        return

    dataset = Dataset.from_dict({
        "text": [text for text, _, _, _ in generated_dialogues]
    })
    tokenized = tokenize_texts(tokenizer, dataset, args.max_length)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        overwrite_output_dir=False,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
