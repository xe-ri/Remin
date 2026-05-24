import argparse
import os
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if os.environ.get("OMP_NUM_THREADS") in (None, "", "0"):
    os.environ["OMP_NUM_THREADS"] = "1"

import unsloth
import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel


DEFAULT_BASE_MODEL = "unsloth/Qwen3-8B-bnb-4bit"
DEFAULT_DATASET = "training/datasets/train.jsonl"
DEFAULT_OUTPUT = "models/remin_adapter"


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Remin on Qwen3-8B.")
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="现有大模型名称或本地路径，例如 unsloth/Qwen3-8B-bnb-4bit",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="训练数据 JSONL 路径，需包含 text 字段。",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT,
        help="LoRA adapter 输出目录。",
    )
    parser.add_argument("--max-seq-length", type=int, default=3072)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accumulation", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"未找到训练数据文件: {dataset_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_gradient_checkpointing="unsloth",
    )

    dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accumulation,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=5,
            save_steps=args.max_steps,
            lr_scheduler_type="cosine",
            optim="adamw_torch",
            output_dir=str(output_dir),
            seed=args.seed,
            report_to="none",
            dataset_text_field="text",
            max_seq_length=args.max_seq_length,
            packing=False,
            padding_free=False,
        ),
    )

    trainer.train()
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Qwen3-8B LoRA adapter 已保存到: {output_dir}")


if __name__ == "__main__":
    main()
