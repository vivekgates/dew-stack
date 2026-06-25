"""
Phase 4: finetune Mistral on your data with QLoRA (4-bit, single GPU).
The resulting adapter is small (~100-200 MB) and saved on /workspace, so it
persists and can be served by vLLM on top of the base model.

Run in the SEPARATE finetune venv:
  source /workspace/venv-finetune/bin/activate
  python finetune/train_qlora.py --data /workspace/data/train.jsonl --name mistral-lora
"""
import os, sys, argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to JSONL training file")
    ap.add_argument("--name", default="mistral-lora", help="Adapter output folder name")
    ap.add_argument("--epochs", type=float, default=3.0)
    args = ap.parse_args()

    out_dir = os.path.join(config.ADAPTERS_DIR, args.name)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL, quantization_config=bnb, device_map="auto"
    )

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    dataset = load_dataset("json", data_files=args.data, split="train")

    sft_config = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_seq_length=2048,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=lora,
        processing_class=tokenizer,  # older TRL: use tokenizer=tokenizer instead
    )

    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    print(f"\nAdapter saved to {out_dir}")
    print("Serve it alongside the base model with:")
    print(f"  vllm serve {config.BASE_MODEL} --enable-lora "
          f"--lora-modules {args.name}={out_dir}")


if __name__ == "__main__":
    main()
