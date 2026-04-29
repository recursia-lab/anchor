"""
Fine-tune PaliGemma2 with PEFT LoRA for a custom visual task.

This script trains a LoRA adapter that can then be loaded into Anchor.

Requirements:
    pip install transformers peft accelerate torch Pillow

Usage:
    python finetune.py \
        --model google/paligemma2-3b-pt-448 \
        --data ./my_dataset \
        --output ./my_adapter \
        --task "Is there a defect? Answer YES or NO."

Dataset format (CSV with columns: image_path, label):
    image_path,label
    images/img001.jpg,YES
    images/img002.jpg,NO
"""

import argparse
import csv
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration


class SimpleVLDataset(Dataset):
    def __init__(self, csv_path: str, processor, task_prompt: str, image_dir: str = ""):
        self.processor = processor
        self.task_prompt = task_prompt
        self.samples = []

        with open(csv_path) as f:
            for row in csv.DictReader(f):
                img_path = Path(image_dir) / row["image_path"] if image_dir else Path(row["image_path"])
                self.samples.append((str(img_path), row["label"]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        inputs = self.processor(
            text=f"<image>{self.task_prompt}",
            images=image,
            suffix=label,
            return_tensors="pt",
            padding=True,
        )
        return {k: v.squeeze(0) for k, v in inputs.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/paligemma2-3b-pt-448")
    p.add_argument("--data", required=True, help="Path to CSV dataset file")
    p.add_argument("--output", required=True, help="Output directory for LoRA adapter")
    p.add_argument("--task", required=True, help="Task prompt (e.g. 'Is there a defect? YES or NO.')")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--image-dir", default="", help="Base directory for image paths in CSV")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading model: {args.model}")

    processor = AutoProcessor.from_pretrained(args.model)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # LoRA config — targets language model layers only
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = SimpleVLDataset(args.data, processor, args.task, args.image_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            if (step + 1) % 10 == 0:
                print(f"  Epoch {epoch+1} step {step+1}/{len(loader)} loss={total_loss/(step+1):.4f}")

        print(f"Epoch {epoch+1} avg loss: {total_loss/len(loader):.4f}")

    # Save adapter
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output))
    print(f"\nAdapter saved to: {output}")
    print(f"\nTo serve with Anchor, copy to /lora/{output.name}/ and restart.")


if __name__ == "__main__":
    main()
