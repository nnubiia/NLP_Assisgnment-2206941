"""
finetune.py
Fine-tunes Qwen3.5-0.8B on the CineBot movie recommendation dataset
using LoRA (Low-Rank Adaptation) via HuggingFace transformers + peft.

Run this ONCE on your PC before running run_chatbot.py.
The fine-tuned model will be saved to ./cinebot-model/

Requirements:
    pip install transformers peft accelerate datasets torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Usage:
    python finetune.py
"""

import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME   = "Qwen/Qwen3.5-0.8B"
OUTPUT_DIR   = "./cinebot-model"
DATA_FILE    = "training_data.json"
MAX_LENGTH   = 256
EPOCHS       = 3
BATCH_SIZE   = 4
LEARNING_RATE = 2e-4

# ── Load training data ────────────────────────────────────────────────────────

print("Loading training data...")
with open(DATA_FILE, "r") as f:
    raw_data = json.load(f)

# Format as instruction-response pairs
def format_example(example):
    return {
        "text": f"### User: {example['input']}\n### CineBot: {example['output']}<|endoftext|>"
    }

formatted = [format_example(ex) for ex in raw_data]
dataset   = Dataset.from_list(formatted)
print(f"Loaded {len(dataset)} training examples.")

# ── Load model and tokenizer ──────────────────────────────────────────────────

print(f"\nDownloading and loading {MODEL_NAME}...")
print("(This may take a few minutes on first run — the model is ~1.5GB)\n")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,    # fp16 to fit on 6GB VRAM
    device_map="auto",            # automatically use GPU if available
    trust_remote_code=True,
)

# ── Apply LoRA ────────────────────────────────────────────────────────────────
# LoRA adds small trainable matrices to the attention layers.

print("Applying LoRA configuration...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                          # rank of LoRA matrices (lower = fewer params)
    lora_alpha=16,                # scaling factor
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],  # apply LoRA to attention query/value layers
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── Tokenise dataset ──────────────────────────────────────────────────────────

def tokenise(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )

print("\nTokenising dataset...")
tokenised_dataset = dataset.map(tokenise, remove_columns=["text"])
tokenised_dataset = tokenised_dataset.train_test_split(test_size=0.1, seed=42)

# ── Training ──────────────────────────────────────────────────────────────────

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    fp16=True,                    # use fp16 for GTX 1060
    logging_steps=5,
    save_steps=50,
    eval_strategy="epoch",
    save_total_limit=1,
    report_to="none",             # disable wandb/tensorboard
    warmup_steps=10,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,                    # causal LM, not masked LM
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenised_dataset["train"],
    eval_dataset=tokenised_dataset["test"],
    data_collator=data_collator,
)

print("\nStarting fine-tuning...")
print("This will take around 5-15 minutes on a GTX 1060.\n")
trainer.train()

# ── Save ──────────────────────────────────────────────────────────────────────

print(f"\nSaving fine-tuned model to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("\nDone! Your fine-tuned model is saved to ./cinebot-model/")
print("You can now run: python run_chatbot.py")
