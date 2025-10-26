import torch
import json
import os
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import wandb

# Force use of specific GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# ============================================
# 0. SETUP DIRECTORIES AND WANDB
# ============================================
MODEL_CACHE_DIR = "./model_cache"
RESULTS_DIR = "./results"
OUTPUT_DIR = "./output"

Path(MODEL_CACHE_DIR).mkdir(exist_ok=True)
Path(RESULTS_DIR).mkdir(exist_ok=True)
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# Initialize Weights & Biases
wandb.init(
    project="citation-intent-classification",
    name="llama-3.1-8b-sft",
    config={
        "model_name": "Meta-Llama-3.1-8B-Instruct",
        "dataset": "allenai/SciRIFF",
        "task": "citation_intent_classification",
        "method": "SFT with LoRA",
        "learning_rate": 2e-4,
    }
)

# ============================================
# 1. LOAD SCIRIFF DATASET
# ============================================
print("="*60)
print("LOADING SCIRIFF DATASET")
print("="*60)

dataset = load_dataset("allenai/SciRIFF", "4096")

# Filter for citation classification only (single-label)
filtered_train = dataset['train'].filter(
    lambda x: 'classification' in x['metadata']['task_family'].lower() and
              'citation' in x['input'].lower() and
              not x['output'].strip().startswith('[')
)
filtered_val = dataset['validation'].filter(
    lambda x: 'classification' in x['metadata']['task_family'].lower() and
              'citation' in x['input'].lower() and
              not x['output'].strip().startswith('[')
)

print(f"Filtered train: {len(filtered_train)} (citation classification only)")
print(f"Filtered val: {len(filtered_val)}")

# Extract unique labels
unique_labels = set()
for example in filtered_train:
    unique_labels.add(example['output'].strip())
for example in filtered_val:
    unique_labels.add(example['output'].strip())

label_list = sorted(list(unique_labels))
num_labels = len(label_list)
print(f"Number of labels: {num_labels}")
print(f"Labels: {label_list}")

wandb.config.update({
    "train_samples": len(filtered_train),
    "val_samples": len(filtered_val),
    "num_labels": num_labels
})

# ============================================
# 2. LOAD MODEL WITH QLORA
# ============================================
print("\n" + "="*60)
print("LOADING LLAMA 3.1 8B WITH QLORA")
print("="*60)

model_path = "/gpfs1/llm/llama-3.1-hf/Meta-Llama-3.1-8B-Instruct"
print(f"Model path: {model_path}")

# QLoRA config - 4-bit quantization for efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

# LoRA config - only train small adapters
lora_config = LoraConfig(
    r=16,  # LoRA rank
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

wandb.config.update({
    "lora_r": 16,
    "lora_alpha": 32,
    "trainable_params": trainable_params
})

# ============================================
# 3. FORMAT DATA FOR SFT
# ============================================
print("\n" + "="*60)
print("FORMATTING DATA FOR SFT")
print("="*60)

def extract_citation_context(text):
    """Extract citation context, removing task instructions"""
    if "Section Title:" in text:
        content = text.split("Section Title:")[1].strip()
        content = content.replace("Context before the citation:", "")
        content = content.replace("Citation Sentence:", "")
        content = content.replace("Context after the citation:", "")
        content = " ".join(content.split())
        return content  # Let tokenizer handle truncation (max_seq_length=1024)
    return text

def format_instruction(example):
    """Format as instruction-following example for Llama"""
    citation_context = extract_citation_context(example['input'])
    label = example['output'].strip()

    # Use Llama 3.1 chat format
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert at classifying citation intents in academic papers. Classify citations into one of these categories: Background, Method, Result, CompareOrContrast, Uses, Extends, Motivation, FutureWork.<|eot_id|><|start_header_id|>user<|end_header_id|>

Classify the intent of this citation:

{citation_context}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{label}<|eot_id|>"""

    return {"text": prompt}

# Format datasets
print("Formatting training data...")
train_dataset = filtered_train.map(format_instruction, remove_columns=filtered_train.column_names)
print("Formatting validation data...")
val_dataset = filtered_val.map(format_instruction, remove_columns=filtered_val.column_names)

print(f"Formatted {len(train_dataset)} training examples")
print(f"Formatted {len(val_dataset)} validation examples")

# Show example
print("\nExample formatted prompt:")
print(train_dataset[0]['text'][:500] + "...")

# ============================================
# 4. TRAINING ARGUMENTS
# ============================================
print("\n" + "="*60)
print("TRAINING CONFIGURATION")
print("="*60)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,  # Small batch for 4-bit
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch size = 16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=False,
    bf16=True,  # Use bfloat16 for better stability
    optim="paged_adamw_8bit",  # 8-bit optimizer for memory efficiency
    report_to="wandb",
    max_grad_norm=0.3,
    weight_decay=0.001,
)

print(f"Epochs: {training_args.num_train_epochs}")
print(f"Batch size: {training_args.per_device_train_batch_size}")
print(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")

# ============================================
# 5. TRAIN WITH SFT
# ============================================
print("\n" + "="*60)
print("TRAINING")
print("="*60)

# Data collator for completion-only training (only train on assistant response)
response_template = "<|start_header_id|>assistant<|end_header_id|>"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=collator,
    dataset_text_field="text",
    max_seq_length=1024,
)

print("Starting training...")
trainer.train()

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)

# Save model
cached_model_path = os.path.join(MODEL_CACHE_DIR, "citation_intent_llama_lora")
print(f"\nSaving LoRA adapters to {cached_model_path}...")
model.save_pretrained(cached_model_path)
tokenizer.save_pretrained(cached_model_path)
print("Model saved successfully!")

# ============================================
# 6. EVALUATION
# ============================================
print("\n" + "="*60)
print("EVALUATION")
print("="*60)

eval_results = trainer.evaluate()
print(f"Validation Loss: {eval_results['eval_loss']:.4f}")

# Save results
results_path = os.path.join(OUTPUT_DIR, "llama_sft_results.json")
with open(results_path, 'w') as f:
    json.dump({
        'eval_loss': eval_results['eval_loss'],
        'train_samples': len(filtered_train),
        'val_samples': len(filtered_val),
        'num_labels': num_labels,
        'labels': label_list,
        'model_name': 'Meta-Llama-3.1-8B-Instruct',
        'method': 'SFT with QLoRA'
    }, f, indent=2)

print(f"\nResults saved to {results_path}")

# Finish wandb
wandb.finish()

print("\n" + "="*60)
print("DONE!")
print("="*60)
print(f"LoRA adapters saved to: {cached_model_path}")
print("To use: Load base model + LoRA adapters for inference")
