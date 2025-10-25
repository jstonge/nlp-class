import numpy as np
import torch
import json
import os
from pathlib import Path
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, classification_report
import wandb

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
    name="roberta-large-citation-intent",
    config={
        "model_name": "roberta-large",
        "dataset": "allenai/SciRIFF",
        "task": "citation_intent_classification",
        "learning_rate": 2e-5,
    }
)

# Sample ratio for initial testing
SAMPLE_RATIO = 1.0  # Use full citation classification dataset

# ============================================
# 1. LOAD SCIRIFF DATASET
# ============================================
print("="*60)
print("LOADING SCIRIFF DATASET")
print("="*60)

dataset = load_dataset("allenai/SciRIFF", "4096")

print(f"Dataset splits: {dataset.keys()}")
print(f"Full train size: {len(dataset['train'])}")
print(f"Full validation size: {len(dataset['validation'])}")

# Filter for citation intent classification tasks only
print("\n" + "="*60)
print("FILTERING FOR CITATION INTENT TASKS")
print("="*60)

def filter_citation_tasks(examples):
    # Keep only classification tasks that involve citations
    task_families = examples['metadata']['task_family']
    return ['classification' in tf.lower() and 'citation' in examples['input'][i].lower()
            for i, tf in enumerate(task_families)]

# Apply filter - citation classification only AND single-label only (no JSON arrays)
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

# Sample the filtered dataset
def sample_dataset(dataset_split, sample_ratio=0.01):
    if sample_ratio >= 1.0 or len(dataset_split) == 0:
        return dataset_split

    num_samples = max(1, int(len(dataset_split) * sample_ratio))
    indices = list(range(len(dataset_split)))
    sampled_indices, _ = train_test_split(
        indices,
        train_size=num_samples,
        random_state=42
    )
    return dataset_split.select(sampled_indices)

sampled_train = sample_dataset(filtered_train, SAMPLE_RATIO)
sampled_val = sample_dataset(filtered_val, SAMPLE_RATIO)

print(f"\nSampled train: {len(sampled_train)}")
print(f"Sampled val: {len(sampled_val)}")
print(f"Sample ratio: {SAMPLE_RATIO*100}%")

# Explore the data structure
print("\n" + "="*60)
print("DATASET STRUCTURE")
print("="*60)
example = sampled_train[0]
print(f"Keys: {example.keys()}")
print(f"\nExample input (first 200 chars):\n{example['input'][:200]}...")
print(f"\nExample output (first 200 chars):\n{example['output'][:200]}...")
print(f"\nTask family: {example['metadata']['task_family']}")
print(f"Domains: {example['metadata']['domains']}")

# Log dataset info to wandb
wandb.config.update({
    "train_samples": len(sampled_train),
    "val_samples": len(sampled_val),
    "sample_ratio": SAMPLE_RATIO
})

# Extract unique labels from outputs
print("\n" + "="*60)
print("ANALYZING LABELS")
print("="*60)

unique_labels = set()
for example in sampled_train:
    unique_labels.add(example['output'].strip())
for example in sampled_val:
    unique_labels.add(example['output'].strip())

label_list = sorted(list(unique_labels))
num_labels = len(label_list)
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

print(f"Number of citation intent labels: {num_labels}")
print(f"Labels: {label_list}")

# ============================================
# 2. LOAD MODEL (RoBERTa for classification)
# ============================================
print("\n" + "="*60)
print("LOADING MODEL")
print("="*60)

model_name = "roberta-large"  # Larger model for better performance
print(f"Model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    USE_GPU = True
else:
    print("WARNING: No GPU detected!")
    USE_GPU = False

# Log device info to wandb
wandb.config.update({
    "device": str(device),
    "gpu_available": USE_GPU,
    "gpu_name": torch.cuda.get_device_name(0) if USE_GPU else "N/A"
})

# ============================================
# 3. PREPROCESS DATA
# ============================================
print("\n" + "="*60)
print("PREPROCESSING")
print("="*60)

max_input_length = 512

def preprocess_function(examples):
    # Tokenize inputs
    model_inputs = tokenizer(
        examples['input'],
        max_length=max_input_length,
        truncation=True,
        padding='max_length'
    )

    # Convert text labels to numeric IDs
    model_inputs['labels'] = [label2id[output.strip()] for output in examples['output']]
    return model_inputs

print("Tokenizing training data...")
tokenized_train = sampled_train.map(preprocess_function, batched=True, remove_columns=sampled_train.column_names)
print("Tokenizing validation data...")
tokenized_val = sampled_val.map(preprocess_function, batched=True, remove_columns=sampled_val.column_names)

print("Preprocessing complete!")

# ============================================
# 4. TRAINING ARGUMENTS
# ============================================
print("\n" + "="*60)
print("TRAINING CONFIGURATION")
print("="*60)

if USE_GPU:
    batch_size = 16
    eval_batch_size = 32
    num_epochs = 3
    use_fp16 = True
    logging_steps = 50
else:
    batch_size = 8
    eval_batch_size = 16
    num_epochs = 2
    use_fp16 = False
    logging_steps = 100

# Compute metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    return {'accuracy': accuracy}

training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=eval_batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    logging_steps=logging_steps,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    save_total_limit=2,
    report_to='wandb',
    fp16=use_fp16,
)

print(f"Device: {device}")
print(f"Epochs: {num_epochs}")
print(f"Batch size: {batch_size}")
print(f"Mixed precision (FP16): {use_fp16}")
print(f"Training samples: {len(tokenized_train)}")

wandb.config.update({
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    "fp16": use_fp16,
    "max_input_length": max_input_length,
    "num_labels": num_labels
})

# ============================================
# 5. TRAIN
# ============================================
print("\n" + "="*60)
print("TRAINING")
print("="*60)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics
)

print("Starting training...")
train_result = trainer.train()

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)

# Save model
cached_model_path = os.path.join(MODEL_CACHE_DIR, "citation_intent_roberta")
print(f"\nSaving model to {cached_model_path}...")
model.save_pretrained(cached_model_path)
tokenizer.save_pretrained(cached_model_path)

# Save label mappings
with open(os.path.join(cached_model_path, "label_mappings.json"), 'w') as f:
    json.dump({"label2id": label2id, "id2label": id2label, "labels": label_list}, f, indent=2)
print("Model and label mappings saved successfully!")

# ============================================
# 6. EVALUATION
# ============================================
print("\n" + "="*60)
print("EVALUATION")
print("="*60)

eval_results = trainer.evaluate()
print(f"Validation Accuracy: {eval_results['eval_accuracy']:.4f}")
print(f"Validation Loss: {eval_results['eval_loss']:.4f}")

# Detailed classification report
predictions = trainer.predict(tokenized_val)
pred_labels = np.argmax(predictions.predictions, axis=-1)
true_labels = predictions.label_ids

print("\nClassification Report:")
# Get unique labels present in validation set
unique_val_labels = sorted(set(true_labels.tolist() + pred_labels.tolist()))
target_names_present = [id2label[i] for i in unique_val_labels]
print(classification_report(true_labels, pred_labels, labels=unique_val_labels, target_names=target_names_present))

# ============================================
# 7. TEST ON EXAMPLES
# ============================================
print("\n" + "="*60)
print("EXAMPLE PREDICTIONS")
print("="*60)

model.to(device)
model.eval()

# Test on a few validation examples
test_examples = sampled_val.select(range(min(5, len(sampled_val))))

for idx, example in enumerate(test_examples):
    print(f"\n--- Example {idx+1} ---")
    print(f"Input: {example['input'][:300]}...")
    print(f"\nExpected label: {example['output']}")

    inputs = tokenizer(example['input'], return_tensors='pt', max_length=max_input_length, truncation=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_id].item()

    predicted_label = id2label[pred_id]
    print(f"Predicted label: {predicted_label} (confidence: {confidence:.3f})")
    print("-" * 60)

# ============================================
# 8. SAVE RESULTS
# ============================================
results_path = os.path.join(OUTPUT_DIR, "citation_intent_results.json")
with open(results_path, 'w') as f:
    json.dump({
        'eval_accuracy': eval_results['eval_accuracy'],
        'eval_loss': eval_results['eval_loss'],
        'sample_ratio': SAMPLE_RATIO,
        'train_samples': len(sampled_train),
        'val_samples': len(sampled_val),
        'num_labels': num_labels,
        'labels': label_list,
        'model_name': model_name
    }, f, indent=2)

print(f"\nResults saved to {results_path}")

# Finish wandb
wandb.finish()
print(f"\nWandB run completed!")

print("\n" + "="*60)
print("DONE!")
print("="*60)
