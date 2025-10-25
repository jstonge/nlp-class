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
# Create directories for saving models and results
MODEL_CACHE_DIR = "./model_cache"
RESULTS_DIR = "./results"
OUTPUT_DIR = "./output"

Path(MODEL_CACHE_DIR).mkdir(exist_ok=True)
Path(RESULTS_DIR).mkdir(exist_ok=True)
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# Initialize Weights & Biases
# Set WANDB_MODE=offline environment variable if you want to run offline
wandb.init(
    project="sentiment-analysis-sst2",
    name="distilroberta-sst2-gpu",
    config={
        "model_name": "distilroberta-base",
        "dataset": "glue/sst2",
        "sample_ratio": 1.0,  # Will be updated below
        "learning_rate": 2e-5,
    }
)

# Sample ratio - set to 1.0 to use full dataset, or smaller value (e.g., 0.05, 0.2) to sample
SAMPLE_RATIO = 1.0  # Use 0.05 for quick tests, 0.2 for medium, 1.0 for full dataset

# ============================================
# 1. LOAD AND SAMPLE DATASET
# ============================================
print("="*60)
print("LOADING & SAMPLING DATASET")
print("="*60)

dataset = load_dataset("glue", "sst2")

# Stratified sampling function
def stratified_sample(dataset_split, sample_ratio=0.2):
    if sample_ratio >= 1.0:
        # Use full dataset
        return dataset_split

    labels = dataset_split['label']
    indices = list(range(len(labels)))

    sampled_indices, _ = train_test_split(
        indices,
        train_size=sample_ratio,
        stratify=labels,
        random_state=42
    )

    return dataset_split.select(sampled_indices)

sampled_train = stratified_sample(dataset['train'], SAMPLE_RATIO)
sampled_val = stratified_sample(dataset['validation'], SAMPLE_RATIO)

print(f"Original: {len(dataset['train'])} train, {len(dataset['validation'])} val")
print(f"Sampled:  {len(sampled_train)} train, {len(sampled_val)} val")
print(f"Sampling ratio: {SAMPLE_RATIO*100}%")

# Verify class balance
train_neg = sum(1 for l in sampled_train['label'] if l==0)
train_pos = sum(1 for l in sampled_train['label'] if l==1)
print(f"\nClass balance: Negative={train_neg}, Positive={train_pos}")

# Log dataset info to wandb
wandb.config.update({
    "sample_ratio": SAMPLE_RATIO,
    "train_samples": len(sampled_train),
    "val_samples": len(sampled_val),
    "train_negative": train_neg,
    "train_positive": train_pos
})

# ============================================
# 2. LOAD MODEL (Use DistilRoBERTa - 40% faster!)
# ============================================
print("\n" + "="*60)
print("LOADING MODEL")
print("="*60)

# Use distilroberta-base instead of roberta-base (much faster!)
model_name = "distilroberta-base"
print(f"Model: {model_name}")

# Try to load from cache first, otherwise download
cached_model_path = os.path.join(MODEL_CACHE_DIR, "distilroberta_sst2")
if os.path.exists(cached_model_path) and os.path.exists(os.path.join(cached_model_path, "config.json")):
    print(f"Loading cached model from {cached_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(cached_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(cached_model_path)
else:
    print(f"Downloading model from HuggingFace...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
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
    print("WARNING: No GPU detected! Training will be slow on CPU.")
    USE_GPU = False

# Log device info to wandb
wandb.config.update({
    "device": str(device),
    "gpu_available": USE_GPU,
    "gpu_name": torch.cuda.get_device_name(0) if USE_GPU else "N/A"
})

# ============================================
# 3. TOKENIZE
# ============================================
print("\n" + "="*60)
print("TOKENIZING")
print("="*60)

def tokenize_function(examples):
    return tokenizer(
        examples['sentence'],
        padding='max_length',
        truncation=True,
        max_length=128  # Increased for GPU (can handle longer sequences)
    )

tokenized_train = sampled_train.map(tokenize_function, batched=True)
tokenized_val = sampled_val.map(tokenize_function, batched=True)

print("Tokenization complete!")

# ============================================
# 4. METRICS
# ============================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    return {'accuracy': accuracy}

# ============================================
# 5. TRAINING ARGUMENTS (Optimized for GPU)
# ============================================
print("\n" + "="*60)
print("TRAINING CONFIGURATION")
print("="*60)

# GPU-optimized settings
if USE_GPU:
    batch_size = 32  # Larger batch size for GPU
    eval_batch_size = 64
    num_epochs = 3
    use_fp16 = True  # Mixed precision for faster training on GPU
    logging_steps = 20  # More frequent logging
else:
    batch_size = 8  # Smaller batch for CPU
    eval_batch_size = 16
    num_epochs = 2
    use_fp16 = False
    logging_steps = 50

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
    save_total_limit=1,
    report_to='wandb',  # Enable wandb logging
    fp16=use_fp16,  # Mixed precision on GPU
    dataloader_num_workers=4 if USE_GPU else 0,  # Parallel data loading on GPU
    gradient_accumulation_steps=1,
    warmup_steps=100,  # Learning rate warmup
)

print(f"Device: {device}")
print(f"Epochs: {training_args.num_train_epochs}")
print(f"Batch size: {training_args.per_device_train_batch_size}")
print(f"Mixed precision (FP16): {use_fp16}")
print(f"Training samples: {len(tokenized_train)}")
print(f"Steps per epoch: {len(tokenized_train) // training_args.per_device_train_batch_size}")

# Update wandb config with training settings
wandb.config.update({
    "batch_size": batch_size,
    "eval_batch_size": eval_batch_size,
    "num_epochs": num_epochs,
    "fp16": use_fp16,
    "max_length": 128
})

# ============================================
# 6. TRAIN
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
if USE_GPU:
    print("(Training on GPU - should take 1-3 minutes with GPU acceleration)\n")
else:
    print("(Training on CPU - this will take 5-15 minutes)\n")

train_result = trainer.train()

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)

# Save the fine-tuned model and tokenizer
print(f"\nSaving model to {cached_model_path}...")
model.save_pretrained(cached_model_path)
tokenizer.save_pretrained(cached_model_path)
print("Model saved successfully!")

# ============================================
# 7. EVALUATE
# ============================================
print("\n" + "="*60)
print("EVALUATION")
print("="*60)

eval_results = trainer.evaluate()
print(f"Validation Accuracy: {eval_results['eval_accuracy']:.4f}")

# Detailed predictions
predictions = trainer.predict(tokenized_val)
pred_labels = np.argmax(predictions.predictions, axis=-1)
true_labels = predictions.label_ids

print("\nClassification Report:")
classification_rep = classification_report(
    true_labels,
    pred_labels,
    target_names=['Negative', 'Positive'],
    output_dict=True
)
print(classification_report(
    true_labels,
    pred_labels,
    target_names=['Negative', 'Positive']
))

# Log evaluation metrics to wandb
wandb.log({
    "final_eval_accuracy": eval_results['eval_accuracy'],
    "final_eval_loss": eval_results.get('eval_loss', 0),
    "precision_negative": classification_rep['Negative']['precision'],
    "recall_negative": classification_rep['Negative']['recall'],
    "f1_negative": classification_rep['Negative']['f1-score'],
    "precision_positive": classification_rep['Positive']['precision'],
    "recall_positive": classification_rep['Positive']['recall'],
    "f1_positive": classification_rep['Positive']['f1-score'],
})

# Save evaluation results
eval_output_path = os.path.join(OUTPUT_DIR, "evaluation_results.json")
with open(eval_output_path, 'w') as f:
    json.dump({
        'validation_accuracy': eval_results['eval_accuracy'],
        'classification_report': classification_rep,
        'sample_ratio': SAMPLE_RATIO,
        'num_train_samples': len(sampled_train),
        'num_val_samples': len(sampled_val),
        'model_name': model_name
    }, f, indent=2)
print(f"\nEvaluation results saved to {eval_output_path}")

# ============================================
# 8. TEST ON EXAMPLES
# ============================================
print("\n" + "="*60)
print("EXAMPLE PREDICTIONS")
print("="*60)

# Use the same device (GPU or CPU) for predictions
model.to(device)

def predict_sentiment(text):
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)  # Move inputs to GPU if available

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probs, dim=-1).item()

    return prediction, probs[0].cpu().numpy()  # Move back to CPU for numpy

test_reviews = [
    "This movie was absolutely fantastic!",
    "Terrible waste of time.",
    "It was okay, nothing special.",
    "Masterpiece! Must watch!",
    "Boring and predictable."
]

# Collect predictions for saving
example_predictions = []
for review in test_reviews:
    pred, probs = predict_sentiment(review)
    label = "Positive" if pred == 1 else "Negative"
    print(f"\n\"{review}\"")
    print(f"  → {label} (confidence: {probs[pred]:.3f})")

    example_predictions.append({
        'text': review,
        'predicted_label': label,
        'confidence': float(probs[pred]),
        'probabilities': {
            'negative': float(probs[0]),
            'positive': float(probs[1])
        }
    })

# Save example predictions
examples_output_path = os.path.join(OUTPUT_DIR, "example_predictions.json")
with open(examples_output_path, 'w') as f:
    json.dump(example_predictions, f, indent=2)
print(f"\nExample predictions saved to {examples_output_path}")

# Log example predictions as a table to wandb
example_table = wandb.Table(columns=["Text", "Predicted", "Confidence", "Neg Prob", "Pos Prob"])
for pred in example_predictions:
    example_table.add_data(
        pred['text'],
        pred['predicted_label'],
        pred['confidence'],
        pred['probabilities']['negative'],
        pred['probabilities']['positive']
    )
wandb.log({"example_predictions": example_table})

# ============================================
# 9. QUICK NAIVE BAYES COMPARISON
# ============================================
print("\n" + "="*60)
print("COMPARISON WITH NAIVE BAYES")
print("="*60)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

nb_model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
    ('clf', MultinomialNB(alpha=0.1))
])

nb_model.fit(sampled_train['sentence'], sampled_train['label'])
nb_pred = nb_model.predict(sampled_val['sentence'])
nb_acc = accuracy_score(sampled_val['label'], nb_pred)

print(f"Naive Bayes:     {nb_acc:.4f}")
print(f"DistilRoBERTa:   {eval_results['eval_accuracy']:.4f}")
print(f"Improvement:     {(eval_results['eval_accuracy'] - nb_acc)*100:+.2f}%")

# Log model comparison to wandb
wandb.log({
    "naive_bayes_accuracy": nb_acc,
    "distilroberta_accuracy": eval_results['eval_accuracy'],
    "improvement_percentage": (eval_results['eval_accuracy'] - nb_acc) * 100
})

# Create a comparison bar chart for wandb
comparison_data = [
    ["Naive Bayes", nb_acc],
    ["DistilRoBERTa", eval_results['eval_accuracy']]
]
comparison_table = wandb.Table(data=comparison_data, columns=["Model", "Accuracy"])
wandb.log({
    "model_comparison_chart": wandb.plot.bar(
        comparison_table, "Model", "Accuracy",
        title="Model Accuracy Comparison"
    )
})

# Save model comparison results
comparison_output_path = os.path.join(OUTPUT_DIR, "model_comparison.json")
with open(comparison_output_path, 'w') as f:
    json.dump({
        'naive_bayes_accuracy': float(nb_acc),
        'distilroberta_accuracy': eval_results['eval_accuracy'],
        'improvement_percentage': float((eval_results['eval_accuracy'] - nb_acc) * 100),
        'sample_ratio': SAMPLE_RATIO,
        'num_train_samples': len(sampled_train),
        'num_val_samples': len(sampled_val)
    }, f, indent=2)
print(f"\nModel comparison saved to {comparison_output_path}")

print("\n" + "="*60)
print("DONE! 🚀")
print("="*60)
print(f"\nAll results saved to: {OUTPUT_DIR}/")
print(f"Trained model cached at: {cached_model_path}")
print("\nGenerated files:")
print(f"  - {eval_output_path}")
print(f"  - {examples_output_path}")
print(f"  - {comparison_output_path}")

# Finish wandb run
wandb.finish()
print(f"\nWandB run completed! View your results at: {wandb.run.url if wandb.run else 'N/A'}")