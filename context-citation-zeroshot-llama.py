import torch
import json
import os
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from tqdm import tqdm
from enum import Enum
from outlines import from_transformers

# Force use of specific GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# ============================================
# 0. SETUP
# ============================================
OUTPUT_DIR = "./output"
Path(OUTPUT_DIR).mkdir(exist_ok=True)

print("="*60)
print("ZERO-SHOT CITATION INTENT WITH LLAMA 3.1")
print("="*60)

# ============================================
# 1. LOAD DATASET
# ============================================
print("\n" + "="*60)
print("LOADING SCIRIFF DATASET")
print("="*60)

dataset = load_dataset("allenai/SciRIFF", "4096")

# Filter for citation classification only (single-label)
filtered_val = dataset['validation'].filter(
    lambda x: 'classification' in x['metadata']['task_family'].lower() and
              'citation' in x['input'].lower() and
              not x['output'].strip().startswith('[')
)

print(f"Filtered validation: {len(filtered_val)} (citation classification only)")

# Take a sample for faster testing
SAMPLE_SIZE = 200  # Start with 200 examples
test_data = filtered_val.select(range(min(SAMPLE_SIZE, len(filtered_val))))
print(f"Testing on {len(test_data)} examples")

# ============================================
# 2. LOAD LLAMA MODEL
# ============================================
print("\n" + "="*60)
print("LOADING LLAMA 3.1 8B")
print("="*60)

model_path = "/gpfs1/llm/llama-3.1-hf/Meta-Llama-3.1-8B-Instruct"
print(f"Model path: {model_path}")

# Load model with Outlines (matching notebook example)
model = from_transformers(
    AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda"),
    AutoTokenizer.from_pretrained(model_path)
)

print(f"Model loaded with Outlines")

# ============================================
# 3. DEFINE CITATION INTENT ENUM
# ============================================

# Define citation intent as Enum (as per notebook example)
class CitationIntent(str, Enum):
    BACKGROUND = "Background"
    COMPARE_OR_CONTRAST = "CompareOrContrast"
    EXTENDS = "Extends"
    FUTURE_WORK = "FutureWork"
    METHOD = "Method"
    MOTIVATION = "Motivation"
    RESULT = "Result"
    USES = "Uses"

LABELS = [intent.value for intent in CitationIntent]

def create_prompt(citation_text):
    """Create detailed zero-shot prompt for Llama following best practices"""
    # Extract just the citation context (remove task instructions)
    if "Section Title:" in citation_text:
        content = citation_text.split("Section Title:")[1].strip()
        # Remove labels but keep structure
        content = content.replace("Context before the citation:", "")
        content = content.replace("Citation Sentence:", "")
        content = content.replace("Context after the citation:", "")
        content = " ".join(content.split())
    else:
        content = citation_text

    prompt = f"""You are an assistant trained to identify the intent behind citations in academic papers.

Definitions:
- "Background": The citation provides foundational context or prior work (e.g., "Previous studies have shown...").
- "Method": The citation describes a methodology, algorithm, or technical approach being used or referenced.
- "Result": The citation references specific findings, experimental results, or performance metrics.
- "CompareOrContrast": The citation explicitly compares or contrasts the current work with the cited work (e.g., "Unlike Smith et al., we...").
- "Uses": The current paper applies or builds upon methods/tools from the cited work (e.g., "We used their framework...").
- "Extends": The current paper extends, improves, or builds upon the cited work (e.g., "Building on their approach...").
- "Motivation": The citation provides rationale or justification for the research direction (e.g., "Motivated by their findings...").
- "FutureWork": The citation is referenced as a direction for future research.

Citation Context:
"{content[:1000]}"

Instructions:
1. Read the citation in context.
2. Determine which ONE category best describes the citation's primary intent.
3. Choose from: Background, Method, Result, CompareOrContrast, Uses, Extends, Motivation, FutureWork

Answer with ONLY the category name:"""

    return prompt

# No separate setup needed - we'll call model directly with CitationIntent enum

# ============================================
# 5. RUN ZERO-SHOT INFERENCE
# ============================================
print("\n" + "="*60)
print("RUNNING ZERO-SHOT INFERENCE")
print("="*60)

predictions = []
true_labels = []

for i, example in enumerate(tqdm(test_data, desc="Predicting")):
    prompt = create_prompt(example['input'])

    # Generate with constrained output using Enum (as per notebook example)
    predicted_label = model(prompt, CitationIntent)

    predictions.append(predicted_label)
    true_labels.append(example['output'].strip())

    # Show first few examples
    if i < 3:
        print(f"\n--- Example {i+1} ---")
        print(f"True: {example['output'].strip()}")
        print(f"Predicted: {predicted_label}")

# ============================================
# 5. EVALUATE
# ============================================
print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)

# Convert to label IDs for metrics
unique_labels = sorted(list(set(true_labels + predictions)))
label2id = {label: i for i, label in enumerate(unique_labels)}

true_ids = [label2id[label] for label in true_labels]
pred_ids = [label2id[label] for label in predictions]

accuracy = accuracy_score(true_ids, pred_ids)
print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(true_ids, pred_ids, target_names=unique_labels, zero_division=0))

# Save results
results = {
    'accuracy': float(accuracy),
    'num_samples': len(test_data),
    'model': 'Meta-Llama-3.1-8B-Instruct',
    'method': 'zero-shot',
    'predictions': [
        {'true': t, 'predicted': p}
        for t, p in zip(true_labels, predictions)
    ]
}

results_path = os.path.join(OUTPUT_DIR, "llama_zeroshot_results.json")
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {results_path}")

print("\n" + "="*60)
print("DONE!")
print("="*60)
