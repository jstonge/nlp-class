# Weights & Biases (wandb) Setup for Sentiment Analysis

## Quick Start

### 1. Check GPU availability (recommended)
```bash
python check_gpu.py
```
This will verify that your GPU is detected and properly configured.

### 2. Install wandb
```bash
pip install wandb
```

### 3. Login to wandb (first time only)
```bash
wandb login
```
This will prompt you to paste your API key from https://wandb.ai/authorize

### 4. Run the script
```bash
python sent-analysis-02.py
```

The script will automatically:
- Detect if GPU is available
- Use GPU-optimized settings (larger batch size, FP16, etc.)
- Fall back to CPU if no GPU is detected

## Running Offline (No Internet / HPC Environments)

If you're on an HPC cluster or don't have internet access:

```bash
# Option 1: Set environment variable
export WANDB_MODE=offline
python sent-analysis-02.py

# Option 2: Set inline
WANDB_MODE=offline python sent-analysis-02.py
```

Later, when you have internet access, you can sync the offline runs:
```bash
wandb sync wandb/offline-run-*
```

## GPU vs CPU Settings

The script automatically adapts based on hardware:

| Setting | GPU | CPU |
|---------|-----|-----|
| Batch Size | 32 | 8 |
| Eval Batch Size | 64 | 16 |
| Epochs | 3 | 2 |
| FP16 (Mixed Precision) | ✓ | ✗ |
| Max Sequence Length | 128 | 128 |
| Dataloader Workers | 4 | 0 |
| Training Time (5% sample) | 1-3 min | 5-15 min |

## What's Logged to WandB

The script automatically logs:

### 📊 Training Metrics (real-time)
- Training loss
- Evaluation loss
- Evaluation accuracy
- Learning rate
- Epoch progress

### 📈 Final Evaluation Metrics
- Final validation accuracy
- Precision, Recall, F1 for both classes (Negative/Positive)
- Classification report details

### 🔍 Example Predictions
- Interactive table showing predictions on test examples
- Confidence scores and probability distributions

### 📉 Model Comparison
- Naive Bayes vs DistilRoBERTa accuracy
- Bar chart visualization
- Improvement percentage

### ⚙️ Configuration
- Model hyperparameters
- Dataset information (sample size, class balance)
- Training settings
- **Hardware info** (GPU name, memory, device type)

## Viewing Your Results

After running the script, you'll see a URL printed:
```
WandB run completed! View your results at: https://wandb.ai/your-username/...
```

Visit this URL to see:
- Interactive plots of training progress
- Model performance metrics
- Comparison visualizations
- System metrics (CPU, memory usage)

## Tips

### Change Project Name
Edit line 32 in the script:
```python
wandb.init(
    project="your-custom-project-name",  # Change this
    name="run-description-here",         # And this
    ...
)
```

### Track Multiple Experiments
The script will create separate runs for each execution. Compare them in the wandb dashboard!

### Disable WandB Temporarily
```python
# Add at the top of the script
import os
os.environ['WANDB_DISABLED'] = 'true'
```

## Troubleshooting

**Problem**: `wandb: ERROR Error while calling W&B API`
- **Solution**: Check internet connection or use offline mode

**Problem**: Script asks for API key every time
- **Solution**: Run `wandb login` once and save credentials

**Problem**: Too much logging slowing down training
- **Solution**: Increase `logging_steps` in TrainingArguments (line 169)

## Resources

- WandB Documentation: https://docs.wandb.ai/
- WandB Quickstart: https://docs.wandb.ai/quickstart
- Example Dashboards: https://wandb.ai/gallery
