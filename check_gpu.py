#!/usr/bin/env python3
"""
Quick script to check GPU availability and configuration.
Run this before running the main training script.
"""

import torch
import sys

print("="*60)
print("GPU AVAILABILITY CHECK")
print("="*60)

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print(f"\nCUDA Available: {cuda_available}")

if cuda_available:
    print(f"✓ GPU is available!")
    print(f"\nGPU Information:")
    print(f"  Number of GPUs: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        print(f"\n  GPU {i}:")
        print(f"    Name: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"    Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"    Compute Capability: {props.major}.{props.minor}")
        print(f"    Multi Processors: {props.multi_processor_count}")

    # Test tensor creation on GPU
    print(f"\nTesting GPU tensor creation...")
    try:
        test_tensor = torch.randn(1000, 1000).cuda()
        print(f"✓ Successfully created tensor on GPU")
        print(f"  Tensor device: {test_tensor.device}")
        del test_tensor
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"✗ Error creating tensor on GPU: {e}")

    print(f"\nCurrent GPU Memory Usage:")
    print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"  Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

else:
    print("✗ No GPU detected!")
    print("\nPossible reasons:")
    print("  1. No NVIDIA GPU in the system")
    print("  2. CUDA toolkit not installed")
    print("  3. PyTorch installed without CUDA support")
    print("\nTo install PyTorch with CUDA support:")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

# Check PyTorch version
print(f"\n{'='*60}")
print(f"PyTorch Configuration:")
print(f"{'='*60}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Version: {torch.version.cuda if cuda_available else 'N/A'}")
print(f"cuDNN Version: {torch.backends.cudnn.version() if cuda_available else 'N/A'}")

print(f"\n{'='*60}")
if cuda_available:
    print("✓ Ready to train on GPU!")
    sys.exit(0)
else:
    print("⚠ Will use CPU (slow training)")
    sys.exit(1)
