# SmolVLM Setup Guide

This guide will help you set up your environment for fine-tuning SmolVLM on your aesthetic score pseudolabels.

## Prerequisites

Before starting, ensure you have:

- Python 3.8 or higher
- CUDA 11.8+ (for GPU support)
- 12GB+ GPU VRAM (recommended: 16GB+)
- 50GB+ free disk space

## Step-by-Step Setup

### 1. Install PyTorch with CUDA Support

First, install PyTorch with CUDA support. Visit [pytorch.org](https://pytorch.org/get-started/locally/) for the latest installation command, or use:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify installation:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.x.x
CUDA available: True
```

### 2. Install Core Dependencies

Install the main dependencies:

```bash
pip install transformers datasets accelerate peft trl bitsandbytes
```

### 3. Install Flash Attention 2 (Recommended)

Flash Attention 2 significantly speeds up training. Install it with:

```bash
pip install flash-attn --no-build-isolation
```

**Note:** This may take 5-10 minutes to compile. If it fails, you can skip it and the model will use standard attention (slower but works).

**Troubleshooting Flash Attention:**

If installation fails, try:

```bash
# Install with specific CUDA version
pip install flash-attn --no-build-isolation --no-cache-dir

# Or install from pre-built wheels
pip install flash-attn --no-build-isolation --find-links https://github.com/Dao-AILab/flash-attention/releases
```

If all else fails, you can disable Flash Attention by editing `config/training/smolvlm_config.yaml`:

```yaml
model:
  attn_implementation: "eager"  # Change from "flash_attention_2"
```

### 4. Install Remaining Dependencies

Install all other dependencies:

```bash
pip install -r requirements.txt
```

### 5. Verify Installation

Run the verification script:

```bash
python -c "
import torch
import transformers
import trl
import peft
import bitsandbytes

print('✓ PyTorch:', torch.__version__)
print('✓ Transformers:', transformers.__version__)
print('✓ TRL:', trl.__version__)
print('✓ PEFT:', peft.__version__)
print('✓ BitsAndBytes:', bitsandbytes.__version__)
print('✓ CUDA available:', torch.cuda.is_available())

try:
    import flash_attn
    print('✓ Flash Attention:', flash_attn.__version__)
except ImportError:
    print('⚠ Flash Attention: Not installed (optional)')

print('\nAll required packages installed successfully!')
"
```

### 6. Configure Your Paths

Edit `config/training/smolvlm_config.yaml` to set your data paths:

```yaml
dataset:
  pseudolabels_path: "results/pseudolabels.json"  # Path to your pseudolabels
  images_dir: "src/data/images"  # Directory with your images
```

### 7. Test with Small Dataset

Before running full training, test with a small subset:

```bash
python scripts/train_smolvlm.py --max-samples 10 --epochs 1
```

This should complete in a few minutes and verify everything is working.

## Common Installation Issues

### Issue: CUDA Out of Memory

**Solution:** Ensure no other processes are using the GPU:

```bash
# Check GPU usage
nvidia-smi

# Kill Python processes if needed
pkill -9 python
```

### Issue: BitsAndBytes Not Found

**Solution:** Install from source:

```bash
pip install bitsandbytes --no-cache-dir
```

### Issue: Transformers Version Conflict

**Solution:** Upgrade to latest version:

```bash
pip install --upgrade transformers
```

### Issue: Permission Denied on Scripts

**Solution:** Make scripts executable:

```bash
chmod +x scripts/*.py
chmod +x examples/*.py
```

## Environment Variables

For better performance, set these environment variables:

```bash
# Enable TF32 for faster training on Ampere GPUs
export TORCH_ALLOW_TF32=1

# Disable tokenizers parallelism warning
export TOKENIZERS_PARALLELISM=false

# Set HuggingFace cache directory (optional)
export HF_HOME=/path/to/cache

# For Weights & Biases (optional)
export WANDB_API_KEY=your_api_key
```

Add these to your `~/.bashrc` or `~/.zshrc` to make them permanent.

## GPU Memory Requirements

Here's what you can expect for different GPU configurations:

| GPU | VRAM | Batch Size | Gradient Accumulation | Status |
|-----|------|------------|----------------------|--------|
| RTX 3060 | 12GB | 1 | 16 | ✓ Works |
| RTX 3090 | 24GB | 2 | 8 | ✓ Recommended |
| RTX 4090 | 24GB | 4 | 4 | ✓ Fast |
| L4 | 24GB | 2 | 8 | ✓ Recommended |
| A10 | 24GB | 2 | 8 | ✓ Recommended |
| A100 | 40GB | 8 | 2 | ✓ Very Fast |

**Note:** With 4-bit quantization (QLoRA), the model uses ~4GB VRAM. The rest is for activations and gradients.

## Next Steps

Once setup is complete:

1. **Generate Pseudolabels** (if you haven't already):
   ```bash
   python scripts/generate_pseudolabels.py
   ```

2. **Start Training**:
   ```bash
   python scripts/train_smolvlm.py
   ```

3. **Monitor Training**:
   - TensorBoard: `tensorboard --logdir outputs/smolvlm-aesthetic-scorer/logs`
   - W&B: Enable with `--use-wandb`

4. **Run Inference**:
   ```bash
   python scripts/inference_smolvlm.py --image test.jpg --adapter outputs/smolvlm-aesthetic-scorer
   ```

## Additional Resources

- [Main README](../README.md)
- [SmolVLM Fine-tuning Guide](SMOLVLM_FINETUNING.md)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [PEFT Documentation](https://huggingface.co/docs/peft)

## Getting Help

If you encounter issues:

1. Check the [Troubleshooting section](SMOLVLM_FINETUNING.md#troubleshooting) in the main guide
2. Verify all dependencies are installed correctly
3. Try with a smaller dataset (`--max-samples 10`)
4. Check GPU memory usage with `nvidia-smi`
5. Open an issue on GitHub with error details

Happy fine-tuning! 🚀

