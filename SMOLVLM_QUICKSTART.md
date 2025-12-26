# SmolVLM Quick Start Guide

Get started with fine-tuning SmolVLM for aesthetic score prediction in 5 minutes!

## 🚀 Quick Start (TL;DR)

```bash
# 1. Install dependencies
./setup_smolvlm.sh

# 2. Configure your data paths
nano config/training/smolvlm_config.yaml

# 3. Train the model
python scripts/train_smolvlm.py

# 4. Run inference
python scripts/inference_smolvlm.py \
  --image path/to/image.jpg \
  --adapter outputs/smolvlm-aesthetic-scorer \
  --pretty-print
```

## 📋 Prerequisites

- **GPU**: 12GB+ VRAM (RTX 3060, 4060 Ti, 3090, 4090, L4, A10, etc.)
- **Python**: 3.8+
- **CUDA**: 11.8+
- **Storage**: 50GB+ free space

## 📦 Installation

### Option 1: Automated Setup (Recommended)

```bash
./setup_smolvlm.sh
```

This script will:
- Install PyTorch with CUDA
- Install all required dependencies
- Install Flash Attention 2 (optional)
- Verify installation
- Make scripts executable

### Option 2: Manual Setup

```bash
# Install PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Install Flash Attention (optional but recommended)
pip install flash-attn --no-build-isolation
```

See [SMOLVLM_SETUP.md](docs/SMOLVLM_SETUP.md) for detailed setup instructions.

## ⚙️ Configuration

Edit `config/training/smolvlm_config.yaml`:

```yaml
dataset:
  pseudolabels_path: "results/pseudolabels.json"  # Your pseudolabels
  images_dir: "src/data/images"  # Your images

training:
  output_dir: "outputs/smolvlm-aesthetic-scorer"
  num_train_epochs: 3
  per_device_train_batch_size: 2
```

## 🎯 Training

### Basic Training

```bash
python scripts/train_smolvlm.py
```

### Training with Custom Parameters

```bash
python scripts/train_smolvlm.py \
  --pseudolabels-path results/pseudolabels.json \
  --images-dir src/data/images \
  --epochs 3 \
  --batch-size 2 \
  --learning-rate 1e-4
```

### Training with Weights & Biases

```bash
python scripts/train_smolvlm.py \
  --use-wandb \
  --wandb-project "aesthetic-scorer" \
  --wandb-run-name "experiment-1"
```

### Quick Test (Small Dataset)

```bash
python scripts/train_smolvlm.py --max-samples 100 --epochs 1
```

## 🔮 Inference

### Single Image

```bash
python scripts/inference_smolvlm.py \
  --image path/to/image.jpg \
  --adapter outputs/smolvlm-aesthetic-scorer \
  --pretty-print
```

Output:
```
Image: path/to/image.jpg
  Impact: 7.5
  Style: 7.0
  Composition: 8.0
  Lighting: 7.5
  Color Balance: 7.0
  Reasoning: The image demonstrates strong visual impact...
```

### Batch Processing

```bash
python scripts/inference_smolvlm.py \
  --image-dir path/to/images \
  --adapter outputs/smolvlm-aesthetic-scorer \
  --output results.json
```

### Programmatic Usage

```python
from src.training.smolvlm_trainer import load_trained_model
from scripts.inference_smolvlm import generate_scores_from_image

# Load model
model, processor = load_trained_model(
    base_model_id="HuggingFaceTB/SmolVLM-Instruct",
    adapter_path="outputs/smolvlm-aesthetic-scorer"
)

# Generate scores
result = generate_scores_from_image(
    model=model,
    processor=processor,
    image_path="photo.jpg",
    system_message="You are an expert photography critic...",
    user_message="Analyze this photograph."
)

print(f"Impact: {result['impact']}")
print(f"Composition: {result['composition']}")
```

## 📊 Expected Results

### Training Time

| GPU | Batch Size | Samples | Time per Epoch |
|-----|------------|---------|----------------|
| RTX 3060 (12GB) | 1 | 5,000 | ~45 min |
| RTX 3090 (24GB) | 2 | 5,000 | ~25 min |
| RTX 4090 (24GB) | 4 | 5,000 | ~15 min |
| L4 (24GB) | 2 | 5,000 | ~25 min |

### Memory Usage

- **Model (4-bit)**: ~4GB VRAM
- **Activations**: ~2-4GB VRAM
- **Gradients**: ~2-4GB VRAM
- **Total**: ~8-12GB VRAM

### Inference Speed

- **Single image**: ~1-2 seconds
- **Batch (10 images)**: ~10-15 seconds

## 🎓 Examples

### Example 1: Quick Start

```bash
# Run the quick start example
python examples/smolvlm_quickstart.py
```

This trains on 100 samples for 1 epoch (takes ~5 minutes).

### Example 2: Full Training

```bash
# Train on full dataset
python scripts/train_smolvlm.py \
  --pseudolabels-path results/pseudolabels.json \
  --images-dir src/data/images \
  --epochs 3 \
  --output-dir outputs/my-model
```

### Example 3: Inference

```bash
# Run inference example
python examples/smolvlm_inference_example.py
```

## 🐛 Troubleshooting

### Out of Memory

```bash
# Reduce batch size
python scripts/train_smolvlm.py --batch-size 1

# Increase gradient accumulation
python scripts/train_smolvlm.py --gradient-accumulation-steps 16
```

### Flash Attention Not Working

Edit `config/training/smolvlm_config.yaml`:

```yaml
model:
  attn_implementation: "eager"  # Change from "flash_attention_2"
```

### Slow Training

```bash
# Ensure Flash Attention is installed
pip install flash-attn --no-build-isolation

# Use larger batch size if memory allows
python scripts/train_smolvlm.py --batch-size 4
```

## 📚 Documentation

- **[Complete Fine-tuning Guide](docs/SMOLVLM_FINETUNING.md)**: Comprehensive documentation
- **[Setup Guide](docs/SMOLVLM_SETUP.md)**: Detailed installation instructions
- **[Main README](README.md)**: Project overview

## 🎯 What's Next?

1. **Generate Pseudolabels**: If you haven't already
   ```bash
   python scripts/generate_pseudolabels.py
   ```

2. **Fine-tune SmolVLM**: Train on your pseudolabels
   ```bash
   python scripts/train_smolvlm.py
   ```

3. **Evaluate**: Test on new images
   ```bash
   python scripts/inference_smolvlm.py --image test.jpg --adapter outputs/smolvlm-aesthetic-scorer
   ```

4. **Deploy**: Use the model in production
   - Push to Hugging Face Hub
   - Create an API endpoint
   - Integrate into your application

## 💡 Tips

- **Start small**: Test with `--max-samples 100` first
- **Monitor training**: Use `--use-wandb` for experiment tracking
- **Save checkpoints**: Keep multiple checkpoints for comparison
- **Validate early**: Check validation loss after first epoch
- **Test inference**: Try inference during training to check quality

## 🤝 Getting Help

- Check the [Troubleshooting Guide](docs/SMOLVLM_FINETUNING.md#troubleshooting)
- Read the [Complete Documentation](docs/SMOLVLM_FINETUNING.md)
- Open an issue on GitHub

## 🎉 Success!

If you've made it this far, you should have:

✅ Installed all dependencies  
✅ Configured your data paths  
✅ Trained a SmolVLM model  
✅ Generated aesthetic scores for images  

Congratulations! You now have a working aesthetic score prediction model! 🚀

---

**Next Steps**: Check out the [Complete Fine-tuning Guide](docs/SMOLVLM_FINETUNING.md) for advanced usage and optimization tips.

