# Fine-tuning SmolVLM for Aesthetic Score Prediction

This guide walks you through fine-tuning the SmolVLM Vision Language Model on your pseudolabeled aesthetic scores dataset using the TRL library with QLoRA for efficient training on consumer GPUs.

## 📋 Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset Format](#dataset-format)
- [Configuration](#configuration)
- [Training](#training)
- [Inference](#inference)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

## 🌟 Overview

SmolVLM is a highly performant and memory-efficient Vision Language Model (2B parameters) that achieves state-of-the-art performance while being trainable on consumer GPUs. This pipeline enables you to:

- Fine-tune SmolVLM on your custom pseudolabeled aesthetic scores
- Use QLoRA (4-bit quantization) for efficient training
- Train on consumer GPUs (tested on L4, RTX 3090, RTX 4090)
- Generate aesthetic scores for new images based on 5 criteria:
  - **Impact**: Emotional response and memorability
  - **Style**: Artistic expression and creative vision
  - **Composition**: Visual arrangement and balance
  - **Lighting**: Quality and effectiveness of illumination
  - **Color Balance**: Harmony of colors

### Key Features

- ✅ **Memory Efficient**: QLoRA with 4-bit quantization
- ✅ **Fast Training**: Flash Attention 2 support
- ✅ **Flexible**: Easy configuration via YAML
- ✅ **Production Ready**: Includes inference and evaluation scripts
- ✅ **Well Documented**: Comprehensive examples and guides

## 🔧 Prerequisites

### Hardware Requirements

**Minimum:**
- GPU: 12GB VRAM (e.g., RTX 3060, RTX 4060 Ti)
- RAM: 16GB system RAM
- Storage: 20GB free space

**Recommended:**
- GPU: 16GB+ VRAM (e.g., RTX 3090, RTX 4090, L4, A10)
- RAM: 32GB system RAM
- Storage: 50GB+ free space

### Software Requirements

- Python 3.8+
- CUDA 11.8+ (for GPU support)
- PyTorch 2.0+

## 📦 Installation

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Install Flash Attention 2 (optional but recommended):**

```bash
pip install flash-attn --no-build-isolation
```

Note: Flash Attention requires CUDA and may take several minutes to compile.

3. **Verify installation:**

```bash
python -c "import torch; import transformers; import trl; print('All packages installed successfully!')"
```

## 📊 Dataset Format

Your pseudolabels should be in JSON format with the following structure:

```json
[
  {
    "impact": 6.5,
    "style": 7.0,
    "composition": 7.5,
    "lighting": 7.0,
    "color_balance": 6.0,
    "reasoning": "The image possesses a strong initial impact...",
    "image_path": "/path/to/image.jpg",
    "ava_score": 5.088,
    "image_id": "102954"
  },
  {
    "impact": 6.5,
    "style": 5.0,
    "composition": 7.5,
    "lighting": 6.0,
    "color_balance": 7.0,
    "reasoning": "The image possesses a strong, immediate impact...",
    "image_path": "/path/to/image2.jpg",
    "ava_score": 5.326,
    "image_id": "192744"
  }
]
```

### Required Fields

- `impact`: Float (1-10)
- `style`: Float (1-10)
- `composition`: Float (1-10)
- `lighting`: Float (1-10)
- `color_balance`: Float (1-10)
- `reasoning`: String (explanation of scores)
- `image_path`: String (path to image file)

### Optional Fields

- `ava_score`: Float (overall aesthetic score)
- `image_id`: String (unique identifier)
- `raw_response`: String (original model response)

## ⚙️ Configuration

The training pipeline is configured via `config/training/smolvlm_config.yaml`. Here are the key sections:

### Model Configuration

```yaml
model:
  model_id: "HuggingFaceTB/SmolVLM-Instruct"
  torch_dtype: "bfloat16"
  attn_implementation: "flash_attention_2"
  
  quantization:
    load_in_4bit: true  # Enable QLoRA
    bnb_4bit_use_double_quant: true
    bnb_4bit_quant_type: "nf4"
    bnb_4bit_compute_dtype: "bfloat16"
```

### LoRA Configuration

```yaml
lora:
  r: 8  # LoRA rank (higher = more parameters)
  lora_alpha: 8
  lora_dropout: 0.1
  use_dora: true  # Use DoRA for better performance
```

### Training Configuration

```yaml
training:
  output_dir: "outputs/smolvlm-aesthetic-scorer"
  num_train_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 1.0e-4
  
  # Evaluation
  eval_strategy: "steps"
  eval_steps: 100
  
  # Checkpointing
  save_strategy: "steps"
  save_steps: 100
  save_total_limit: 3
```

### Dataset Configuration

```yaml
dataset:
  pseudolabels_path: "results/pseudolabels.json"
  images_dir: "src/data/images"
  
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  random_seed: 42
```

## 🚀 Training

### Basic Training

Run training with default configuration:

```bash
python scripts/train_smolvlm.py
```

### Custom Configuration

Use a custom config file:

```bash
python scripts/train_smolvlm.py --config path/to/custom_config.yaml
```

### Override Parameters

Override specific parameters via command line:

```bash
python scripts/train_smolvlm.py \
  --learning-rate 5e-5 \
  --batch-size 4 \
  --epochs 5 \
  --output-dir outputs/my-model
```

### Training with Weights & Biases

Enable W&B logging for experiment tracking:

```bash
python scripts/train_smolvlm.py \
  --use-wandb \
  --wandb-project "smolvlm-aesthetic" \
  --wandb-run-name "experiment-1"
```

### Training on Limited Data

For testing or quick iterations:

```bash
python scripts/train_smolvlm.py --max-samples 1000
```

### Full Training Example

```bash
python scripts/train_smolvlm.py \
  --pseudolabels-path results/pseudolabels.json \
  --images-dir src/data/images \
  --output-dir outputs/smolvlm-v1 \
  --learning-rate 1e-4 \
  --batch-size 2 \
  --gradient-accumulation-steps 8 \
  --epochs 3 \
  --use-wandb \
  --wandb-project "aesthetic-scorer"
```

### Training Output

During training, you'll see:

```
================================================================================
SmolVLM Fine-tuning Pipeline
================================================================================

[1/6] Setting up model and processor...
Loading model: HuggingFaceTB/SmolVLM-Instruct
Using 4-bit quantization (QLoRA)
Model loaded successfully

[2/6] Setting up LoRA adapters...
trainable params: 11,269,248 || all params: 2,257,542,128 || trainable%: 0.4992

[3/6] Loading datasets...
Loaded 8000 samples from results/pseudolabels.json
Dataset splits: train=6400, val=800, test=800

[4/6] Configuring trainer...
Trainer configured successfully

[5/6] Training model...
Training samples: 6400
Validation samples: 800
Epoch 1/3: 100%|████████████| 400/400 [12:34<00:00, 1.89s/it]
...

[6/6] Saving model...
Model saved to: outputs/smolvlm-aesthetic-scorer
```

## 🔮 Inference

### Single Image Inference

Generate scores for a single image:

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
  Reasoning: The image demonstrates strong visual impact with excellent composition...
```

### Batch Inference

Process multiple images in a directory:

```bash
python scripts/inference_smolvlm.py \
  --image-dir path/to/images \
  --adapter outputs/smolvlm-aesthetic-scorer \
  --output results.json
```

### Inference from File List

Process images from a text file:

```bash
# Create image list
echo "image1.jpg" > images.txt
echo "image2.jpg" >> images.txt

# Run inference
python scripts/inference_smolvlm.py \
  --image-list images.txt \
  --adapter outputs/smolvlm-aesthetic-scorer \
  --output results.json
```

### Using Custom Base Model

If you've pushed your adapter to Hugging Face:

```bash
python scripts/inference_smolvlm.py \
  --image path/to/image.jpg \
  --base-model HuggingFaceTB/SmolVLM-Instruct \
  --adapter your-username/smolvlm-aesthetic-scorer \
  --pretty-print
```

## 🔬 Advanced Usage

### Evaluation Only Mode

Evaluate a trained model without training:

```bash
python scripts/train_smolvlm.py \
  --eval-only \
  --adapter-path outputs/smolvlm-aesthetic-scorer
```

### Custom Prompts

Modify the system message in your config to customize behavior:

```yaml
prompt:
  system_message: |
    You are an expert photography critic with 20 years of experience.
    Focus on technical excellence and artistic merit.
    Be strict in your evaluations.
```

### Adjusting LoRA Parameters

For more capacity (but more memory):

```bash
python scripts/train_smolvlm.py \
  --lora-r 16 \
  --lora-alpha 16
```

### Training Without Quantization

If you have enough GPU memory:

```bash
python scripts/train_smolvlm.py --no-quantization
```

Note: This requires ~16GB+ VRAM.

### Resume Training

Training automatically saves checkpoints. To resume:

```bash
python scripts/train_smolvlm.py \
  --output-dir outputs/smolvlm-aesthetic-scorer
```

The trainer will automatically detect and resume from the latest checkpoint.

## 🐛 Troubleshooting

### Out of Memory Errors

**Problem:** CUDA out of memory during training

**Solutions:**

1. **Reduce batch size:**
   ```bash
   python scripts/train_smolvlm.py --batch-size 1
   ```

2. **Increase gradient accumulation:**
   ```bash
   python scripts/train_smolvlm.py --gradient-accumulation-steps 16
   ```

3. **Use smaller LoRA rank:**
   ```bash
   python scripts/train_smolvlm.py --lora-r 4
   ```

4. **Disable gradient checkpointing:**
   Edit config and set `gradient_checkpointing: false`

### Flash Attention Installation Issues

**Problem:** Flash Attention fails to install

**Solutions:**

1. **Install without Flash Attention:**
   Edit config and change `attn_implementation: "flash_attention_2"` to `attn_implementation: "eager"`

2. **Use pre-built wheels:**
   ```bash
   pip install flash-attn --no-build-isolation
   ```

### JSON Parsing Errors During Inference

**Problem:** Model output cannot be parsed as JSON

**Solutions:**

1. **Check training data quality:** Ensure your training data has valid JSON in the assistant responses

2. **Increase training epochs:** Model may need more training to learn the format

3. **Adjust generation parameters:** Lower temperature for more deterministic output

### Slow Training

**Problem:** Training is slower than expected

**Solutions:**

1. **Enable Flash Attention:**
   ```bash
   pip install flash-attn --no-build-isolation
   ```

2. **Use fused optimizer:**
   Already enabled by default (`adamw_torch_fused`)

3. **Increase batch size:**
   If you have GPU memory available

4. **Use mixed precision:**
   Already enabled by default (`bf16: true`)

### Model Not Learning

**Problem:** Loss not decreasing or poor validation performance

**Solutions:**

1. **Check learning rate:**
   ```bash
   python scripts/train_smolvlm.py --learning-rate 5e-5
   ```

2. **Increase training epochs:**
   ```bash
   python scripts/train_smolvlm.py --epochs 5
   ```

3. **Verify data quality:** Check that pseudolabels are reasonable and diverse

4. **Increase LoRA rank:**
   ```bash
   python scripts/train_smolvlm.py --lora-r 16
   ```

## 📈 Performance Tips

### Memory Optimization

- Use 4-bit quantization (QLoRA) - enabled by default
- Reduce batch size and increase gradient accumulation
- Enable gradient checkpointing
- Use Flash Attention 2

### Speed Optimization

- Use Flash Attention 2
- Use fused optimizer (`adamw_torch_fused`)
- Enable mixed precision (bf16)
- Increase batch size if memory allows
- Use multiple dataloader workers

### Quality Optimization

- Train for more epochs (3-5 recommended)
- Use higher LoRA rank (8-16)
- Ensure diverse training data
- Use validation set for early stopping
- Experiment with learning rate

## 🎯 Best Practices

1. **Start Small**: Test with `--max-samples 100` first
2. **Monitor Training**: Use W&B or TensorBoard
3. **Validate Early**: Check validation loss after first epoch
4. **Save Checkpoints**: Keep multiple checkpoints
5. **Test Inference**: Try inference on test images during training
6. **Document Experiments**: Track hyperparameters and results

## 📚 Additional Resources

- [SmolVLM Model Card](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

## 🤝 Contributing

Found a bug or have a suggestion? Please open an issue or submit a pull request!

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

