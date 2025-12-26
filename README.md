# Focus Photo Model - Aesthetic Quality Assessment

A deep learning model for evaluating photograph aesthetics using the AVA (Aesthetic Visual Analysis) dataset with pseudolabeling via Gemma 3 Vision-Language Model.

## Features

- **Pseudolabel Generation**: Generate detailed aesthetic attribute scores using Google's Gemma 3 4B vision-language model
- **SmolVLM Fine-tuning**: Fine-tune SmolVLM (2B Vision Language Model) on your pseudolabels using TRL and QLoRA
- **Stratified Sampling**: Ensure representative coverage across all quality levels when working with limited budgets
- **Multi-Criteria Evaluation**: Assess images across 5 key dimensions:
  - Impact (emotional response and memorability)
  - Style (artistic expression and creative vision)
  - Composition (visual arrangement and balance)
  - Lighting (quality and effectiveness)
  - Color Balance (harmony of colors)
- **Flexible Training**: Support for supervised, self-supervised, and hybrid training approaches
- **Efficient Training**: QLoRA with 4-bit quantization for consumer GPU training
- **Production Ready**: Complete inference pipeline for deployed models
- **Checkpoint Support**: Resume pseudolabel generation from interruptions

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd focus-photo-model

# Install dependencies
pip install -r requirements.txt

# For pseudolabel generation, you may need accelerate
pip install accelerate
```

## Quick Start

### 1. Generate Pseudolabels for Your Dataset

First, configure your dataset paths in `config/training/pseudolabel_config.yaml`:

```yaml
dataset:
  ava_csv_path: "data/ava/AVA.txt"
  images_dir: "data/ava/images"
  output_dir: "data/ava/pseudolabels"
```

Then run the pseudolabel generation:

```bash
# Process entire dataset
python scripts/generate_pseudolabels.py

# Process first 1000 images with stratified sampling (ensures representative coverage)
python scripts/generate_pseudolabels.py --max-images 1000

# Use sequential sampling instead of stratified
python scripts/generate_pseudolabels.py --max-images 1000 --no-stratified

# Use 8-bit quantization for lower memory usage
python scripts/generate_pseudolabels.py --load-in-8bit

# Resume from checkpoint
python scripts/generate_pseudolabels.py  # automatically resumes if checkpoint exists
```

### 2. Using the Pseudolabel Generator Programmatically

```python
from src.training.pseudolabel_generator import PseudolabelGenerator

# Initialize generator
generator = PseudolabelGenerator(config_path="config/training/pseudolabel_config.yaml")

# Generate pseudolabel for a single image
result = generator.generate_pseudolabel(
    image_path="path/to/image.jpg",
    ava_score=6.5
)

print(result)
# Output:
# {
#     'impact': 6.8,
#     'style': 6.2,
#     'composition': 7.1,
#     'lighting': 6.5,
#     'color_balance': 6.4,
#     'reasoning': 'The image shows good composition with balanced elements...',
#     'image_path': 'path/to/image.jpg',
#     'ava_score': 6.5
# }

# Process entire dataset
generator.process_dataset()
```

### 3. Fine-tune SmolVLM on Your Pseudolabels

Once you have generated pseudolabels, you can fine-tune SmolVLM to predict aesthetic scores:

```bash
# Basic training with default config
python scripts/train_smolvlm.py

# Custom training with specific parameters
python scripts/train_smolvlm.py \
  --pseudolabels-path results/pseudolabels.json \
  --images-dir src/data/images \
  --output-dir outputs/my-aesthetic-model \
  --epochs 3 \
  --batch-size 2

# Training with Weights & Biases logging
python scripts/train_smolvlm.py \
  --use-wandb \
  --wandb-project "aesthetic-scorer"
```

### 4. Run Inference with Your Fine-tuned Model

Use your trained model to score new images:

```bash
# Single image
python scripts/inference_smolvlm.py \
  --image path/to/image.jpg \
  --adapter outputs/smolvlm-aesthetic-scorer \
  --pretty-print

# Batch processing
python scripts/inference_smolvlm.py \
  --image-dir path/to/images \
  --adapter outputs/smolvlm-aesthetic-scorer \
  --output results.json
```

**Programmatic Usage:**

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
    image_path="path/to/image.jpg",
    system_message="You are an expert photography critic...",
    user_message="Analyze this photograph and provide aesthetic scores."
)

print(f"Impact: {result['impact']}")
print(f"Style: {result['style']}")
print(f"Composition: {result['composition']}")
print(f"Lighting: {result['lighting']}")
print(f"Color Balance: {result['color_balance']}")
```

## Project Structure

```
focus-photo-model/
├── config/
│   ├── data/
│   │   └── dataset_config.yaml       # Dataset configuration
│   └── training/
│       ├── pseudolabel_config.yaml   # Pseudolabel generation config
│       ├── smolvlm_config.yaml       # SmolVLM fine-tuning config
│       └── train_config.yaml         # Training configuration
├── scripts/
│   ├── generate_pseudolabels.py      # Generate pseudolabels script
│   ├── train_smolvlm.py              # SmolVLM fine-tuning script
│   ├── inference_smolvlm.py          # SmolVLM inference script
│   ├── train_supervised.py           # Supervised training
│   ├── train_self_supervised.py      # Self-supervised training
│   └── train_hybrid.py               # Hybrid training
├── src/
│   ├── training/
│   │   ├── pseudolabel_generator.py  # Pseudolabel generation module
│   │   ├── smolvlm_trainer.py        # SmolVLM trainer with QLoRA
│   │   ├── smolvlm_dataset.py        # Dataset loader for SmolVLM
│   │   ├── supervised_trainer.py     # Supervised trainer
│   │   ├── self_supervised_trainer.py # Self-supervised trainer
│   │   └── hybrid_trainer.py         # Hybrid trainer
│   ├── eval/
│   │   ├── evaluator.py              # Model evaluation
│   │   └── metrics.py                # Evaluation metrics
│   └── utils/
│       └── logging.py                # Logging utilities
├── docs/
│   └── SMOLVLM_FINETUNING.md         # Complete SmolVLM guide
├── examples/
│   ├── smolvlm_quickstart.py         # Quick start example
│   └── smolvlm_inference_example.py  # Inference example
└── requirements.txt                  # Python dependencies
```

## SmolVLM Fine-tuning

For detailed information about fine-tuning SmolVLM on your pseudolabels, see the **[Complete SmolVLM Fine-tuning Guide](docs/SMOLVLM_FINETUNING.md)**.

### Quick Overview

SmolVLM is a 2B parameter Vision Language Model that can be efficiently fine-tuned on consumer GPUs using QLoRA:

- **Memory Efficient**: Trains on 12GB+ VRAM GPUs (RTX 3060, 4060 Ti, L4)
- **Fast Training**: Uses Flash Attention 2 and 4-bit quantization
- **Production Ready**: Complete inference pipeline included
- **Easy to Use**: Simple configuration via YAML

**Training Example:**

```bash
python scripts/train_smolvlm.py \
  --pseudolabels-path results/pseudolabels.json \
  --images-dir src/data/images \
  --epochs 3 \
  --use-wandb
```

**Inference Example:**

```bash
python scripts/inference_smolvlm.py \
  --image photo.jpg \
  --adapter outputs/smolvlm-aesthetic-scorer \
  --pretty-print
```

See the [full documentation](docs/SMOLVLM_FINETUNING.md) for:
- Installation and setup
- Configuration options
- Training best practices
- Troubleshooting guide
- Advanced usage examples

## Configuration

### Pseudolabel Generation Configuration

Edit `config/training/pseudolabel_config.yaml` to customize:

- **Model Settings**: Model ID, quantization, device mapping
- **Generation Parameters**: Temperature, top-p, max tokens
- **Dataset Paths**: AVA dataset location and output directory
- **Processing Options**: Batch size, checkpoint intervals
- **Prompt Configuration**: System message, criteria, scoring

Key configuration options:

```yaml
model:
  model_id: "google/gemma-3-4b-it"
  load_in_8bit: false  # Set to true for memory-constrained environments

generation:
  max_new_tokens: 200
  do_sample: false
  temperature: 0.1

processing:
  checkpoint_interval: 100  # Save every 100 images
  resume_from_checkpoint: true
  max_images: null  # Set to limit processing
  
  # Stratified sampling (when max_images is set)
  stratified_sampling: true  # Ensures representative sampling across quality levels
  n_stratification_bins: 5  # Number of quality bins (default: 5)
  random_seed: 42  # For reproducibility
```

### Stratified Sampling

When you can't pseudolabel the entire dataset, stratified sampling ensures you get representative coverage across all quality levels. Instead of just taking the first N images (which could be biased), it:

1. **Divides scores into bins** based on percentiles (e.g., low, medium-low, medium, medium-high, high)
2. **Samples proportionally** from each bin to maintain the original distribution
3. **Preserves statistical properties** (mean, std, percentiles) of the full dataset

**Example:** If you have 250,000 images but can only pseudolabel 5,000:
- Without stratification: You get the first 5,000 images (potential bias if dataset is sorted)
- With stratification: You get ~1,000 images from each quality quintile (representative sample)

**Test the sampling strategy:**

```bash
# Visualize how stratified sampling compares to the full dataset
python scripts/test_stratified_sampling.py --n-samples 5000 --n-bins 5 --visualize
```

## Command-Line Options

The `generate_pseudolabels.py` script supports various options:

```bash
# Limit number of images with stratified sampling
python scripts/generate_pseudolabels.py --max-images 500

# Use sequential sampling instead (first N images)
python scripts/generate_pseudolabels.py --max-images 500 --no-stratified

# Custom stratification bins
python scripts/generate_pseudolabels.py --max-images 500 --n-bins 10

# Custom random seed for reproducibility
python scripts/generate_pseudolabels.py --max-images 500 --random-seed 123

# Custom paths
python scripts/generate_pseudolabels.py \
    --ava-csv data/custom/AVA.txt \
    --images-dir data/custom/images \
    --output-dir data/custom/pseudolabels

# Memory optimization
python scripts/generate_pseudolabels.py --load-in-8bit  # or --load-in-4bit

# Don't resume from checkpoint
python scripts/generate_pseudolabels.py --no-resume

# Custom checkpoint interval
python scripts/generate_pseudolabels.py --checkpoint-interval 50

# Verbose logging
python scripts/generate_pseudolabels.py --verbose
```

## How It Works

### Pseudolabel Generation Process

1. **Model Loading**: Loads Google's Gemma 3 4B vision-language model
2. **Prompt Construction**: Creates a structured prompt with the 5 aesthetic criteria
3. **Image Processing**: Processes each image through the VLM
4. **JSON Parsing**: Extracts structured scores from model responses
5. **Checkpoint Saving**: Regularly saves progress for resumption

### Prompt Template

The system uses a carefully crafted prompt that:
- Lists the 5 aesthetic criteria with descriptions
- Includes the AVA score as context for calibration
- Requests structured JSON output
- Asks for reasoning to improve consistency

Example prompt:

```
Analyze this photograph and rate it on these 5 criteria (scale 1-10):

1. IMPACT: Emotional response and memorability upon first viewing
2. STYLE: Artistic expression and creative vision
3. COMPOSITION: Visual arrangement and balance of elements
4. LIGHTING: Quality and effectiveness of illumination
5. COLOR BALANCE: Harmony and effectiveness of color relationships

The overall aesthetic score for this image is 7.2/10.
Your individual scores should reflect this overall quality level.

Respond in JSON format:
{
    "impact": X.X,
    "style": X.X,
    "composition": X.X,
    "lighting": X.X,
    "color_balance": X.X,
    "reasoning": "brief explanation of scores"
}
```

## Output Format

Pseudolabels are saved as JSON:

```json
[
  {
    "impact": 7.5,
    "style": 7.2,
    "composition": 8.1,
    "lighting": 7.8,
    "color_balance": 7.3,
    "reasoning": "Strong visual impact with excellent composition...",
    "image_id": "123456",
    "image_path": "data/ava/images/123456.jpg",
    "ava_score": 7.4,
    "raw_response": "..."
  }
]
```

## Performance Tips

1. **Memory Management**: Use `--load-in-8bit` or `--load-in-4bit` for GPU memory constraints
2. **Checkpoint Intervals**: Adjust based on stability (lower for unstable environments)
3. **Batch Processing**: Currently single-image processing (VLM limitation)
4. **Resume Support**: Always enabled by default, safe to interrupt and restart

## Troubleshooting

### Out of Memory Errors

```bash
# Try 8-bit quantization
python scripts/generate_pseudolabels.py --load-in-8bit

# Or 4-bit for extreme cases
python scripts/generate_pseudolabels.py --load-in-4bit
```

### Failed Image Processing

Check `data/ava/pseudolabels/failed_images.txt` for images that failed processing.

### Resume from Checkpoint

The script automatically resumes from the last checkpoint. To start fresh:

```bash
python scripts/generate_pseudolabels.py --no-resume
```

## License

[Your License Here]

## Citation

If you use this code, please cite:

```bibtex
@misc{focus-photo-model,
  author = {Your Name},
  title = {Focus Photo Model: Aesthetic Quality Assessment},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/focus-photo-model}
}
```

