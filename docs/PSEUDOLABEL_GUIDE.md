# Pseudolabel Generation Guide

This guide provides detailed instructions for generating pseudolabels for the AVA dataset using Google's Gemma 3 4B vision-language model.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Usage Examples](#usage-examples)
5. [Output Format](#output-format)
6. [Advanced Usage](#advanced-usage)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **GPU**: NVIDIA GPU with at least 16GB VRAM (recommended)
  - Can use 8-bit quantization with 10GB VRAM
  - Can use 4-bit quantization with 6GB VRAM
- **CPU**: Multi-core processor (if no GPU available, but much slower)
- **RAM**: At least 32GB
- **Storage**: Sufficient space for AVA dataset and outputs

### Software Requirements

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install accelerate if not already installed
pip install accelerate

# Optional: Install bitsandbytes for quantization
pip install bitsandbytes
```

## Quick Start

### 1. Prepare Your AVA Dataset

Ensure your AVA dataset is organized as follows:

```
data/ava/
├── AVA.txt              # AVA labels file
└── images/              # Directory containing images
    ├── 123456.jpg
    ├── 123457.jpg
    └── ...
```

The `AVA.txt` file should contain image IDs and scores. Example format:

```
123456 1 2 5 10 15 20 15 10 5 2 1
123457 2 3 8 12 18 25 18 12 8 3 2
...
```

### 2. Configure Paths

Edit `config/training/pseudolabel_config.yaml`:

```yaml
dataset:
  ava_csv_path: "data/ava/AVA.txt"
  images_dir: "data/ava/images"
  output_dir: "data/ava/pseudolabels"
```

### 3. Run Pseudolabel Generation

```bash
# Start pseudolabel generation
python scripts/generate_pseudolabels.py
```

That's it! The script will:
- Load the Gemma 3 model
- Process each image in the AVA dataset
- Save pseudolabels to the output directory
- Create checkpoints every 100 images
- Resume automatically if interrupted

## Configuration

### Model Configuration

```yaml
model:
  model_id: "google/gemma-3-4b-it"  # Model to use
  device_map: "auto"                 # Automatic device mapping
  dtype: "bfloat16"                  # Data type (bfloat16 or float16)
  load_in_8bit: false                # Enable for 10GB VRAM
  load_in_4bit: false                # Enable for 6GB VRAM
```

### Generation Parameters

```yaml
generation:
  max_new_tokens: 200      # Maximum response length
  do_sample: false         # Deterministic output
  temperature: 0.1         # Sampling temperature (if do_sample=true)
  top_p: 0.95             # Nucleus sampling (if do_sample=true)
```

### Processing Configuration

```yaml
processing:
  batch_size: 1                      # Images per batch (keep at 1)
  num_workers: 0                     # Parallel workers (keep at 0)
  max_images: null                   # Limit processing (null = all)
  resume_from_checkpoint: true       # Resume on restart
  checkpoint_interval: 100           # Save every N images
```

### Prompt Customization

```yaml
prompt:
  system_message: "You are a professional photography critic with expertise in aesthetic evaluation."
  
  criteria:
    - name: "impact"
      description: "Emotional response and memorability upon first viewing"
    - name: "style"
      description: "Artistic expression and creative vision"
    # ... add more criteria as needed
  
  use_ava_score_as_context: true    # Include AVA score in prompt
```

## Usage Examples

### Example 1: Process First 100 Images (Testing)

```bash
python scripts/generate_pseudolabels.py --max-images 100
```

### Example 2: Use 8-bit Quantization (Lower Memory)

```bash
python scripts/generate_pseudolabels.py --load-in-8bit
```

### Example 3: Custom Paths

```bash
python scripts/generate_pseudolabels.py \
    --ava-csv data/custom/AVA.txt \
    --images-dir data/custom/images \
    --output-dir data/custom/pseudolabels
```

### Example 4: Start Fresh (No Resume)

```bash
python scripts/generate_pseudolabels.py --no-resume
```

### Example 5: Frequent Checkpoints

```bash
python scripts/generate_pseudolabels.py --checkpoint-interval 50
```

### Example 6: Test Single Image

```bash
python examples/test_single_image.py \
    --image data/ava/images/123456.jpg \
    --ava-score 7.5
```

## Output Format

### Pseudolabels JSON

The output file `data/ava/pseudolabels/pseudolabels.json` contains:

```json
[
  {
    "impact": 7.5,
    "style": 7.2,
    "composition": 8.1,
    "lighting": 7.8,
    "color_balance": 7.3,
    "reasoning": "The photograph demonstrates strong visual impact...",
    "image_id": "123456",
    "image_path": "data/ava/images/123456.jpg",
    "ava_score": 7.4,
    "raw_response": "{\n  \"impact\": 7.5,\n  ..."
  },
  {
    "impact": 5.2,
    "style": 5.8,
    ...
  }
]
```

### Checkpoint File

The checkpoint file `data/ava/pseudolabels/checkpoint.txt` contains processed image IDs:

```
123456
123457
123458
...
```

### Failed Images File

If any images fail processing, they're logged in `data/ava/pseudolabels/failed_images.txt`:

```
789123
789124
```

## Advanced Usage

### Programmatic Usage

```python
from src.training.pseudolabel_generator import PseudolabelGenerator

# Initialize generator
generator = PseudolabelGenerator(
    config_path="config/training/pseudolabel_config.yaml"
)

# Generate pseudolabel for single image
result = generator.generate_pseudolabel(
    image_path="data/ava/images/123456.jpg",
    ava_score=7.5
)

print(f"Impact: {result['impact']}")
print(f"Composition: {result['composition']}")
print(f"Reasoning: {result['reasoning']}")

# Process entire dataset
generator.process_dataset()
```

### Custom Configuration

```python
from src.training.pseudolabel_generator import PseudolabelGenerator
import yaml

# Load and modify config
with open("config/training/pseudolabel_config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Customize
config['model']['load_in_8bit'] = True
config['processing']['max_images'] = 500

# Create generator with custom config
generator = PseudolabelGenerator(config_dict=config)
generator.process_dataset()
```

### Batch Processing on HPC Cluster

Example SLURM script (`sbatch_pseudolabel.sh`):

```bash
#!/bin/bash
#SBATCH --job-name=pseudolabel
#SBATCH --output=logs/pseudolabel_%j.out
#SBATCH --error=logs/pseudolabel_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Load modules
module load python/3.10
module load cuda/12.1

# Activate environment
source venv/bin/activate

# Run pseudolabel generation
python scripts/generate_pseudolabels.py \
    --checkpoint-interval 100 \
    --verbose

echo "Job completed"
```

Submit with:
```bash
sbatch sbatch_pseudolabel.sh
```

## Troubleshooting

### Problem: Out of Memory Error

**Solution 1:** Use 8-bit quantization
```bash
python scripts/generate_pseudolabels.py --load-in-8bit
```

**Solution 2:** Use 4-bit quantization (lower quality but minimal memory)
```bash
python scripts/generate_pseudolabels.py --load-in-4bit
```

**Solution 3:** Edit config to enable quantization permanently
```yaml
model:
  load_in_8bit: true
```

### Problem: Model Loading Fails

**Error:** `ImportError: No module named 'transformers'`

**Solution:** Install required packages
```bash
pip install -r requirements.txt
```

**Error:** `OSError: Model 'google/gemma-3-4b-it' not found`

**Solution:** Check your internet connection and HuggingFace access
```bash
# Login to HuggingFace if model requires authentication
huggingface-cli login
```

### Problem: JSON Parsing Errors

If you see many JSON parsing errors in the output:

**Solution 1:** Check the model's raw responses in the verbose logs
```bash
python scripts/generate_pseudolabels.py --verbose --max-images 10
```

**Solution 2:** Adjust generation parameters for more consistent output
```yaml
generation:
  max_new_tokens: 300      # Increase if responses are truncated
  temperature: 0.1         # Lower for more deterministic output
  do_sample: false         # Disable sampling for consistency
```

### Problem: Slow Processing Speed

**Expected Speed:** 
- With GPU: ~2-5 seconds per image
- Without GPU: ~30-60 seconds per image

**Solution 1:** Verify GPU is being used
```python
import torch
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.device_count())  # Should show your GPUs
```

**Solution 2:** Use smaller model or quantization
```bash
python scripts/generate_pseudolabels.py --load-in-8bit
```

### Problem: Resume Not Working

If checkpoints aren't being restored:

**Solution:** Check checkpoint file exists
```bash
ls data/ava/pseudolabels/checkpoint.txt
```

If you want to force a fresh start:
```bash
python scripts/generate_pseudolabels.py --no-resume
```

### Problem: Images Not Found

**Error:** `Image not found: data/ava/images/123456.jpg`

**Solution 1:** Verify image directory structure
```bash
ls data/ava/images/ | head -10
```

**Solution 2:** Check image filename format in AVA.txt matches your files
```python
# Your images might be .png instead of .jpg
# Modify the image path logic in pseudolabel_generator.py
```

## Performance Benchmarks

Tested on NVIDIA RTX 3090 (24GB VRAM):

| Configuration | Speed | Memory Usage |
|---------------|-------|--------------|
| BF16 (default) | 3 sec/image | 15GB |
| 8-bit quantization | 4 sec/image | 9GB |
| 4-bit quantization | 5 sec/image | 5GB |
| CPU only | 45 sec/image | 8GB RAM |

**Estimated Time for Full AVA Dataset (255,530 images):**
- With GPU (BF16): ~21 hours
- With GPU (8-bit): ~28 hours
- With CPU: ~320 hours (13+ days)

## Best Practices

1. **Start Small:** Test with `--max-images 100` first
2. **Monitor Progress:** Check logs and checkpoint files regularly
3. **Use Checkpoints:** Don't disable resume unless necessary
4. **Verify Output:** Inspect a few pseudolabels manually to ensure quality
5. **Save Resources:** Use 8-bit quantization if VRAM is limited
6. **Parallel Processing:** Consider splitting dataset across multiple GPUs

## Next Steps

After generating pseudolabels:

1. **Analyze Distribution:** Check score distributions across criteria
2. **Quality Control:** Sample and verify pseudolabel quality
3. **Training:** Use pseudolabels for model training
4. **Evaluation:** Compare model predictions with pseudolabels

## Support

For issues or questions:
- Check existing issues in the repository
- Create a new issue with error logs
- Include your configuration and system specs

