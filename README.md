# Focus Photo Model - Photography Quality Assessment

A minimal pipeline for generating pseudolabels using Gemma 3-4B to assess photography quality on 5 key criteria.

## Overview

This project uses a vision-language model (Gemma 3-4B) to generate pseudolabels for photography quality assessment. The model evaluates images on 5 criteria:

1. **Impact** - Emotional response and memorability upon first viewing
2. **Style** - Artistic expression and creative vision  
3. **Composition** - Visual arrangement and balance of elements
4. **Lighting** - Quality and effectiveness of illumination
5. **Color Balance** - Harmony and effectiveness of color relationships

## Installation

```bash
pip install -e .
```

## Usage

### Basic Usage

Generate pseudolabels for images in a directory:

```bash
python scripts/generate_pseudolabels.py --input /path/to/images --output results/
```

### With Configuration

Use a custom configuration file:

```bash
python scripts/generate_pseudolabels.py --config config/training/pseudolabel_config.yaml --input images/ --output results/
```

### Command Line Options

```bash
python scripts/generate_pseudolabels.py --help
```

Key options:
- `--input`: Image files or directories to process
- `--output`: Output directory for results
- `--config`: Configuration file path
- `--num-samples`: Limit number of images to process
- `--model-name`: Override model name
- `--device`: Specify device (auto, cuda, cpu, mps)

## Output Format

The generator produces JSON files with scores for each criterion:

```json
{
  "image_id": "example_image",
  "ava_score": 6.5,
  "scores": {
    "impact": 7.2,
    "style": 6.8,
    "composition": 7.0,
    "lighting": 6.3,
    "color_balance": 6.7,
    "reasoning": "Strong composition with good lighting..."
  },
  "composite_score": 6.8,
  "raw_response": "..."
}
```

## Configuration

Edit `config/training/pseudolabel_config.yaml` to customize:

- Model name and parameters
- Generation settings (temperature, max tokens)
- Processing options

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- PIL/Pillow
- See `requirements.txt` for full list

## Notes

- The AVA dataset should be prepared separately with image paths and scores
- Default AVA score of 5.0 is used if not provided
- Model responses are parsed as JSON for structured output
- Failed images are skipped with error logging
