#!/usr/bin/env python3
"""
Inference script for SmolVLM aesthetic score prediction.

This script loads a fine-tuned SmolVLM model and generates aesthetic scores
for input images.

Usage:
    # Single image inference
    python scripts/inference_smolvlm.py --image path/to/image.jpg --adapter path/to/adapter
    
    # Batch inference on directory
    python scripts/inference_smolvlm.py --image-dir path/to/images --adapter path/to/adapter
    
    # Save results to JSON
    python scripts/inference_smolvlm.py --image-dir path/to/images --adapter path/to/adapter --output results.json
    
    # Use custom config
    python scripts/inference_smolvlm.py --image path/to/image.jpg --config config/training/smolvlm_config.yaml --adapter path/to/adapter
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.smolvlm_trainer import load_trained_model
import yaml


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def generate_scores_from_image(
    model,
    processor,
    image_path: str,
    system_message: str,
    user_message: str,
    max_new_tokens: int = 512,
    device: str = "cuda"
) -> Dict:
    """
    Generate aesthetic scores for a single image.
    
    Args:
        model: The SmolVLM model
        processor: The processor
        image_path: Path to the image
        system_message: System prompt
        user_message: User query
        max_new_tokens: Maximum tokens to generate
        device: Device to use
    
    Returns:
        Dictionary with scores and reasoning
    """
    # Load image
    try:
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None
    
    # Prepare messages
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_message
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": user_message,
                }
            ],
        },
    ]
    
    # Apply chat template
    text_input = processor.apply_chat_template(
        messages,
        add_generation_prompt=True
    )
    
    # Prepare inputs
    model_inputs = processor(
        text=text_input,
        images=[[image]],
        return_tensors="pt",
    ).to(device)
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.1,
        )
    
    # Decode output
    trimmed_generated_ids = [
        out_ids[len(in_ids):] 
        for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        trimmed_generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    # Parse JSON output
    try:
        # Try to extract JSON from the output
        if "```json" in output_text:
            json_str = output_text.split("```json")[1].split("```")[0].strip()
        elif "{" in output_text and "}" in output_text:
            # Find the first complete JSON object
            start = output_text.find("{")
            end = output_text.rfind("}") + 1
            json_str = output_text[start:end]
        else:
            json_str = output_text
        
        result = json.loads(json_str)
        result['raw_response'] = output_text
        result['image_path'] = str(image_path)
        
        return result
    
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from model output: {e}")
        print(f"Raw output: {output_text}")
        return {
            'error': 'Failed to parse JSON',
            'raw_response': output_text,
            'image_path': str(image_path)
        }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with fine-tuned SmolVLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Input
    parser.add_argument(
        '--image',
        type=str,
        help='Path to single image for inference'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        help='Directory containing images for batch inference'
    )
    parser.add_argument(
        '--image-list',
        type=str,
        help='Text file with list of image paths (one per line)'
    )
    
    # Model
    parser.add_argument(
        '--adapter',
        type=str,
        required=True,
        help='Path to trained adapter'
    )
    parser.add_argument(
        '--base-model',
        type=str,
        default='HuggingFaceTB/SmolVLM-Instruct',
        help='Base model ID'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/training/smolvlm_config.yaml',
        help='Path to configuration file'
    )
    
    # Generation
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=512,
        help='Maximum tokens to generate'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda/cpu)'
    )
    
    # Output
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save results JSON file'
    )
    parser.add_argument(
        '--pretty-print',
        action='store_true',
        help='Pretty print results to console'
    )
    
    # Image extensions for batch processing
    parser.add_argument(
        '--extensions',
        type=str,
        nargs='+',
        default=['.jpg', '.jpeg', '.png', '.bmp', '.webp'],
        help='Image file extensions to process'
    )
    
    return parser.parse_args()


def get_image_paths(args) -> List[Path]:
    """
    Get list of image paths from command-line arguments.
    
    Args:
        args: Parsed arguments
    
    Returns:
        List of image paths
    """
    image_paths = []
    
    if args.image:
        # Single image
        image_paths.append(Path(args.image))
    
    elif args.image_dir:
        # Directory of images
        image_dir = Path(args.image_dir)
        for ext in args.extensions:
            image_paths.extend(image_dir.glob(f"*{ext}"))
            image_paths.extend(image_dir.glob(f"*{ext.upper()}"))
        image_paths = sorted(image_paths)
    
    elif args.image_list:
        # List of image paths from file
        with open(args.image_list, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    image_paths.append(Path(line))
    
    else:
        raise ValueError("Must provide --image, --image-dir, or --image-list")
    
    return image_paths


def main():
    """Main inference script."""
    args = parse_args()
    
    print("=" * 80)
    print("SmolVLM Aesthetic Score Inference")
    print("=" * 80)
    
    # Load config
    print(f"\nLoading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Get image paths
    image_paths = get_image_paths(args)
    print(f"\nFound {len(image_paths)} images to process")
    
    if len(image_paths) == 0:
        print("No images found. Exiting.")
        sys.exit(1)
    
    # Load model
    print(f"\nLoading model...")
    print(f"  Base model: {args.base_model}")
    print(f"  Adapter: {args.adapter}")
    
    model, processor = load_trained_model(
        base_model_id=args.base_model,
        adapter_path=args.adapter,
        device_map="auto",
        torch_dtype="bfloat16",
    )
    
    print("Model loaded successfully!")
    
    # Get prompts from config
    system_message = config['prompt']['system_message']
    user_message = config['prompt']['user_message']
    
    # Run inference
    print(f"\nRunning inference on {len(image_paths)} images...")
    results = []
    
    for image_path in tqdm(image_paths, desc="Processing images"):
        result = generate_scores_from_image(
            model=model,
            processor=processor,
            image_path=str(image_path),
            system_message=system_message,
            user_message=user_message,
            max_new_tokens=args.max_tokens,
            device=args.device,
        )
        
        if result:
            results.append(result)
            
            if args.pretty_print:
                print(f"\n{'-' * 80}")
                print(f"Image: {image_path}")
                if 'error' not in result:
                    print(f"  Impact: {result.get('impact', 'N/A')}")
                    print(f"  Style: {result.get('style', 'N/A')}")
                    print(f"  Composition: {result.get('composition', 'N/A')}")
                    print(f"  Lighting: {result.get('lighting', 'N/A')}")
                    print(f"  Color Balance: {result.get('color_balance', 'N/A')}")
                    print(f"  Reasoning: {result.get('reasoning', 'N/A')}")
                else:
                    print(f"  Error: {result['error']}")
    
    # Save results
    if args.output:
        print(f"\nSaving results to: {args.output}")
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print("Results saved!")
    
    # Print summary
    print("\n" + "=" * 80)
    print("Inference Summary")
    print("=" * 80)
    print(f"Total images processed: {len(results)}")
    
    if results and 'error' not in results[0]:
        # Calculate average scores
        avg_impact = sum(r.get('impact', 0) for r in results if 'error' not in r) / len(results)
        avg_style = sum(r.get('style', 0) for r in results if 'error' not in r) / len(results)
        avg_composition = sum(r.get('composition', 0) for r in results if 'error' not in r) / len(results)
        avg_lighting = sum(r.get('lighting', 0) for r in results if 'error' not in r) / len(results)
        avg_color_balance = sum(r.get('color_balance', 0) for r in results if 'error' not in r) / len(results)
        
        print(f"\nAverage Scores:")
        print(f"  Impact: {avg_impact:.2f}")
        print(f"  Style: {avg_style:.2f}")
        print(f"  Composition: {avg_composition:.2f}")
        print(f"  Lighting: {avg_lighting:.2f}")
        print(f"  Color Balance: {avg_color_balance:.2f}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()

