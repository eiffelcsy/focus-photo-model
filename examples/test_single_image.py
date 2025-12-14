#!/usr/bin/env python3
"""
Simple example script to test pseudolabel generation on a single image.

Usage:
    python examples/test_single_image.py --image path/to/image.jpg --ava-score 7.5
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.pseudolabel_generator import PseudolabelGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Test pseudolabel generation on a single image"
    )
    
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to test image"
    )
    
    parser.add_argument(
        "--ava-score",
        type=float,
        default=7.0,
        help="AVA aesthetic score for the image (default: 7.0)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/training/pseudolabel_config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load model in 8-bit quantization"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Testing Pseudolabel Generation with Gemma 3")
    print("=" * 70)
    print(f"Image: {args.image}")
    print(f"AVA Score: {args.ava_score}")
    print()
    
    # Load configuration
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config for single image test
    if args.load_in_8bit:
        config['model']['load_in_8bit'] = True
        config['model']['load_in_4bit'] = False
    
    # Reduce verbosity for cleaner output
    config['logging']['verbose'] = False
    
    print("Initializing model (this may take a minute)...")
    generator = PseudolabelGenerator(config_dict=config)
    
    print("Generating pseudolabel...")
    result = generator.generate_pseudolabel(args.image, args.ava_score)
    
    if result is None:
        print("\n✗ Failed to generate pseudolabel")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("Results:")
    print("=" * 70)
    
    # Print scores
    print(f"\nAesthetic Scores (AVA: {args.ava_score:.1f}/10):")
    print(f"  Impact:        {result['impact']:.1f}/10")
    print(f"  Style:         {result['style']:.1f}/10")
    print(f"  Composition:   {result['composition']:.1f}/10")
    print(f"  Lighting:      {result['lighting']:.1f}/10")
    print(f"  Color Balance: {result['color_balance']:.1f}/10")
    
    # Calculate average
    avg_score = (
        result['impact'] + result['style'] + result['composition'] + 
        result['lighting'] + result['color_balance']
    ) / 5
    print(f"\n  Average:       {avg_score:.1f}/10")
    
    # Print reasoning
    if 'reasoning' in result:
        print(f"\nReasoning:")
        print(f"  {result['reasoning']}")
    
    # Print raw response (optional)
    print(f"\nRaw Model Response:")
    print(f"  {result['raw_response'][:200]}...")
    
    # Save to file
    output_file = Path(args.image).stem + "_pseudolabel.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    print()


if __name__ == "__main__":
    main()

