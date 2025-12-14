#!/usr/bin/env python3
"""
Script to generate pseudolabels for AVA dataset using Gemma 3 VLM.

Usage:
    python scripts/generate_pseudolabels.py --config config/training/pseudolabel_config.yaml
    
    # Or with custom settings:
    python scripts/generate_pseudolabels.py --config config/training/pseudolabel_config.yaml --max-images 1000
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.pseudolabel_generator import PseudolabelGenerator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate pseudolabels for AVA dataset using Gemma 3 VLM"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/training/pseudolabel_config.yaml",
        help="Path to configuration file (default: config/training/pseudolabel_config.yaml)"
    )
    
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to process (default: process all)"
    )
    
    parser.add_argument(
        "--ava-csv",
        type=str,
        default=None,
        help="Path to AVA CSV file (overrides config)"
    )
    
    parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Path to images directory (overrides config)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for pseudolabels (overrides config)"
    )
    
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from checkpoint, start fresh"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for processing (overrides config)"
    )
    
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help="Save checkpoint every N images (overrides config)"
    )
    
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Model ID to use (overrides config, e.g., google/gemma-3-4b-it)"
    )
    
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load model in 8-bit quantization for memory efficiency"
    )
    
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit quantization for memory efficiency"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    print("=" * 70)
    print("AVA Dataset Pseudolabel Generation with Gemma 3")
    print("=" * 70)
    print(f"Config file: {args.config}")
    print()
    
    # Load configuration
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.max_images is not None:
        config['processing']['max_images'] = args.max_images
        print(f"Limiting to {args.max_images} images")
    
    if args.ava_csv is not None:
        config['dataset']['ava_csv_path'] = args.ava_csv
        print(f"Using AVA CSV: {args.ava_csv}")
    
    if args.images_dir is not None:
        config['dataset']['images_dir'] = args.images_dir
        print(f"Using images directory: {args.images_dir}")
    
    if args.output_dir is not None:
        config['dataset']['output_dir'] = args.output_dir
        print(f"Output directory: {args.output_dir}")
    
    if args.no_resume:
        config['processing']['resume_from_checkpoint'] = False
        print("Starting fresh (not resuming from checkpoint)")
    
    if args.batch_size is not None:
        config['processing']['batch_size'] = args.batch_size
    
    if args.checkpoint_interval is not None:
        config['processing']['checkpoint_interval'] = args.checkpoint_interval
        print(f"Checkpoint interval: {args.checkpoint_interval}")
    
    if args.model_id is not None:
        config['model']['model_id'] = args.model_id
        print(f"Using model: {args.model_id}")
    
    if args.load_in_8bit:
        config['model']['load_in_8bit'] = True
        config['model']['load_in_4bit'] = False
        print("Loading model in 8-bit quantization")
    
    if args.load_in_4bit:
        config['model']['load_in_4bit'] = True
        config['model']['load_in_8bit'] = False
        print("Loading model in 4-bit quantization")
    
    if args.verbose:
        config['logging']['verbose'] = True
    
    print()
    print("Initializing pseudolabel generator...")
    print()
    
    # Create generator and process dataset
    try:
        generator = PseudolabelGenerator(config_dict=config)
        generator.process_dataset()
        
        print()
        print("✓ Pseudolabel generation completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress has been saved.")
        print("Run the script again to resume from checkpoint.")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n\n✗ Error during pseudolabel generation: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

