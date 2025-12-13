#!/usr/bin/env python3
"""
Generate Pseudolabels Script

Command-line interface for generating pseudolabels using SmolVLM for photography quality assessment.
This script can process individual images, directories of images, or the AVA dataset.

Usage:
    python scripts/generate_pseudolabels.py --input /path/to/images --output /path/to/results
    python scripts/generate_pseudolabels.py --ava-split train --num-samples 1000 --output results/
    python scripts/generate_pseudolabels.py --config config/training/pseudolabel_config.yaml
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional
import yaml
import json
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training.pseudolabel_generator import GemmaPseudolabelGenerator, load_ava_dataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pseudolabel_generation.log')
    ]
)
logger = logging.getLogger(__name__)


def find_images_in_directory(directory: str, recursive: bool = True) -> List[str]:
    """
    Find all image files in a directory.
    
    Args:
        directory: Path to directory to search
        recursive: Whether to search subdirectories
        
    Returns:
        List of image file paths
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}
    image_paths = []
    
    directory = Path(directory)
    if not directory.exists():
        logger.error(f"Directory does not exist: {directory}")
        return []
    
    if recursive:
        pattern = "**/*"
    else:
        pattern = "*"
    
    for file_path in directory.glob(pattern):
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_paths.append(str(file_path))
    
    logger.info(f"Found {len(image_paths)} images in {directory}")
    return sorted(image_paths)


def validate_config(config: dict) -> dict:
    """
    Validate and set defaults for configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Validated configuration dictionary
    """
    defaults = {
        'model_name': 'HuggingFaceTB/SmolVLM-Instruct',
        'device': 'auto',
        'batch_size': 4,
        'max_new_tokens': 256,
        'temperature': 0.7,
        'cache_dir': None,
        'save_individual': True,
        'save_summary': True,
        'recursive_search': True
    }
    
    # Apply defaults for missing keys
    for key, default_value in defaults.items():
        if key not in config:
            config[key] = default_value
    
    # Validate specific values
    if config['batch_size'] < 1:
        config['batch_size'] = 1
        logger.warning("Batch size must be >= 1, setting to 1")
    
    if config['max_new_tokens'] < 50:
        config['max_new_tokens'] = 50
        logger.warning("max_new_tokens too small, setting to 50")
    
    if not 0.0 <= config['temperature'] <= 2.0:
        config['temperature'] = 0.7
        logger.warning("Temperature should be between 0.0 and 2.0, setting to 0.7")
    
    return config


def create_output_directory(output_path: str) -> Path:
    """
    Create output directory with timestamp if it doesn't exist.
    
    Args:
        output_path: Base output path
        
    Returns:
        Path object for the created directory
    """
    output_path = Path(output_path)
    
    # If output path doesn't exist, create it
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {output_path}")
    
    # Create timestamped subdirectory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_path / f"pseudolabels_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    
    logger.info(f"Results will be saved to: {run_dir}")
    return run_dir


def save_run_metadata(output_dir: Path, args: argparse.Namespace, config: dict, image_paths: List[str]):
    """
    Save metadata about the current run.
    
    Args:
        output_dir: Output directory
        args: Command line arguments
        config: Configuration used
        image_paths: List of processed image paths
    """
    metadata = {
        'run_info': {
            'timestamp': datetime.now().isoformat(),
            'command_line_args': vars(args),
            'total_images': len(image_paths),
            'config': config
        },
        'input_info': {
            'source_type': 'ava_dataset' if args.ava_dataset else 'local_files',
            'ava_csv': str(args.ava_csv) if args.ava_csv else None,
            'ava_images_dir': str(args.ava_images_dir) if args.ava_images_dir else None,
            'input_paths': args.input if args.input else [],
            'num_samples': args.num_samples,
            'recursive_search': config.get('recursive_search', True)
        },
        'sample_paths': image_paths[:100]  # Save first 100 paths as sample
    }
    
    metadata_file = output_dir / "run_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"Run metadata saved to: {metadata_file}")


def main():
    """Main function for pseudolabel generation script."""
    parser = argparse.ArgumentParser(
        description="Generate pseudolabels for photography quality assessment using SmolVLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Process a directory of images
            python scripts/generate_pseudolabels.py --input /path/to/images --output results/

            # Process AVA dataset with 1000 samples
            python scripts/generate_pseudolabels.py --ava-dataset --num-samples 1000 --output results/

            # Use custom configuration
            python scripts/generate_pseudolabels.py --config config/training/pseudolabel_config.yaml --input images/ --output results/

            # Process specific image files
            python scripts/generate_pseudolabels.py --input image1.jpg image2.jpg --output results/
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input', '-i',
        nargs='+',
        help='Input image files or directories to process'
    )
    input_group.add_argument(
        '--ava-dataset',
        action='store_true',
        help='Load AVA dataset from data folder (uses ground_truth_dataset.csv and images/)'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output directory for pseudolabel results'
    )
    
    # Configuration
    parser.add_argument(
        '--config', '-c',
        help='Path to YAML configuration file'
    )
    
    # Processing options
    parser.add_argument(
        '--num-samples', '-n',
        type=int,
        help='Maximum number of samples to process (useful for testing)'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=4,
        help='Batch size for processing (default: 4)'
    )
    
    parser.add_argument(
        '--device',
        choices=['auto', 'cuda', 'cpu', 'mps'],
        default='auto',
        help='Device to use for inference (default: auto)'
    )
    
    parser.add_argument(
        '--model-name',
        default='HuggingFaceTB/SmolVLM-Instruct',
        help='HuggingFace model name to use (default: HuggingFaceTB/SmolVLM-Instruct)'
    )
    
    # Output format options
    parser.add_argument(
        '--no-individual',
        action='store_true',
        help='Skip saving individual assessment files'
    )
    
    parser.add_argument(
        '--no-summary',
        action='store_true',
        help='Skip saving summary CSV file'
    )
    
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='Do not search subdirectories when processing directories'
    )
    
    # Advanced options
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature for text generation (default: 0.7)'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=256,
        help='Maximum tokens to generate per assessment (default: 256)'
    )
    
    parser.add_argument(
        '--cache-dir',
        help='Directory to cache model files'
    )
    
    parser.add_argument(
        '--ava-csv',
        help='Path to AVA ground truth CSV file (default: src/data/ground_truth_dataset.csv)'
    )
    
    parser.add_argument(
        '--ava-images-dir',
        help='Path to AVA images directory (default: src/data/images)'
    )
    
    parser.add_argument(
        '--skip-missing-images',
        action='store_true',
        help='Skip images that are not found (default: True when using AVA dataset)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting pseudolabel generation")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Load configuration
        config = {}
        # Default config path
        default_config_path = Path(__file__).parent.parent / "config" / "training" / "pseudolabel_config.yaml"
        
        # Use provided config or default config if it exists
        config_path = args.config if args.config else (default_config_path if default_config_path.exists() else None)
        
        if config_path and os.path.exists(config_path):
            logger.info(f"Loading configuration from: {config_path}")
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                if loaded_config:
                    config.update(loaded_config)
        
        # Override config with command line arguments (CLI args take precedence)
        config.update({
            'model_name': args.model_name,
            'device': args.device,
            'batch_size': args.batch_size,
            'max_new_tokens': args.max_tokens,
            'temperature': args.temperature,
            'cache_dir': args.cache_dir,
            'save_individual': not args.no_individual,
            'save_summary': not args.no_summary,
            'recursive_search': not args.no_recursive
        })
        
        # Validate configuration
        config = validate_config(config)
        logger.info(f"Using configuration: {config}")
        
        # Create pseudolabel generator
        logger.info("Initializing Gemma pseudolabel generator...")
        generator = GemmaPseudolabelGenerator(
            model_name=config['model_name'],
            device=config['device'],
            max_new_tokens=config['max_new_tokens'],
            temperature=config['temperature']
        )
        
        # Get image data (paths + AVA scores)
        image_data = []
        
        if args.ava_dataset:
            # Load AVA dataset
            project_root = Path(__file__).parent.parent
            csv_path = args.ava_csv or (project_root / "src" / "data" / "ground_truth_dataset.csv")
            images_dir = args.ava_images_dir or (project_root / "src" / "data" / "images")
            
            logger.info(f"Loading AVA dataset from CSV: {csv_path}")
            logger.info(f"Images directory: {images_dir}")
            
            try:
                image_data = load_ava_dataset(
                    csv_path=str(csv_path),
                    images_dir=str(images_dir),
                    max_samples=args.num_samples,
                    check_image_exists=not args.skip_missing_images
                )
            except Exception as e:
                logger.error(f"Failed to load AVA dataset: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                return 1
        else:
            # Process local files/directories
            if not args.input:
                logger.error("Either --input or --ava-dataset must be specified")
                return 1
                
            for input_path in args.input:
                input_path = Path(input_path)
                
                if input_path.is_file():
                    # Single file
                    if input_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}:
                        image_data.append({
                            'image_path': str(input_path),
                            'ava_score': 5.0  # Default score if not provided
                        })
                    else:
                        logger.warning(f"Skipping non-image file: {input_path}")
                
                elif input_path.is_dir():
                    # Directory
                    dir_images = find_images_in_directory(
                        str(input_path),
                        recursive=config['recursive_search']
                    )
                    for img_path in dir_images:
                        image_data.append({
                            'image_path': img_path,
                            'ava_score': 5.0  # Default score if not provided
                        })
                
                else:
                    logger.error(f"Input path does not exist: {input_path}")
        
        # Apply sample limit if specified (only for non-AVA datasets, as AVA loader handles it)
        if not args.ava_dataset and args.num_samples and len(image_data) > args.num_samples:
            logger.info(f"Limiting to {args.num_samples} samples (from {len(image_data)} total)")
            image_data = image_data[:args.num_samples]
        
        if not image_data:
            logger.error("No images found to process!")
            return 1
        
        logger.info(f"Found {len(image_data)} images to process")
        
        # Create output directory
        output_dir = create_output_directory(args.output)
        
        # Save run metadata
        save_run_metadata(output_dir, args, config, [item['image_path'] for item in image_data])
        
        # Generate pseudolabels
        logger.info("Starting pseudolabel generation...")
        results = generator.generate_pseudolabels(
            image_data=image_data,
            output_path=str(output_dir)
        )
        
        # Print summary
        logger.info("=" * 60)
        logger.info("PSEUDOLABEL GENERATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total images: {len(image_data)}")
        logger.info(f"Successfully processed: {len(results['results'])}")
        logger.info(f"Results saved to: {results['output_file']}")
        logger.info("=" * 60)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return 130
    
    except Exception as e:
        logger.error(f"Error during pseudolabel generation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
