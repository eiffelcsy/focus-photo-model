"""
Fine-tune SmolVLM on aesthetic score prediction using pseudolabels.

This script uses the TRL library with QLoRA for efficient fine-tuning on consumer GPUs.

Usage:
    # Basic usage with default config
    python scripts/train_smolvlm.py
    
    # Custom config file
    python scripts/train_smolvlm.py --config config/training/custom_config.yaml
    
    # Override specific parameters
    python scripts/train_smolvlm.py --learning-rate 5e-5 --batch-size 4 --epochs 5
    
    # Use wandb for logging
    python scripts/train_smolvlm.py --use-wandb --wandb-project "smolvlm-aesthetic"
    
    # Limit dataset size for testing
    python scripts/train_smolvlm.py --max-samples 100
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.smolvlm_trainer import SmolVLMTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune SmolVLM for aesthetic score prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Config file
    parser.add_argument(
        '--config',
        type=str,
        default='config/training/smolvlm_config.yaml',
        help='Path to configuration YAML file'
    )
    
    # Dataset overrides
    parser.add_argument(
        '--pseudolabels-path',
        type=str,
        help='Path to pseudolabels JSON file (overrides config)'
    )
    parser.add_argument(
        '--images-dir',
        type=str,
        help='Directory containing images (overrides config)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        help='Maximum number of samples to use (overrides config)'
    )
    
    # Training overrides
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for model checkpoints (overrides config)'
    )
    parser.add_argument(
        '--learning-rate', '--lr',
        type=float,
        help='Learning rate (overrides config)'
    )
    parser.add_argument(
        '--batch-size', '--bs',
        type=int,
        help='Per-device training batch size (overrides config)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs (overrides config)'
    )
    parser.add_argument(
        '--gradient-accumulation-steps',
        type=int,
        help='Gradient accumulation steps (overrides config)'
    )
    
    # Model overrides
    parser.add_argument(
        '--model-id',
        type=str,
        help='Model ID to use (overrides config)'
    )
    parser.add_argument(
        '--no-quantization',
        action='store_true',
        help='Disable 4-bit quantization (requires more GPU memory)'
    )
    
    # LoRA overrides
    parser.add_argument(
        '--lora-r',
        type=int,
        help='LoRA rank (overrides config)'
    )
    parser.add_argument(
        '--lora-alpha',
        type=int,
        help='LoRA alpha (overrides config)'
    )
    
    # Logging
    parser.add_argument(
        '--use-wandb',
        action='store_true',
        help='Use Weights & Biases for logging'
    )
    parser.add_argument(
        '--wandb-project',
        type=str,
        default='smolvlm-aesthetic-scorer',
        help='W&B project name'
    )
    parser.add_argument(
        '--wandb-run-name',
        type=str,
        help='W&B run name'
    )
    
    # Evaluation
    parser.add_argument(
        '--eval-only',
        action='store_true',
        help='Only run evaluation (no training)'
    )
    parser.add_argument(
        '--adapter-path',
        type=str,
        help='Path to trained adapter for evaluation'
    )
    
    # Other
    parser.add_argument(
        '--skip-setup',
        action='store_true',
        help='Skip model setup (for debugging)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def apply_overrides(trainer: SmolVLMTrainer, args: argparse.Namespace):
    """
    Apply command-line argument overrides to trainer config.
    
    Args:
        trainer: SmolVLMTrainer instance
        args: Parsed command-line arguments
    """
    # Dataset overrides
    if args.pseudolabels_path:
        trainer.config['dataset']['pseudolabels_path'] = args.pseudolabels_path
    if args.images_dir:
        trainer.config['dataset']['images_dir'] = args.images_dir
    if args.max_samples:
        trainer.config['dataset']['max_samples'] = args.max_samples
    
    # Training overrides
    if args.output_dir:
        trainer.config['training']['output_dir'] = args.output_dir
    if args.learning_rate:
        trainer.config['training']['learning_rate'] = args.learning_rate
    if args.batch_size:
        trainer.config['training']['per_device_train_batch_size'] = args.batch_size
        trainer.config['training']['per_device_eval_batch_size'] = args.batch_size
    if args.epochs:
        trainer.config['training']['num_train_epochs'] = args.epochs
    if args.gradient_accumulation_steps:
        trainer.config['training']['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    
    # Model overrides
    if args.model_id:
        trainer.config['model']['model_id'] = args.model_id
    if args.no_quantization:
        trainer.config['model']['quantization']['load_in_4bit'] = False
    
    # LoRA overrides
    if args.lora_r:
        trainer.config['lora']['r'] = args.lora_r
    if args.lora_alpha:
        trainer.config['lora']['lora_alpha'] = args.lora_alpha
    
    # Logging overrides
    if args.use_wandb:
        trainer.config['training']['report_to'] = 'wandb'
        # Set wandb environment variables
        import os
        os.environ['WANDB_PROJECT'] = args.wandb_project
        if args.wandb_run_name:
            os.environ['WANDB_RUN_NAME'] = args.wandb_run_name
    
    # Verbose logging
    if args.verbose:
        trainer.config['logging']['verbose'] = True
        trainer.config['logging']['log_level'] = 'DEBUG'


def main():
    """Main training script."""
    args = parse_args()
    
    print("=" * 80)
    print("SmolVLM Fine-tuning for Aesthetic Score Prediction")
    print("=" * 80)
    print(f"\nConfiguration file: {args.config}")
    
    # Initialize trainer
    trainer = SmolVLMTrainer(config_path=args.config)
    
    # Apply command-line overrides
    apply_overrides(trainer, args)
    
    # Print key configuration
    print("\nKey Configuration:")
    print(f"  Model: {trainer.config['model']['model_id']}")
    print(f"  Quantization: {trainer.config['model']['quantization']['load_in_4bit']}")
    print(f"  LoRA rank: {trainer.config['lora']['r']}")
    print(f"  Learning rate: {trainer.config['training']['learning_rate']}")
    print(f"  Batch size: {trainer.config['training']['per_device_train_batch_size']}")
    print(f"  Epochs: {trainer.config['training']['num_train_epochs']}")
    print(f"  Output dir: {trainer.config['training']['output_dir']}")
    print(f"  Pseudolabels: {trainer.config['dataset']['pseudolabels_path']}")
    print(f"  Images dir: {trainer.config['dataset']['images_dir']}")
    
    if args.eval_only:
        # Evaluation only mode
        print("\n" + "=" * 80)
        print("Running evaluation only (no training)")
        print("=" * 80)
        
        if not args.adapter_path:
            print("ERROR: --adapter-path required for --eval-only mode")
            sys.exit(1)
        
        # Setup model with adapter
        from src.training.smolvlm_trainer import load_trained_model
        
        model, processor = load_trained_model(
            base_model_id=trainer.config['model']['model_id'],
            adapter_path=args.adapter_path,
            device_map=trainer.config['model']['device_map'],
            torch_dtype=trainer.config['model']['torch_dtype'],
        )
        
        trainer.model = model
        trainer.processor = processor
        
        # Load datasets
        trainer.setup_datasets()
        
        # Setup trainer (needed for evaluation)
        trainer.setup_trainer()
        
        # Evaluate
        eval_result = trainer.evaluate()
        
        print("\nEvaluation Results:")
        for key, value in eval_result.items():
            print(f"  {key}: {value}")
    
    else:
        # Full training pipeline
        if args.skip_setup:
            print("\nSkipping model setup (debug mode)")
        else:
            train_result = trainer.run_full_pipeline()
            
            print("\nTraining Results:")
            for key, value in train_result.metrics.items():
                print(f"  {key}: {value}")
            
            # Evaluate on test set
            print("\n" + "=" * 80)
            print("Evaluating on test set")
            print("=" * 80)
            
            eval_result = trainer.evaluate()
            
            print("\nTest Set Results:")
            for key, value in eval_result.items():
                print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == '__main__':
    main()

