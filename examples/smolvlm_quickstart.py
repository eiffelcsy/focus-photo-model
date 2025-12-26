#!/usr/bin/env python3
"""
Quick start example for SmolVLM fine-tuning.

This script demonstrates the minimal code needed to fine-tune SmolVLM
on your pseudolabeled aesthetic scores.

Usage:
    python examples/smolvlm_quickstart.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.smolvlm_trainer import SmolVLMTrainer


def main():
    """
    Minimal example of SmolVLM fine-tuning.
    """
    
    print("=" * 80)
    print("SmolVLM Quick Start Example")
    print("=" * 80)
    
    # 1. Initialize trainer with config
    print("\n[1/3] Initializing trainer...")
    trainer = SmolVLMTrainer(config_path="config/training/smolvlm_config.yaml")
    
    # Optional: Override config for quick testing
    trainer.config['dataset']['max_samples'] = 100  # Use only 100 samples
    trainer.config['training']['num_train_epochs'] = 1  # Train for 1 epoch
    trainer.config['training']['save_steps'] = 50  # Save every 50 steps
    
    # 2. Run the complete pipeline
    print("\n[2/3] Running training pipeline...")
    train_result = trainer.run_full_pipeline()
    
    # 3. Evaluate on test set
    print("\n[3/3] Evaluating on test set...")
    eval_result = trainer.evaluate()
    
    print("\n" + "=" * 80)
    print("Quick Start Complete!")
    print("=" * 80)
    print(f"\nModel saved to: {trainer.config['training']['output_dir']}")
    print(f"\nTest Loss: {eval_result.get('eval_loss', 'N/A')}")
    
    print("\nNext steps:")
    print("1. Run full training: python scripts/train_smolvlm.py")
    print("2. Run inference: python scripts/inference_smolvlm.py --image path/to/image.jpg --adapter outputs/smolvlm-aesthetic-scorer")
    print("3. Read the full guide: docs/SMOLVLM_FINETUNING.md")


if __name__ == '__main__':
    main()

