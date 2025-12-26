#!/usr/bin/env python3
"""
Example of using a fine-tuned SmolVLM model for inference.

This script demonstrates how to load a trained model and generate
aesthetic scores for images programmatically.

Usage:
    python examples/smolvlm_inference_example.py
"""

import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.smolvlm_trainer import load_trained_model
from scripts.inference_smolvlm import generate_scores_from_image
import yaml


def main():
    """
    Example of programmatic inference with SmolVLM.
    """
    
    print("=" * 80)
    print("SmolVLM Inference Example")
    print("=" * 80)
    
    # Configuration
    config_path = "config/training/smolvlm_config.yaml"
    adapter_path = "outputs/smolvlm-aesthetic-scorer"  # Path to your trained adapter
    image_path = "src/data/images/example.jpg"  # Replace with your image
    
    # Load config
    print("\n[1/4] Loading configuration...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    print("\n[2/4] Loading model and adapter...")
    model, processor = load_trained_model(
        base_model_id=config['model']['model_id'],
        adapter_path=adapter_path,
        device_map="auto",
        torch_dtype="bfloat16",
    )
    
    print("Model loaded successfully!")
    
    # Get prompts
    system_message = config['prompt']['system_message']
    user_message = config['prompt']['user_message']
    
    # Run inference
    print(f"\n[3/4] Running inference on: {image_path}")
    result = generate_scores_from_image(
        model=model,
        processor=processor,
        image_path=image_path,
        system_message=system_message,
        user_message=user_message,
        max_new_tokens=512,
        device="cuda",
    )
    
    # Display results
    print("\n[4/4] Results:")
    print("=" * 80)
    
    if result and 'error' not in result:
        print(f"Image: {result['image_path']}")
        print(f"\nAesthetic Scores:")
        print(f"  Impact:         {result['impact']:.1f}/10")
        print(f"  Style:          {result['style']:.1f}/10")
        print(f"  Composition:    {result['composition']:.1f}/10")
        print(f"  Lighting:       {result['lighting']:.1f}/10")
        print(f"  Color Balance:  {result['color_balance']:.1f}/10")
        print(f"\nReasoning:")
        print(f"  {result['reasoning']}")
        
        # Calculate overall average
        avg_score = (
            result['impact'] + 
            result['style'] + 
            result['composition'] + 
            result['lighting'] + 
            result['color_balance']
        ) / 5
        print(f"\nOverall Average: {avg_score:.1f}/10")
        
        # Save to JSON
        output_path = "example_result.json"
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
        print(f"Raw response: {result.get('raw_response', 'N/A')}")
    
    print("\n" + "=" * 80)
    print("Done!")


if __name__ == '__main__':
    main()

