"""
Pseudolabel Generator for AVA Dataset using Gemma 3 Vision-Language Model

This module generates detailed aesthetic attribute scores (impact, style, composition,
lighting, color_balance) for images using Google's Gemma 3 4B model.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

import torch
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from tqdm import tqdm
import yaml


class PseudolabelGenerator:
    """
    Generates pseudolabels for images using Gemma 3 vision-language model.
    
    Attributes:
        model: The Gemma 3 vision-language model
        processor: The image and text processor
        config: Configuration dictionary
        logger: Logger instance
    """
    
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict] = None):
        """
        Initialize the pseudolabel generator.
        
        Args:
            config_path: Path to YAML configuration file
            config_dict: Configuration dictionary (if provided, overrides config_path)
        """
        # Load configuration
        if config_dict is not None:
            self.config = config_dict
        elif config_path is not None:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            raise ValueError("Either config_path or config_dict must be provided")
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Load model and processor
        self.model = None
        self.processor = None
        self._load_model()
        
        # Track statistics
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'failed_images': []
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO if self.config['logging']['verbose'] else logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _load_model(self):
        """Load the Gemma 3 model and processor."""
        self.logger.info(f"Loading model: {self.config['model']['model_id']}")
        
        model_id = self.config['model']['model_id']
        device_map = self.config['model']['device_map']
        
        # Model loading kwargs
        model_kwargs = {
            'device_map': device_map,
        }
        
        # Add quantization if specified
        if self.config['model'].get('load_in_8bit', False):
            model_kwargs['load_in_8bit'] = True
        elif self.config['model'].get('load_in_4bit', False):
            model_kwargs['load_in_4bit'] = True
        
        # Load model
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            **model_kwargs
        ).eval()
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        
        self.logger.info("Model loaded successfully")
    
    def _build_prompt(self, ava_score: float) -> str:
        """
        Build the prompt for the model.
        
        Args:
            ava_score: The AVA aesthetic score for the image
            
        Returns:
            Formatted prompt string
        """
        criteria_text = "\n    ".join([
            f"{i+1}. {criterion['name'].upper()}: {criterion['description']}"
            for i, criterion in enumerate(self.config['prompt']['criteria'])
        ])
        
        prompt = f"""Analyze this photograph and rate it on these 5 criteria (scale 1-10):
    
    {criteria_text}
    """
        
        if self.config['prompt']['use_ava_score_as_context']:
            prompt += f"""
    The overall aesthetic score for this image is {ava_score:.1f}/10.
    Your individual scores should reflect this overall quality level.
    """
        
        prompt += """
    Respond in JSON format:
    {{
        "impact": X.X,
        "style": X.X,
        "composition": X.X,
        "lighting": X.X,
        "color_balance": X.X,
        "reasoning": "brief explanation of scores"
    }}
    """
        
        return prompt
    
    def _extract_json_from_response(self, response: str) -> Optional[Dict]:
        """
        Extract JSON from model response, handling various formats.
        
        Args:
            response: Raw model response
            
        Returns:
            Parsed JSON dictionary or None if parsing fails
        """
        try:
            # Try to parse directly
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON in markdown code blocks
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find any JSON object
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        self.logger.warning(f"Failed to extract JSON from response: {response[:200]}")
        return None
    
    def generate_pseudolabel(
        self, 
        image_path: str, 
        ava_score: float
    ) -> Optional[Dict]:
        """
        Generate pseudolabel for a single image.
        
        Args:
            image_path: Path to the image file
            ava_score: AVA aesthetic score for the image
            
        Returns:
            Dictionary containing pseudolabel scores or None if generation fails
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Build prompt
            prompt = self._build_prompt(ava_score)
            
            # Prepare messages
            system_message = self.config['prompt']['system_message']
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_message}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Process inputs
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )
            
            # Move to device and convert dtype
            dtype = getattr(torch, self.config['model']['dtype'])
            inputs = {k: v.to(self.model.device, dtype=dtype if v.dtype == torch.float32 else v.dtype) 
                     for k, v in inputs.items()}
            
            input_len = inputs["input_ids"].shape[-1]
            
            # Generate response
            gen_config = self.config['generation']
            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=gen_config['max_new_tokens'],
                    do_sample=gen_config['do_sample'],
                    temperature=gen_config.get('temperature', 1.0) if gen_config['do_sample'] else None,
                    top_p=gen_config.get('top_p', 1.0) if gen_config['do_sample'] else None,
                )
                generation = generation[0][input_len:]
            
            # Decode response
            decoded = self.processor.decode(generation, skip_special_tokens=True)
            
            # Extract JSON from response
            result = self._extract_json_from_response(decoded)
            
            if result is None:
                self.logger.error(f"Failed to parse response for {image_path}")
                return None
            
            # Validate required fields
            required_fields = [criterion['name'] for criterion in self.config['prompt']['criteria']]
            if not all(field in result for field in required_fields):
                self.logger.error(f"Missing required fields in response for {image_path}")
                return None
            
            # Add metadata
            result['image_path'] = image_path
            result['ava_score'] = ava_score
            result['raw_response'] = decoded
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {str(e)}")
            return None
    
    def load_ava_dataset(self) -> List[Tuple[str, float]]:
        """
        Load AVA dataset from CSV/TXT file.
        
        Returns:
            List of tuples (image_id, ava_score)
        """
        ava_path = self.config['dataset']['ava_csv_path']
        self.logger.info(f"Loading AVA dataset from {ava_path}")
        
        data = []
        with open(ava_path, 'r') as f:
            # Read first line to check if it's CSV format
            first_line = f.readline().strip()
            
            # Determine delimiter (comma for CSV, space for TXT)
            delimiter = ',' if ',' in first_line else ' '
            
            # Check if first line is a header
            is_header = 'image' in first_line.lower() or 'vote' in first_line.lower()
            
            # If not a header, process it
            if not is_header:
                f.seek(0)  # Reset to beginning
            
            for line in f:
                parts = line.strip().split(delimiter)
                if len(parts) >= 2:
                    # CSV format: image_id, vote_1, vote_2, ..., vote_10
                    # TXT format: image_id, score_1, score_2, ..., score_10, ...
                    image_id = parts[0].strip()
                    
                    # Skip empty lines or invalid data
                    if not image_id:
                        continue
                    
                    # Try to compute AVA score from vote distribution
                    if len(parts) >= 11:
                        try:
                            # Votes are in columns 1-10 (normalized or counts)
                            votes = [float(parts[i]) for i in range(1, 11)]
                            total_votes = sum(votes)
                            
                            if total_votes > 0:
                                # Compute weighted average score (1-10)
                                weighted_sum = sum((i + 1) * votes[i] for i in range(10))
                                ava_score = weighted_sum / total_votes
                            else:
                                continue
                        except (ValueError, IndexError):
                            continue
                    else:
                        # Assume second column is the pre-computed score
                        try:
                            ava_score = float(parts[1])
                        except (ValueError, IndexError):
                            continue
                    
                    data.append((image_id, ava_score))
        
        self.logger.info(f"Loaded {len(data)} images from AVA dataset")
        return data
    
    def load_checkpoint(self, output_dir: str) -> set:
        """
        Load checkpoint to resume processing.
        
        Args:
            output_dir: Directory containing output files
            
        Returns:
            Set of already processed image IDs
        """
        checkpoint_file = os.path.join(output_dir, 'checkpoint.txt')
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                processed = set(line.strip() for line in f)
            self.logger.info(f"Resuming from checkpoint: {len(processed)} images already processed")
            return processed
        return set()
    
    def save_checkpoint(self, output_dir: str, processed_ids: set):
        """Save checkpoint of processed image IDs."""
        checkpoint_file = os.path.join(output_dir, 'checkpoint.txt')
        with open(checkpoint_file, 'w') as f:
            for image_id in sorted(processed_ids):
                f.write(f"{image_id}\n")
    
    def process_dataset(self):
        """Process entire AVA dataset and generate pseudolabels."""
        # Load dataset
        ava_data = self.load_ava_dataset()
        
        # Apply max_images limit if specified
        max_images = self.config['processing'].get('max_images')
        if max_images is not None:
            ava_data = ava_data[:max_images]
            self.logger.info(f"Limited to {max_images} images")
        
        # Setup output directory
        output_dir = self.config['dataset']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        # Load checkpoint if resuming
        processed_ids = set()
        if self.config['processing']['resume_from_checkpoint']:
            processed_ids = self.load_checkpoint(output_dir)
        
        # Setup output file
        output_format = self.config['dataset']['output_format']
        output_file = os.path.join(output_dir, f'pseudolabels.{output_format}')
        
        # Open output file in append mode
        if output_format == 'json':
            results = []
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    try:
                        results = json.load(f)
                    except json.JSONDecodeError:
                        results = []
        
        images_dir = self.config['dataset']['images_dir']
        checkpoint_interval = self.config['processing']['checkpoint_interval']
        log_interval = self.config['logging']['log_interval']
        
        # Process each image
        for idx, (image_id, ava_score) in enumerate(tqdm(ava_data, desc="Generating pseudolabels")):
            # Skip if already processed
            if image_id in processed_ids:
                continue
            
            # Build image path
            image_path = os.path.join(images_dir, f"{image_id}.jpg")
            if not os.path.exists(image_path):
                self.logger.warning(f"Image not found: {image_path}")
                self.stats['failed'] += 1
                self.stats['failed_images'].append(image_id)
                continue
            
            # Generate pseudolabel
            result = self.generate_pseudolabel(image_path, ava_score)
            
            if result is not None:
                result['image_id'] = image_id
                results.append(result)
                processed_ids.add(image_id)
                self.stats['successful'] += 1
            else:
                self.stats['failed'] += 1
                self.stats['failed_images'].append(image_id)
            
            self.stats['total_processed'] += 1
            
            # Log progress
            if (idx + 1) % log_interval == 0:
                self.logger.info(
                    f"Processed {self.stats['total_processed']} images | "
                    f"Successful: {self.stats['successful']} | "
                    f"Failed: {self.stats['failed']}"
                )
            
            # Save checkpoint
            if (idx + 1) % checkpoint_interval == 0:
                # Save results
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                
                # Save checkpoint
                self.save_checkpoint(output_dir, processed_ids)
                self.logger.info(f"Checkpoint saved at {idx + 1} images")
        
        # Final save
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.save_checkpoint(output_dir, processed_ids)
        
        # Save failed images list
        if self.config['logging']['save_failed_images'] and self.stats['failed_images']:
            failed_file = os.path.join(output_dir, 'failed_images.txt')
            with open(failed_file, 'w') as f:
                for image_id in self.stats['failed_images']:
                    f.write(f"{image_id}\n")
        
        # Print final statistics
        self.logger.info("=" * 50)
        self.logger.info("Pseudolabel Generation Complete")
        self.logger.info(f"Total processed: {self.stats['total_processed']}")
        self.logger.info(f"Successful: {self.stats['successful']}")
        self.logger.info(f"Failed: {self.stats['failed']}")
        self.logger.info(f"Output saved to: {output_file}")
        self.logger.info("=" * 50)


if __name__ == "__main__":
    # Example usage
    config_path = "config/training/pseudolabel_config.yaml"
    generator = PseudolabelGenerator(config_path=config_path)
    generator.process_dataset()

