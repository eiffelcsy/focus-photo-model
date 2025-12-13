"""
Pseudolabel Generator for Photography Quality Assessment using Gemma 3-4B

This module implements a minimal pseudolabel generator that uses Gemma 3-4B model
to assess photography quality across 5 criteria: impact, style, composition, lighting, and color balance.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoTokenizer
from tqdm import tqdm
import re
import csv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_ava_dataset(
    csv_path: str,
    images_dir: str,
    max_samples: Optional[int] = None,
    check_image_exists: bool = True
) -> List[Dict[str, Any]]:
    """
    Load AVA dataset from CSV file and map to image paths.
    
    The CSV file contains vote distributions (vote_1 to vote_10) for each image.
    The mean aesthetic score is calculated as: sum(vote_i * i) for i in 1..10
    
    Args:
        csv_path: Path to the ground_truth_dataset.csv file
        images_dir: Directory containing the AVA images
        max_samples: Maximum number of samples to load (None for all)
        check_image_exists: Whether to verify that image files exist before including them
        
    Returns:
        List of dicts with 'image_path' and 'ava_score' keys
    """
    csv_path = Path(csv_path)
    images_dir = Path(images_dir)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    image_data = []
    skipped_missing = 0
    skipped_invalid = 0
    
    logger.info(f"Loading AVA dataset from {csv_path}")
    logger.info(f"Images directory: {images_dir}")
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row_idx, row in enumerate(tqdm(reader, desc="Loading AVA dataset"), start=1):
                if max_samples and len(image_data) >= max_samples:
                    break
                
                try:
                    # Get image number
                    image_num = row['image_num'].strip()
                    
                    # Calculate mean AVA score from vote distribution
                    # Score = sum(vote_i * i) for i in 1..10
                    mean_score = 0.0
                    for i in range(1, 11):
                        vote_key = f'vote_{i}'
                        if vote_key in row:
                            try:
                                vote_proportion = float(row[vote_key])
                                mean_score += vote_proportion * i
                            except (ValueError, TypeError):
                                pass
                    
                    # Construct image path
                    image_path = images_dir / f"{image_num}.jpg"
                    
                    # Check if image exists if requested
                    if check_image_exists and not image_path.exists():
                        skipped_missing += 1
                        if skipped_missing <= 10:  # Log first 10 missing images
                            logger.debug(f"Image not found: {image_path}")
                        continue
                    
                    image_data.append({
                        'image_path': str(image_path),
                        'ava_score': mean_score
                    })
                    
                except Exception as e:
                    skipped_invalid += 1
                    if skipped_invalid <= 10:  # Log first 10 invalid rows
                        logger.warning(f"Error processing row {row_idx}: {e}")
                    continue
        
        logger.info(f"Loaded {len(image_data)} images from AVA dataset")
        if skipped_missing > 0:
            logger.warning(f"Skipped {skipped_missing} images that were not found")
        if skipped_invalid > 0:
            logger.warning(f"Skipped {skipped_invalid} rows with invalid data")
        
        return image_data
        
    except Exception as e:
        logger.error(f"Error loading AVA dataset: {e}")
        raise


class PhotographyQualityDataset:
    """Simple dataset class for loading images."""
    
    def __init__(self, image_data: List[Dict[str, Any]]):
        """
        Args:
            image_data: List of dicts with 'image_path' and 'ava_score' keys
        """
        self.image_data = image_data
    
    def __len__(self):
        return len(self.image_data)
    
    def __getitem__(self, idx):
        data = self.image_data[idx]
        try:
            image = Image.open(data['image_path']).convert('RGB')
            return {
                'image': image,
                'image_path': data['image_path'],
                'image_id': Path(data['image_path']).stem,
                'ava_score': data.get('ava_score', 5.0)
            }
        except Exception as e:
            logger.error(f"Error loading image {data['image_path']}: {e}")
            return None


class GemmaPseudolabelGenerator:
    """
    Pseudolabel generator using Gemma 3-4B for photography quality assessment.
    
    Assesses images on 5 criteria: impact, style, composition, lighting, color_balance.
    """
    
    def __init__(
        self,
        model_name: str = "google/gemma-3-4b-it",
        device: str = "auto",
        max_new_tokens: int = 200,
        temperature: float = 0.3
    ):
        """
        Initialize the Gemma pseudolabel generator.
        
        Args:
            model_name: HuggingFace model identifier for Gemma
            device: Device to run inference on ('auto', 'cuda', 'cpu')
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature for generation
        """
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        # Photography assessment criteria
        self.criteria = ['impact', 'style', 'composition', 'lighting', 'color_balance']
        
        # Load model and processor
        self._load_model()
        
        # Create prompt template
        self.prompt_template = self._create_prompt_template()
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup the appropriate device for inference."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)
    
    def _load_model(self):
        """Load Gemma model and processor."""
        try:
            logger.info(f"Loading Gemma model: {self.model_name}")
            
            # Load processor and tokenizer
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Load model
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None
            )
            
            if self.device.type != "cuda":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _create_prompt_template(self) -> str:
        """Create the prompt template for assessment."""
        return """<image>Analyze this photograph and rate it on these 5 criteria (scale 1-10):

1. IMPACT: Emotional response and memorability upon first viewing
2. STYLE: Artistic expression and creative vision
3. COMPOSITION: Visual arrangement and balance of elements
4. LIGHTING: Quality and effectiveness of illumination
5. COLOR BALANCE: Harmony and effectiveness of color relationships

The overall aesthetic score for this image is {ava_score:.1f}/10.
Your individual scores should reflect this overall quality level.

Respond in JSON format:
{{
    "impact": X.X,
    "style": X.X,
    "composition": X.X,
    "lighting": X.X,
    "color_balance": X.X,
    "reasoning": "brief explanation of scores"
}}"""
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from the model."""
        default_scores = {
            'impact': 5.0,
            'style': 5.0,
            'composition': 5.0,
            'lighting': 5.0,
            'color_balance': 5.0,
            'reasoning': 'Error parsing response'
        }
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[^}]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                
                # Validate and clean scores
                for criterion in self.criteria:
                    if criterion in parsed:
                        try:
                            score = float(parsed[criterion])
                            parsed[criterion] = max(1.0, min(10.0, score))  # Clamp to 1-10
                        except (ValueError, TypeError):
                            parsed[criterion] = default_scores[criterion]
                    else:
                        parsed[criterion] = default_scores[criterion]
                
                if 'reasoning' not in parsed:
                    parsed['reasoning'] = 'No reasoning provided'
                
                return parsed
            else:
                logger.warning("No JSON found in response")
                return default_scores
                
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error: {e}")
            return default_scores
        except Exception as e:
            logger.warning(f"Error parsing response: {e}")
            return default_scores
    
    def assess_single_image(self, image: Image.Image, image_id: str, ava_score: float) -> Dict[str, Any]:
        """
        Assess a single image using the 5 criteria.
        
        Args:
            image: PIL Image to assess
            image_id: Unique identifier for the image
            ava_score: AVA aesthetic score for the image
            
        Returns:
            Dictionary containing assessment results
        """
        try:
            # Create prompt with AVA score
            prompt = self.prompt_template.format(ava_score=ava_score)
            
            # Prepare inputs - processor expects images as a list
            inputs = self.processor(
                text=prompt,
                images=[image],
                return_tensors="pt"
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            # Parse JSON response
            scores = self._parse_json_response(response)
            
            # Calculate composite score
            criterion_scores = [scores[c] for c in self.criteria]
            composite_score = sum(criterion_scores) / len(criterion_scores)
            
            return {
                'image_id': image_id,
                'ava_score': ava_score,
                'scores': scores,
                'composite_score': composite_score,
                'raw_response': response
            }
            
        except Exception as e:
            logger.error(f"Error assessing image {image_id}: {e}")
            # Return default scores on error
            default_scores = {c: 5.0 for c in self.criteria}
            default_scores['reasoning'] = f'Error during assessment: {str(e)}'
            
            return {
                'image_id': image_id,
                'ava_score': ava_score,
                'scores': default_scores,
                'composite_score': 5.0,
                'raw_response': ''
            }
    
    def generate_pseudolabels(
        self,
        image_data: List[Dict[str, Any]],
        output_path: str
    ) -> Dict[str, Any]:
        """
        Generate pseudolabels for a batch of images.
        
        Args:
            image_data: List of dicts with 'image_path' and 'ava_score' keys
            output_path: Directory to save results
            
        Returns:
            Dictionary containing batch processing results
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        dataset = PhotographyQualityDataset(image_data)
        all_results = []
        
        logger.info(f"Starting pseudolabel generation for {len(image_data)} images")
        
        for idx in tqdm(range(len(dataset)), desc="Generating pseudolabels"):
            item = dataset[idx]
            if item is None:
                continue
                
            try:
                # Assess the image
                result = self.assess_single_image(
                    item['image'], 
                    item['image_id'], 
                    item['ava_score']
                )
                result['source_path'] = item['image_path']
                all_results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process image {item['image_path']}: {e}")
        
        # Save results
        results_file = output_path / "pseudolabels.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        logger.info(f"Generated {len(all_results)} pseudolabels. Results saved to {results_file}")
        return {'results': all_results, 'output_file': str(results_file)}
    
