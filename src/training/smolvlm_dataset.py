"""
Dataset loader for SmolVLM fine-tuning with pseudolabeled aesthetic scores.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class AestheticScoreDataset(Dataset):
    """
    Dataset for loading pseudolabeled aesthetic scores for SmolVLM fine-tuning.
    
    Each sample contains:
    - image: PIL Image
    - scores: dict with impact, style, composition, lighting, color_balance
    - reasoning: text explanation
    - image_path: path to image file
    """
    
    def __init__(
        self,
        pseudolabels_path: str,
        images_dir: str,
        system_message: str,
        user_message: str,
        max_samples: Optional[int] = None,
        shuffle: bool = True,
        random_seed: int = 42,
    ):
        """
        Initialize the dataset.
        
        Args:
            pseudolabels_path: Path to JSON file with pseudolabels
            images_dir: Directory containing images
            system_message: System prompt for the model
            user_message: User query message
            max_samples: Maximum number of samples to load (None for all)
            shuffle: Whether to shuffle the data
            random_seed: Random seed for reproducibility
        """
        self.images_dir = Path(images_dir)
        self.system_message = system_message
        self.user_message = user_message
        
        # Load pseudolabels
        with open(pseudolabels_path, 'r') as f:
            self.data = json.load(f)
        
        # Filter out samples with missing required fields
        self.data = [
            sample for sample in self.data
            if all(k in sample for k in ['impact', 'style', 'composition', 'lighting', 'color_balance', 'reasoning'])
        ]
        
        # Shuffle if requested
        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(self.data)
        
        # Limit samples if requested
        if max_samples is not None:
            self.data = self.data[:max_samples]
        
        print(f"Loaded {len(self.data)} samples from {pseudolabels_path}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.
        
        Returns:
            Dict with 'images' and 'messages' keys formatted for SmolVLM
        """
        sample = self.data[idx]
        
        # Load image
        image_path = sample.get('image_path', '')
        if not os.path.isabs(image_path):
            # If relative path, join with images_dir
            image_path = self.images_dir / Path(image_path).name
        else:
            image_path = Path(image_path)
        
        # Load and convert image to RGB
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image as fallback
            image = Image.new('RGB', (224, 224), color='white')
        
        # Format the assistant response as JSON
        # Note: Only include scores for training, not reasoning
        # This focuses the model on learning accurate score prediction
        assistant_response = {
            "impact": sample['impact'],
            "style": sample['style'],
            "composition": sample['composition'],
            "lighting": sample['lighting'],
            "color_balance": sample['color_balance']
        }
        
        # Format in chat template structure
        formatted_sample = {
            "images": [image],
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": self.system_message
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
                            "text": self.user_message,
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(assistant_response, indent=2)
                        }
                    ],
                },
            ]
        }
        
        return formatted_sample


def load_and_split_dataset(
    pseudolabels_path: str,
    images_dir: str,
    system_message: str,
    user_message: str,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    random_seed: int = 42,
    max_samples: Optional[int] = None,
    shuffle: bool = True,
) -> Tuple[AestheticScoreDataset, AestheticScoreDataset, AestheticScoreDataset]:
    """
    Load pseudolabels and split into train/val/test datasets.
    
    Args:
        pseudolabels_path: Path to JSON file with pseudolabels
        images_dir: Directory containing images
        system_message: System prompt for the model
        user_message: User query message
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
        random_seed: Random seed for reproducibility
        max_samples: Maximum number of samples to load (None for all)
        shuffle: Whether to shuffle the data
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, \
        "Train, val, and test splits must sum to 1.0"
    
    # Load all data
    with open(pseudolabels_path, 'r') as f:
        all_data = json.load(f)
    
    # Filter out samples with missing required fields
    all_data = [
        sample for sample in all_data
        if all(k in sample for k in ['impact', 'style', 'composition', 'lighting', 'color_balance', 'reasoning'])
    ]
    
    # Shuffle if requested
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(all_data)
    
    # Limit samples if requested
    if max_samples is not None:
        all_data = all_data[:max_samples]
    
    # Calculate split indices
    n_samples = len(all_data)
    n_train = int(n_samples * train_split)
    n_val = int(n_samples * val_split)
    
    # Split the data
    train_data = all_data[:n_train]
    val_data = all_data[n_train:n_train + n_val]
    test_data = all_data[n_train + n_val:]
    
    print(f"Dataset splits: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    
    # Create temporary JSON files for each split
    import tempfile
    temp_dir = Path(tempfile.mkdtemp())
    
    train_path = temp_dir / "train.json"
    val_path = temp_dir / "val.json"
    test_path = temp_dir / "test.json"
    
    with open(train_path, 'w') as f:
        json.dump(train_data, f)
    with open(val_path, 'w') as f:
        json.dump(val_data, f)
    with open(test_path, 'w') as f:
        json.dump(test_data, f)
    
    # Create datasets
    train_dataset = AestheticScoreDataset(
        str(train_path), images_dir, system_message, user_message,
        max_samples=None, shuffle=False  # Already shuffled
    )
    val_dataset = AestheticScoreDataset(
        str(val_path), images_dir, system_message, user_message,
        max_samples=None, shuffle=False
    )
    test_dataset = AestheticScoreDataset(
        str(test_path), images_dir, system_message, user_message,
        max_samples=None, shuffle=False
    )
    
    return train_dataset, val_dataset, test_dataset


def collate_fn(batch: List[Dict]) -> List[Dict]:
    """
    Collate function for DataLoader.
    Since TRL's SFTTrainer expects a list of samples, we just return the batch as-is.
    
    Args:
        batch: List of samples from the dataset
    
    Returns:
        The batch as-is (list of dicts)
    """
    return batch

