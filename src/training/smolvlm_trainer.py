"""
SmolVLM trainer with QLoRA support for aesthetic score prediction.
"""

import os
import gc
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import yaml
from transformers import (
    Idefics3ForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer, SFTConfig

from .smolvlm_dataset import load_and_split_dataset, AestheticScoreDataset


class SmolVLMTrainer:
    """
    Trainer for fine-tuning SmolVLM on aesthetic score prediction.
    Uses QLoRA for efficient training on consumer GPUs.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the trainer with configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        self.model = None
        self.processor = None
        self.trainer = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        print(f"Initialized SmolVLMTrainer with config: {config_path}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup_model_and_processor(self):
        """
        Load the base model and processor with quantization.
        """
        model_config = self.config['model']
        model_id = model_config['model_id']
        
        print(f"Loading model: {model_id}")
        
        # Setup quantization config
        quant_config = model_config.get('quantization', {})
        if quant_config.get('load_in_4bit', False):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=quant_config.get('bnb_4bit_use_double_quant', True),
                bnb_4bit_quant_type=quant_config.get('bnb_4bit_quant_type', 'nf4'),
                bnb_4bit_compute_dtype=getattr(torch, quant_config.get('bnb_4bit_compute_dtype', 'bfloat16'))
            )
            print("Using 4-bit quantization (QLoRA)")
        else:
            bnb_config = None
            print("No quantization enabled")
        
        # Load model
        dtype = getattr(torch, model_config.get('torch_dtype', 'bfloat16'))
        attn_impl = model_config.get('attn_implementation', 'flash_attention_2')
        
        self.model = Idefics3ForConditionalGeneration.from_pretrained(
            model_id,
            device_map=model_config.get('device_map', 'auto'),
            torch_dtype=dtype,
            quantization_config=bnb_config,
            _attn_implementation=attn_impl,
        )
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        print(f"Model loaded successfully")
        print(f"Model device: {self.model.device}")
        print(f"Model dtype: {self.model.dtype}")
    
    def setup_lora(self):
        """
        Setup LoRA adapters for efficient fine-tuning.
        """
        lora_config = self.config['lora']
        
        peft_config = LoraConfig(
            r=lora_config.get('r', 8),
            lora_alpha=lora_config.get('lora_alpha', 8),
            lora_dropout=lora_config.get('lora_dropout', 0.1),
            target_modules=lora_config.get('target_modules', [
                'down_proj', 'o_proj', 'k_proj', 'q_proj', 'gate_proj', 'up_proj', 'v_proj'
            ]),
            use_dora=lora_config.get('use_dora', True),
            init_lora_weights=lora_config.get('init_lora_weights', 'gaussian'),
        )
        
        # Apply PEFT model adaptation
        self.model = get_peft_model(self.model, peft_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        print("LoRA adapters configured successfully")
    
    def setup_datasets(self):
        """
        Load and prepare datasets for training.
        """
        dataset_config = self.config['dataset']
        prompt_config = self.config['prompt']
        
        pseudolabels_path = dataset_config['pseudolabels_path']
        images_dir = dataset_config['images_dir']
        
        print(f"Loading datasets from: {pseudolabels_path}")
        
        self.train_dataset, self.val_dataset, self.test_dataset = load_and_split_dataset(
            pseudolabels_path=pseudolabels_path,
            images_dir=images_dir,
            system_message=prompt_config['system_message'],
            user_message=prompt_config['user_message'],
            train_split=dataset_config.get('train_split', 0.8),
            val_split=dataset_config.get('val_split', 0.1),
            test_split=dataset_config.get('test_split', 0.1),
            random_seed=dataset_config.get('random_seed', 42),
            max_samples=dataset_config.get('max_samples', None),
            shuffle=dataset_config.get('shuffle', True),
        )
        
        print(f"Datasets loaded: train={len(self.train_dataset)}, "
              f"val={len(self.val_dataset)}, test={len(self.test_dataset)}")
    
    def setup_trainer(self):
        """
        Setup the TRL SFTTrainer for fine-tuning.
        """
        training_config = self.config['training']
        
        # Create SFTConfig
        training_args = SFTConfig(
            output_dir=training_config['output_dir'],
            num_train_epochs=training_config.get('num_train_epochs', 3),
            per_device_train_batch_size=training_config.get('per_device_train_batch_size', 2),
            per_device_eval_batch_size=training_config.get('per_device_eval_batch_size', 2),
            gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 8),
            gradient_checkpointing=training_config.get('gradient_checkpointing', True),
            
            # Optimizer
            optim=training_config.get('optim', 'adamw_torch_fused'),
            learning_rate=training_config.get('learning_rate', 1e-4),
            weight_decay=training_config.get('weight_decay', 0.01),
            warmup_steps=training_config.get('warmup_steps', 100),
            lr_scheduler_type=training_config.get('lr_scheduler_type', 'cosine'),
            
            # Mixed precision
            bf16=training_config.get('bf16', True),
            fp16=training_config.get('fp16', False),
            
            # Logging
            logging_steps=training_config.get('logging_steps', 10),
            logging_dir=training_config.get('logging_dir', None),
            report_to=training_config.get('report_to', 'tensorboard'),
            
            # Checkpointing
            save_strategy=training_config.get('save_strategy', 'steps'),
            save_steps=training_config.get('save_steps', 100),
            save_total_limit=training_config.get('save_total_limit', 3),
            load_best_model_at_end=training_config.get('load_best_model_at_end', True),
            metric_for_best_model=training_config.get('metric_for_best_model', 'eval_loss'),
            
            # Evaluation
            eval_strategy=training_config.get('eval_strategy', 'steps'),
            eval_steps=training_config.get('eval_steps', 100),
            
            # Other
            max_length=training_config.get('max_length', None),
            remove_unused_columns=training_config.get('remove_unused_columns', False),
            dataloader_num_workers=training_config.get('dataloader_num_workers', 4),
            dataloader_pin_memory=training_config.get('dataloader_pin_memory', True),
            
            # Hub
            push_to_hub=training_config.get('push_to_hub', False),
            hub_model_id=training_config.get('hub_model_id', None),
            hub_strategy=training_config.get('hub_strategy', 'every_save'),
        )
        
        # Create trainer
        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            processing_class=self.processor,
        )
        
        print("Trainer configured successfully")
    
    def train(self):
        """
        Run the training loop.
        """
        if self.trainer is None:
            raise ValueError("Trainer not setup. Call setup_trainer() first.")
        
        print("Starting training...")
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
        
        # Train
        train_result = self.trainer.train()
        
        print("Training completed!")
        print(f"Training metrics: {train_result.metrics}")
        
        return train_result
    
    def save_model(self, output_dir: Optional[str] = None):
        """
        Save the fine-tuned model.
        
        Args:
            output_dir: Directory to save the model (uses config if None)
        """
        if output_dir is None:
            output_dir = self.config['training']['output_dir']
        
        print(f"Saving model to: {output_dir}")
        self.trainer.save_model(output_dir)
        
        # Also save processor
        self.processor.save_pretrained(output_dir)
        
        print("Model saved successfully")
    
    def evaluate(self, dataset: Optional[AestheticScoreDataset] = None) -> Dict:
        """
        Evaluate the model on a dataset.
        
        Args:
            dataset: Dataset to evaluate on (uses test_dataset if None)
        
        Returns:
            Dictionary of evaluation metrics
        """
        if dataset is None:
            dataset = self.test_dataset
        
        if self.trainer is None:
            raise ValueError("Trainer not setup. Call setup_trainer() first.")
        
        print(f"Evaluating on {len(dataset)} samples...")
        
        # Evaluate
        eval_result = self.trainer.evaluate(eval_dataset=dataset)
        
        print(f"Evaluation metrics: {eval_result}")
        
        return eval_result
    
    def clear_memory(self):
        """
        Clear GPU memory and delete model/trainer objects.
        """
        print("Clearing memory...")
        
        # Delete objects
        if hasattr(self, 'trainer'):
            del self.trainer
            self.trainer = None
        
        if hasattr(self, 'model'):
            del self.model
            self.model = None
        
        if hasattr(self, 'processor'):
            del self.processor
            self.processor = None
        
        # Garbage collection
        time.sleep(1)
        gc.collect()
        time.sleep(1)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            time.sleep(1)
            gc.collect()
            
            print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
        print("Memory cleared")
    
    def run_full_pipeline(self):
        """
        Run the complete training pipeline from setup to saving.
        """
        print("=" * 80)
        print("Starting SmolVLM Fine-tuning Pipeline")
        print("=" * 80)
        
        # Setup
        print("\n[1/6] Setting up model and processor...")
        self.setup_model_and_processor()
        
        print("\n[2/6] Setting up LoRA adapters...")
        self.setup_lora()
        
        print("\n[3/6] Loading datasets...")
        self.setup_datasets()
        
        print("\n[4/6] Configuring trainer...")
        self.setup_trainer()
        
        print("\n[5/6] Training model...")
        train_result = self.train()
        
        print("\n[6/6] Saving model...")
        self.save_model()
        
        print("\n" + "=" * 80)
        print("Fine-tuning pipeline completed successfully!")
        print("=" * 80)
        
        return train_result


def load_trained_model(
    base_model_id: str,
    adapter_path: str,
    device_map: str = "auto",
    torch_dtype: str = "bfloat16",
) -> Tuple[Idefics3ForConditionalGeneration, AutoProcessor]:
    """
    Load a fine-tuned SmolVLM model with adapters.
    
    Args:
        base_model_id: Base model ID (e.g., "HuggingFaceTB/SmolVLM-Instruct")
        adapter_path: Path to the trained adapter
        device_map: Device mapping strategy
        torch_dtype: Torch dtype for the model
    
    Returns:
        Tuple of (model, processor)
    """
    print(f"Loading base model: {base_model_id}")
    
    dtype = getattr(torch, torch_dtype)
    
    # Load base model
    model = Idefics3ForConditionalGeneration.from_pretrained(
        base_model_id,
        device_map=device_map,
        torch_dtype=dtype,
        _attn_implementation="flash_attention_2",
    )
    
    # Load processor
    processor = AutoProcessor.from_pretrained(base_model_id)
    
    # Load adapter
    print(f"Loading adapter from: {adapter_path}")
    model.load_adapter(adapter_path)
    
    print("Model and adapter loaded successfully")
    
    return model, processor

