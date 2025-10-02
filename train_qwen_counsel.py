#!/usr/bin/env python3
"""
Training script for fine-tuning Qwen2.5 on Counsel Chat dataset.
This script uses modern training techniques including LoRA, gradient checkpointing, and mixed precision.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training,
)
from datasets import load_from_disk, Dataset
import wandb
from accelerate import Accelerator


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CounselChatTrainer:
    """Trainer class for fine-tuning Qwen2.5 on Counsel Chat dataset."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
    def setup_tokenizer(self) -> None:
        """Initialize the tokenizer."""
        model_name = self.config["model_name"]
        logger.info(f"Loading tokenizer from {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logger.info(f"Tokenizer loaded. Vocab size: {len(self.tokenizer)}")
    
    def setup_model(self) -> None:
        """Initialize the model with quantization and LoRA."""
        model_name = self.config["model_name"]
        logger.info(f"Loading model from {model_name}")
        
        # Configure quantization for memory efficiency
        bnb_config = None
        if self.config.get("use_4bit", True):
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
                logger.info("Using 4-bit quantization with BitsAndBytes")
            except ImportError:
                logger.warning("BitsAndBytes not available (likely on macOS ARM64). Using standard model loading.")
                bnb_config = None
        
        # Load model
        if bnb_config:
            # Use device_map="auto" for quantized models
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
        else:
            # For non-quantized models, load to CPU first to avoid meta device issues
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32,  # Use float32 for CPU
                device_map="cpu",
            )
        
        # Prepare model for k-bit training
        if bnb_config:
            try:
                self.model = prepare_model_for_kbit_training(self.model)
            except ImportError:
                logger.warning("prepare_model_for_kbit_training not available. Skipping.")
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.get("lora_r", 16),
            lora_alpha=self.config.get("lora_alpha", 32),
            lora_dropout=self.config.get("lora_dropout", 0.1),
            target_modules=self.config.get("lora_target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            bias="none",
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        logger.info("Model loaded and LoRA configured")
    
    def load_dataset(self) -> Dataset:
        """Load and prepare the dataset."""
        dataset_path = self.config["dataset_path"]
        logger.info(f"Loading dataset from {dataset_path}")
        
        dataset = load_from_disk(dataset_path)
        logger.info(f"Dataset loaded. Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}")
        
        return dataset
    
    def format_prompt(self, example: Dict[str, Any]) -> str:
        """Format the training example into a prompt."""
        instruction = example["instruction"]
        output = example["output"]
        
        # Use Qwen2.5 chat template format
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output}
        ]
        
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    
    def tokenize_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize the dataset."""
        # When batched=True, examples is a dict with lists of values
        # We need to create individual examples from the batch
        batch_size = len(examples["instruction"])
        individual_examples = []
        
        for i in range(batch_size):
            example = {
                "instruction": examples["instruction"][i],
                "input": examples["input"][i],
                "output": examples["output"][i],
                "topic": examples["topic"][i],
                "upvotes": examples["upvotes"][i],
                "question_id": examples["question_id"][i],
            }
            individual_examples.append(example)
        
        # Format prompts
        prompts = [self.format_prompt(ex) for ex in individual_examples]
        
        # Tokenize
        tokenized = self.tokenizer(
            prompts,
            truncation=True,
            padding=False,
            max_length=self.config.get("max_length", 2048),
            return_tensors=None,
        )
        
        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    def setup_data_collator(self) -> DataCollatorForSeq2Seq:
        """Setup data collator for training."""
        return DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            padding=True,
            return_tensors="pt",
        )
    
    def setup_training_arguments(self) -> TrainingArguments:
        """Setup training arguments."""
        output_dir = self.config["output_dir"]
        
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.get("num_epochs", 3),
            per_device_train_batch_size=self.config.get("batch_size", 4),
            per_device_eval_batch_size=self.config.get("eval_batch_size", 4),
            gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 4),
            learning_rate=self.config.get("learning_rate", 2e-4),
            weight_decay=self.config.get("weight_decay", 0.01),
            warmup_ratio=self.config.get("warmup_ratio", 0.1),
            lr_scheduler_type="cosine",
            logging_steps=self.config.get("logging_steps", 10),
            eval_steps=self.config.get("eval_steps", 100),
            save_steps=self.config.get("save_steps", 500),
            eval_strategy="steps",
            save_strategy="steps",
            save_total_limit=self.config.get("save_total_limit", 3),
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=self.config.get("fp16", False),
            bf16=self.config.get("bf16", True),
            dataloader_pin_memory=False,
            gradient_checkpointing=True,
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,
            report_to="wandb" if self.config.get("use_wandb", False) else "none",
            run_name=self.config.get("run_name", "qwen2.5-counsel-chat"),
            seed=self.config.get("seed", 42),
        )
    
    def setup_trainer(self, dataset: Dataset) -> Trainer:
        """Setup the trainer."""
        # Tokenize dataset
        logger.info("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
        )
        
        # Setup data collator
        data_collator = self.setup_data_collator()
        
        # Setup training arguments
        training_args = self.setup_training_arguments()
        
        # Setup trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if self.config.get("early_stopping", True) else None,
        )
        
        return trainer
    
    def train(self) -> None:
        """Main training function."""
        logger.info("Starting training setup...")
        
        # Setup components
        self.setup_tokenizer()
        self.setup_model()
        
        # Load dataset
        dataset = self.load_dataset()
        
        # Setup trainer
        self.trainer = self.setup_trainer(dataset)
        
        # Initialize wandb if enabled
        if self.config.get("use_wandb", False):
            wandb.init(
                project=self.config.get("wandb_project", "qwen2.5-counsel-chat"),
                name=self.config.get("run_name", "qwen2.5-counsel-chat"),
                config=self.config
            )
        
        # Start training
        logger.info("Starting training...")
        self.trainer.train()
        
        # Save final model
        logger.info("Saving final model...")
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.config["output_dir"])
        
        # Close wandb
        if self.config.get("use_wandb", False):
            wandb.finish()
        
        logger.info("Training completed!")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Train Qwen2.5 on Counsel Chat dataset")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.json",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name or path"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="counsel_chat_processed",
        help="Path to processed dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="qwen2.5-counsel-chat-finetuned",
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use Weights & Biases for logging"
    )
    
    args = parser.parse_args()
    
    # Create default config
    config = {
        "model_name": args.model_name,
        "dataset_path": args.dataset_path,
        "output_dir": args.output_dir,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "use_wandb": args.use_wandb,
        "use_4bit": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "max_length": 2048,
        "gradient_accumulation_steps": 4,
        "eval_batch_size": 4,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "logging_steps": 10,
        "eval_steps": 100,
        "save_steps": 500,
        "save_total_limit": 3,
        "fp16": False,
        "bf16": True,
        "early_stopping": True,
        "seed": 42,
        "run_name": "qwen2.5-counsel-chat",
        "wandb_project": "qwen2.5-counsel-chat",
    }
    
    # Load config file if it exists
    if os.path.exists(args.config):
        file_config = load_config(args.config)
        config.update(file_config)
    
    # Override with command line arguments
    for key, value in vars(args).items():
        if value is not None and key in config:
            config[key] = value
    
    # Create output directory
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(os.path.join(config["output_dir"], "training_config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize and start training
    trainer = CounselChatTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
