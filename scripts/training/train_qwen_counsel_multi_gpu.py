#!/usr/bin/env python3
"""
Multi-GPU training script for fine-tuning Qwen2.5-14B on Counsel Chat dataset.
This script uses FSDP (Fully Sharded Data Parallel) for efficient multi-GPU training.
Optimized for 8 RTX 2080 Ti GPUs (11GB VRAM each).
"""

import os
import json
import logging
import argparse
import gc
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
    TrainerCallback,
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
from accelerate.utils import set_seed


# Set up logging
# Configure logging - will be adjusted based on rank in main()
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # Override any existing configuration
)
logger = logging.getLogger(__name__)


class MemoryManagementCallback(TrainerCallback):
    """Callback to manage GPU memory during training and checkpointing."""
    
    def on_save(self, args, state, control, **kwargs):
        """Clear cache before saving checkpoint to reduce OOM risk."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("Cleared CUDA cache before checkpoint save")
    
    def on_evaluate(self, args, state, control, **kwargs):
        """Clear cache before evaluation to reduce OOM risk."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("Cleared CUDA cache before evaluation")

def setup_logging_for_rank():
    """Configure logging based on process rank to avoid duplicate messages."""
    import warnings
    
    # Suppress tokenizer warnings about PAD/BOS/EOS tokens (harmless, just alignment)
    warnings.filterwarnings("ignore", message=".*tokenizer has new PAD/BOS/EOS tokens.*")
    warnings.filterwarnings("ignore", message=".*The tokenizer has new PAD/BOS/EOS tokens.*")
    
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if rank == 0:
        # Main process: log everything
        logging.getLogger().setLevel(logging.INFO)
        # Also configure transformers logging
        logging.getLogger("transformers").setLevel(logging.INFO)
        logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)  # Suppress tokenizer warnings
        logging.getLogger("datasets").setLevel(logging.WARNING)  # Reduce dataset verbosity
        logging.getLogger("accelerate").setLevel(logging.WARNING)  # Reduce accelerate verbosity
        logger.info(f"Logging configured for main process (rank {rank}, local_rank {local_rank})")
    else:
        # Other processes: only log warnings and errors
        logging.getLogger().setLevel(logging.WARNING)
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)  # Suppress tokenizer warnings
        logging.getLogger("datasets").setLevel(logging.WARNING)
        logging.getLogger("accelerate").setLevel(logging.WARNING)
        # Suppress all info/debug logs from worker processes
        logger.setLevel(logging.WARNING)


class MultiGPUTrainer:
    """Multi-GPU trainer class for fine-tuning Qwen2.5-14B on Counsel Chat dataset."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Don't initialize Accelerator here - accelerate launch will handle it
        # We'll get device info from environment variables for model loading
        self.device = None  # Will be set during model loading
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
    def setup_tokenizer(self) -> None:
        """Initialize the tokenizer."""
        model_name = self.config["model_name"]
        logger.info(f"Loading tokenizer from {model_name}")
        
        # Get cache directory from environment
        cache_dir = os.environ.get("TRANSFORMERS_CACHE", None)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="right",
            cache_dir=cache_dir,
        )
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Suppress tokenizer warnings about PAD/BOS/EOS tokens (harmless, just alignment)
        import warnings
        warnings.filterwarnings("ignore", message=".*tokenizer has new PAD/BOS/EOS tokens.*")
        warnings.filterwarnings("ignore", message=".*The tokenizer has new PAD/BOS/EOS tokens.*")
        # Also suppress in transformers logger
        logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
            
        logger.info(f"Tokenizer loaded. Vocab size: {len(self.tokenizer)}")
    
    def setup_model(self) -> None:
        """Initialize the model with quantization and LoRA for multi-GPU training."""
        model_name = self.config["model_name"]
        logger.info(f"Loading model from {model_name}")
        
        # Clean up memory before loading model
        self.cleanup_memory()
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        local_rank_env = int(os.environ.get("LOCAL_RANK", 0))
        bf16_requested = self.config.get("bf16", True)
        fp16_requested = self.config.get("fp16", False)
        if cuda_available:
            num_gpus = torch.cuda.device_count()
            logger.info(f"Number of GPUs available: {num_gpus}")
            for i in range(num_gpus):
                gpu_props = torch.cuda.get_device_properties(i)
                gpu_memory = gpu_props.total_memory / 1024**3
                logger.info(f"GPU {i}: {gpu_props.name} - {gpu_memory:.2f} GB VRAM")
            total_vram = sum(torch.cuda.get_device_properties(i).total_memory for i in range(num_gpus)) / 1024**3
            logger.info(f"Total VRAM across all GPUs: {total_vram:.2f} GB")
            capability = torch.cuda.get_device_capability(local_rank_env)
            bf16_supported = capability[0] >= 8
            if bf16_requested and not bf16_supported:
                logger.warning(
                    "Requested bf16 mixed precision, but the current GPU (compute capability "
                    f"sm_{capability[0]}{capability[1]}) does not support native bf16. Falling back to fp16."
                )
                self.config["bf16"] = False
                if not fp16_requested:
                    self.config["fp16"] = True
        
        if not cuda_available:
            logger.warning("="*60)
            logger.warning("CUDA not available. Training on CPU only (will be much slower).")
            logger.warning("="*60)
        
        # Configure quantization for memory efficiency
        # For 14B model on 8 RTX 2080 Ti (11GB each), 4-bit quantization is essential
        bnb_config = None
        use_4bit = self.config.get("use_4bit", True)
        
        if use_4bit and cuda_available:
            try:
                import bitsandbytes as bnb
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
                logger.info("Using 4-bit quantization with BitsAndBytes")
            except ImportError as e:
                logger.error("="*60)
                logger.error("BitsAndBytes not available but required for 4-bit quantization!")
                logger.error(f"Import error: {e}")
                logger.error("Please install bitsandbytes: pip install bitsandbytes")
                logger.error("="*60)
                raise RuntimeError(
                    "4-bit quantization is required for 14B model on RTX 2080 Ti GPUs. "
                    "Please install bitsandbytes."
                )
        
        # Get cache directory from environment
        cache_dir = os.environ.get("TRANSFORMERS_CACHE", None)
        
        # Load model
        # IMPORTANT: For multi-GPU training with DDP (not FSDP), we need to handle device placement carefully
        # BitsAndBytes quantized models are incompatible with FSDP due to integer tensors
        # We'll use DDP which works with quantized models
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        
        # Load base model and apply LoRA
        # Note: When resuming from checkpoint, Trainer will reload the model from checkpoint
        # This causes temporary double-loading, but we clear memory aggressively before that
        if bnb_config:
            logger.info("Loading model with 4-bit quantization...")
            # For DDP with quantized models, set CUDA device before loading
            # Each process will load the model on its assigned GPU
            
            # For BitsAndBytes with DDP, use device_map to specify the device
            # Each process loads the model on its assigned GPU
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map={"": f"cuda:{local_rank}"},  # Load on the current process's device
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                cache_dir=cache_dir,
            )
            logger.info(f"Model loaded with 4-bit quantization on device {device}")
            
            # Prepare model for k-bit training
            try:
                self.model = prepare_model_for_kbit_training(self.model)
                logger.info("Model prepared for k-bit training")
            except Exception as e:
                logger.warning(f"Error preparing model for k-bit training: {e}. Continuing anyway.")
            
            # CRITICAL FIX: Explicitly disable use_cache for BitsAndBytes quantized models
            # This prevents CUDA illegal memory access errors with gradient checkpointing
            if hasattr(self.model, 'config'):
                self.model.config.use_cache = False
            if hasattr(self.model, 'generation_config'):
                self.model.generation_config.use_cache = False
            logger.info("Disabled use_cache for BitsAndBytes quantized model")
        else:
            # For non-quantized models (not recommended for 14B on RTX 2080 Ti)
            logger.warning("Loading model without quantization - this may cause OOM on RTX 2080 Ti!")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                device_map=None,  # Let DDP handle device placement
                cache_dir=cache_dir,
            )
            # Move model to the correct device
            self.model = self.model.to(device)
        
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
        
        gradient_checkpointing = self.config.get("gradient_checkpointing", True)

        def disable_gc(model):
            """Recursively disable gradient checkpointing on model and submodules."""
            if hasattr(model, 'gradient_checkpointing_disable'):
                try:
                    model.gradient_checkpointing_disable()
                except Exception:
                    pass
            if hasattr(model, 'gradient_checkpointing'):
                try:
                    model.gradient_checkpointing = False
                except Exception:
                    pass
            if hasattr(model, '_gradient_checkpointing_func'):
                try:
                    model._gradient_checkpointing_func = None
                except Exception:
                    pass
            # Recursively check submodules
            for module in model.modules():
                if hasattr(module, 'gradient_checkpointing_disable'):
                    try:
                        module.gradient_checkpointing_disable()
                    except Exception:
                        pass
                if hasattr(module, 'gradient_checkpointing'):
                    try:
                        module.gradient_checkpointing = False
                    except Exception:
                        pass

        if gradient_checkpointing:
            logger.info("Enabling gradient checkpointing for memory savings.")
            if hasattr(self.model, "gradient_checkpointing_enable"):
                try:
                    self.model.gradient_checkpointing_enable()
                except Exception as e:
                    logger.warning(f"Failed to enable gradient checkpointing: {e}")
            if hasattr(self.model, "enable_input_require_grads"):
                try:
                    self.model.enable_input_require_grads()
                except Exception as e:
                    logger.warning(f"Failed to set input requires grad: {e}")
            if hasattr(self.model, "config"):
                self.model.config.use_cache = False
        else:
            disable_gc(self.model)
            logger.info("Gradient checkpointing disabled per configuration")
        
        # Ensure model is in training mode
        self.model.train()
        
        # Verify trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
        logger.info("Model loaded and LoRA configured")
    
    def cleanup_memory(self) -> None:
        """Clean up GPU memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def load_dataset(self) -> Dataset:
        """Load and prepare the dataset."""
        dataset_path = self.config["dataset_path"]
        
        # Resolve relative paths relative to project root
        if not os.path.isabs(dataset_path):
            script_dir = Path(__file__).parent  # scripts/training/
            project_root = script_dir.parent.parent  # FYP-LLM/
            resolved_path = project_root / dataset_path
            
            if not resolved_path.exists():
                if not dataset_path.startswith("datasets/"):
                    alternative_path = project_root / "datasets" / dataset_path
                    if alternative_path.exists():
                        resolved_path = alternative_path
                        logger.info(f"Dataset not found at {dataset_path}, found at {resolved_path}")
            
            dataset_path = str(resolved_path)
        
        logger.info(f"Loading dataset from {dataset_path}")
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Dataset directory not found: {dataset_path}\n"
                f"Please ensure the dataset exists at the specified path."
            )
        
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
        batch_size = len(examples["instruction"])
        individual_examples = []
        
        for i in range(batch_size):
            example = {
                "instruction": examples["instruction"][i],
                "input": examples["input"][i],
                "output": examples["output"][i],
                "topic": examples.get("topic", [None] * batch_size)[i],
                "upvotes": examples.get("upvotes", [None] * batch_size)[i],
                "question_id": examples.get("question_id", [None] * batch_size)[i],
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
        """Setup training arguments with FSDP support."""
        output_dir = self.config["output_dir"]
        
        use_4bit = self.config.get("use_4bit", True)
        gradient_checkpointing = self.config.get("gradient_checkpointing", True)
        
        # Build TrainingArguments
        training_args_dict = {
            "output_dir": output_dir,
            "num_train_epochs": self.config.get("num_epochs", 3),
            "per_device_train_batch_size": self.config.get("batch_size", 2),
            "per_device_eval_batch_size": self.config.get("eval_batch_size", 2),
            "gradient_accumulation_steps": self.config.get("gradient_accumulation_steps", 8),
            "learning_rate": self.config.get("learning_rate", 1.5e-4),
            "weight_decay": self.config.get("weight_decay", 0.01),
            "warmup_ratio": self.config.get("warmup_ratio", 0.1),
            "lr_scheduler_type": self.config.get("lr_scheduler_type", "cosine"),
            "logging_steps": self.config.get("logging_steps", 10),
            "eval_steps": self.config.get("eval_steps", 100),
            "save_steps": self.config.get("save_steps", 500),
            "eval_strategy": "steps",
            "save_strategy": "steps",
            "save_total_limit": self.config.get("save_total_limit", 3),
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "fp16": self.config.get("fp16", False),
            "bf16": self.config.get("bf16", True),
            "dataloader_pin_memory": self.config.get("dataloader_pin_memory", True),
            "dataloader_num_workers": self.config.get("dataloader_num_workers", 0),
            "gradient_checkpointing": gradient_checkpointing,
            "remove_unused_columns": self.config.get("remove_unused_columns", False),
            "optim": self.config.get("optim", "adamw_torch_fused"),
            "adam_beta1": self.config.get("adam_beta1", 0.9),
            "adam_beta2": self.config.get("adam_beta2", 0.999),
            "adam_epsilon": self.config.get("adam_epsilon", 1e-8),
            "max_grad_norm": self.config.get("max_grad_norm", 1.0),
            "report_to": "wandb" if self.config.get("use_wandb", False) else "none",
            "run_name": self.config.get("run_name", "qwen2.5-14b-counsel-chat-8gpu"),
            "seed": self.config.get("seed", 42),
            "ddp_find_unused_parameters": self.config.get("ddp_find_unused_parameters", False),
            "ddp_bucket_cap_mb": 5,  # Reduce DDP bucket size to save memory during sync (lower = less memory but slower)
        }
        
        # Note: FSDP is incompatible with BitsAndBytes quantized models
        # We use DDP instead, which works with quantized models
        # DDP is the default when using accelerate launch with multiple GPUs
        
        return TrainingArguments(**training_args_dict)
    
    def setup_trainer(self, dataset: Dataset) -> Trainer:
        """Setup the trainer."""
        # Check if tokenized dataset already exists
        dataset_path = self.config["dataset_path"]
        if not os.path.isabs(dataset_path):
            script_dir = Path(__file__).parent  # scripts/training/
            project_root = script_dir.parent.parent  # FYP-LLM/
            resolved_path = project_root / dataset_path
            if not resolved_path.exists() and not dataset_path.startswith("datasets/"):
                alternative_path = project_root / "datasets" / dataset_path
                if alternative_path.exists():
                    resolved_path = alternative_path
            dataset_path = str(resolved_path)
        
        tokenized_dataset_path = dataset_path + "_tokenized"
        
        # Try to load tokenized dataset if it exists
        if os.path.exists(tokenized_dataset_path):
            try:
                logger.info(f"Loading pre-tokenized dataset from {tokenized_dataset_path}...")
                tokenized_dataset = load_from_disk(tokenized_dataset_path)
                logger.info("Pre-tokenized dataset loaded successfully!")
            except Exception as e:
                logger.warning(f"Failed to load pre-tokenized dataset: {e}. Will tokenize from scratch.")
                tokenized_dataset = None
        else:
            tokenized_dataset = None
        
        # Tokenize dataset if not already loaded
        if tokenized_dataset is None:
            logger.info("Tokenizing dataset...")
            tokenized_dataset = dataset.map(
                self.tokenize_function,
                batched=True,
                remove_columns=dataset["train"].column_names,
                load_from_cache_file=True,
            )
            
            # Save tokenized dataset for future use
            try:
                logger.info(f"Saving tokenized dataset to {tokenized_dataset_path}...")
                tokenized_dataset.save_to_disk(tokenized_dataset_path)
                logger.info("Tokenized dataset saved successfully!")
            except Exception as e:
                logger.warning(f"Failed to save tokenized dataset: {e}. Will tokenize each time.")
        
        # Setup data collator
        data_collator = self.setup_data_collator()
        
        # Setup training arguments
        training_args = self.setup_training_arguments()
        
        # Setup callbacks
        callbacks = [MemoryManagementCallback()]
        if self.config.get("early_stopping", True):
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
        
        # Setup trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            callbacks=callbacks,
        )
        
        use_4bit = self.config.get("use_4bit", True)
        gradient_checkpointing = self.config.get("gradient_checkpointing", True)
        if use_4bit:
            if gradient_checkpointing:
                if hasattr(trainer.model, "gradient_checkpointing_enable"):
                    try:
                        trainer.model.gradient_checkpointing_enable()
                    except Exception as e:
                        logger.warning(f"Failed to enable gradient checkpointing on trainer model: {e}")
                if hasattr(trainer.model, "enable_input_require_grads"):
                    try:
                        trainer.model.enable_input_require_grads()
                    except Exception as e:
                        logger.warning(f"Failed to set input requires grad on trainer model: {e}")
                trainer.args.gradient_checkpointing = True
                logger.info("Verified gradient checkpointing remains enabled after Trainer initialization")
            else:
                def disable_gc(model):
                    """Recursively disable gradient checkpointing on model and submodules."""
                    if hasattr(model, 'gradient_checkpointing_disable'):
                        try:
                            model.gradient_checkpointing_disable()
                        except Exception:
                            pass
                    if hasattr(model, 'gradient_checkpointing'):
                        try:
                            model.gradient_checkpointing = False
                        except Exception:
                            pass
                    if hasattr(model, '_gradient_checkpointing_func'):
                        try:
                            model._gradient_checkpointing_func = None
                        except Exception:
                            pass
                    for module in model.modules():
                        if hasattr(module, 'gradient_checkpointing_disable'):
                            try:
                                module.gradient_checkpointing_disable()
                            except Exception:
                                pass
                        if hasattr(module, 'gradient_checkpointing'):
                            try:
                                module.gradient_checkpointing = False
                            except Exception:
                                pass

                disable_gc(trainer.model)
                trainer.args.gradient_checkpointing = False
                logger.info("Verified gradient checkpointing is disabled after Trainer initialization")
        
        return trainer
    
    def train(self) -> None:
        """Main training function."""
        logger.info("Starting multi-GPU training setup...")
        # Get process info from environment (set by accelerate launch)
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        logger.info(f"World size: {world_size}, Rank: {rank}, Local rank: {local_rank}")

        # Guard against unsupported torch.nn.DataParallel with BitsAndBytes quantization
        if (
            self.config.get("use_4bit", True)
            and torch.cuda.is_available()
            and torch.cuda.device_count() > 1
            and world_size == 1
        ):
            raise RuntimeError(
                "Detected multiple visible GPUs but WORLD_SIZE=1. "
                "BitsAndBytes 4-bit models are incompatible with torch.nn.DataParallel. "
                "Please launch via `accelerate launch --num_processes <n>` or restrict "
                "`CUDA_VISIBLE_DEVICES` to a single GPU."
            )
        
        # PRIORITY: Check for existing checkpoints FIRST and update config to match
        # This must happen BEFORE model initialization
        resume_from_checkpoint = None
        output_dir = self.config["output_dir"]
        
        if os.path.exists(output_dir):
            checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-") and not d.endswith(".backup")]
            if checkpoints:
                # Get the latest checkpoint by step number (skip non-numeric suffixes)
                checkpoint_steps = []
                for cp in checkpoints:
                    try:
                        step = int(cp.split("-")[1])
                        checkpoint_steps.append(step)
                    except (ValueError, IndexError):
                        continue
                
                if checkpoint_steps:
                    latest_checkpoint = f"checkpoint-{max(checkpoint_steps)}"
                    checkpoint_path = os.path.join(output_dir, latest_checkpoint)
                    
                    # PRIORITY: Load checkpoint config and update current config to match
                    # This ensures we can always resume from checkpoint
                    checkpoint_config_path = os.path.join(checkpoint_path, "training_config.json")
                    if os.path.exists(checkpoint_config_path):
                        try:
                            import json
                            with open(checkpoint_config_path, 'r') as f:
                                checkpoint_config = json.load(f)
                            
                            # Update config to match checkpoint for successful resumption
                            current_lora_r = self.config.get("lora_r", 16)
                            checkpoint_lora_r = checkpoint_config.get("lora_r", current_lora_r)
                            current_lora_alpha = self.config.get("lora_alpha", 32)
                            checkpoint_lora_alpha = checkpoint_config.get("lora_alpha", current_lora_alpha)
                            current_max_length = self.config.get("max_length", 512)
                            checkpoint_max_length = checkpoint_config.get("max_length", current_max_length)
                            
                            config_updated = False
                            if current_lora_r != checkpoint_lora_r:
                                logger.warning(
                                    f"LoRA rank mismatch: current={current_lora_r}, checkpoint={checkpoint_lora_r}. "
                                    f"Updating config to match checkpoint for resumption."
                                )
                                self.config["lora_r"] = checkpoint_lora_r
                                config_updated = True
                            
                            if current_lora_alpha != checkpoint_lora_alpha:
                                logger.warning(
                                    f"LoRA alpha mismatch: current={current_lora_alpha}, checkpoint={checkpoint_lora_alpha}. "
                                    f"Updating config to match checkpoint for resumption."
                                )
                                self.config["lora_alpha"] = checkpoint_lora_alpha
                                config_updated = True
                            
                            if current_max_length != checkpoint_max_length:
                                logger.info(
                                    f"Max length differs: current={current_max_length}, checkpoint={checkpoint_max_length}. "
                                    f"Updating config to match checkpoint for resumption."
                                )
                                self.config["max_length"] = checkpoint_max_length
                                config_updated = True
                            
                            if config_updated:
                                logger.info("Config updated to match checkpoint. Will initialize model with correct config.")
                                
                        except Exception as e:
                            logger.warning(f"Could not read checkpoint config: {e}. Proceeding with current config.")
                    
                    # Always attempt to resume if checkpoint exists (prioritize resumption)
                    resume_from_checkpoint = checkpoint_path
                    logger.info(f"Found existing checkpoint: {resume_from_checkpoint}")
                    logger.info("PRIORITY: Will resume training from checkpoint...")
        
        # Setup components (now with config matching checkpoint if available)
        self.setup_tokenizer()
        # Load base model normally - Trainer will handle checkpoint loading
        # (Loading from checkpoint here causes issues with adapter loading)
        self.setup_model()
        
        # Clean up memory after model loading
        self.cleanup_memory()
        
        # If we loaded from checkpoint, model is already loaded - no need to clear memory for reload
        # (We loaded directly from checkpoint in setup_model() to avoid double-loading)
        if resume_from_checkpoint and torch.cuda.is_available():
            logger.info("Model already loaded from checkpoint - clearing memory before Trainer initialization...")
            for i in range(3):
                torch.cuda.empty_cache()
                gc.collect()
            torch.cuda.synchronize()
        
        # Log memory usage after model loading
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            allocated = torch.cuda.memory_allocated(local_rank) / 1024**3
            reserved = torch.cuda.memory_reserved(local_rank) / 1024**3
            total = torch.cuda.get_device_properties(local_rank).total_memory / 1024**3
            logger.info(f"GPU {local_rank} memory after model load: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved, {total:.2f} GB total")
            free_memory = total - reserved
            if resume_from_checkpoint and free_memory < 2.0:
                logger.warning(f"Low free memory ({free_memory:.2f} GB). Resumption may fail with OOM, but attempting anyway.")
        
        # Load dataset
        dataset = self.load_dataset()
        
        # Setup trainer
        self.trainer = self.setup_trainer(dataset)
        
        # Initialize wandb if enabled (only on main process)
        rank = int(os.environ.get("RANK", 0))
        if self.config.get("use_wandb", False) and rank == 0:
            wandb.init(
                project=self.config.get("wandb_project", "qwen2.5-14b-counsel-chat"),
                name=self.config.get("run_name", "qwen2.5-14b-counsel-chat-8gpu"),
                config=self.config
            )
        
        # Start training
        logger.info("Starting training...")
        logger.info(f"Training on {torch.cuda.device_count()} GPU(s)")
        logger.info(f"Training steps per epoch: {len(self.trainer.get_train_dataloader())}")
        logger.info(f"Total training steps: {self.trainer.args.max_steps if self.trainer.args.max_steps > 0 else len(self.trainer.get_train_dataloader()) * self.trainer.args.num_train_epochs}")
        
        if resume_from_checkpoint:
            logger.info(f"PRIORITY: Resuming from checkpoint: {resume_from_checkpoint}")
            # CRITICAL: Clear memory aggressively before Trainer reloads from checkpoint
            # Trainer will reload the model from checkpoint, which temporarily doubles memory usage
            logger.info("Clearing memory aggressively before Trainer loads checkpoint (will temporarily double-load)...")
            # Clear memory but keep model reference (Trainer needs it)
            for i in range(10):
                torch.cuda.empty_cache()
                gc.collect()
            torch.cuda.synchronize()
            logger.info("Memory cleared. Trainer will reload model from checkpoint.")
        
        # Clear GPU cache aggressively before training to reduce OOM risk during DDP initialization
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            
            # Multiple cache clears to ensure maximum memory is freed
            for i in range(10):
                torch.cuda.empty_cache()
                gc.collect()
            
            # Force synchronization to ensure all operations are complete
            torch.cuda.synchronize()
            
            # Try to reduce reserved memory by resetting peak stats (if available)
            try:
                torch.cuda.reset_peak_memory_stats(local_rank)
            except:
                pass
            
            # Final aggressive clear
            torch.cuda.empty_cache()
            gc.collect()
            
            # Log memory status before training
            allocated = torch.cuda.memory_allocated(local_rank) / 1024**3
            reserved = torch.cuda.memory_reserved(local_rank) / 1024**3
            total = torch.cuda.get_device_properties(local_rank).total_memory / 1024**3
            free_memory = total - reserved
            logger.info(f"GPU {local_rank} memory before training: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved, {free_memory:.2f} GB free")
            logger.info("Cleared CUDA cache aggressively before training start")
            
            if free_memory < 2.5:
                logger.warning(f"WARNING: Very low free memory ({free_memory:.2f} GB). DDP initialization needs ~2 GB. This may fail with OOM.")
        
        try:
            self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            # Try to save emergency checkpoint
            try:
                if rank == 0:
                    emergency_path = os.path.join(output_dir, "emergency_checkpoint")
                    logger.info(f"Attempting to save emergency checkpoint to {emergency_path}")
                    self.trainer.save_model(emergency_path)
            except Exception as save_error:
                logger.error(f"Failed to save emergency checkpoint: {save_error}")
            raise
        
        # Save final model (only on main process)
        rank = int(os.environ.get("RANK", 0))
        if rank == 0:
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
    # Setup logging based on rank (must be done early, before other logging)
    setup_logging_for_rank()
    
    # Set HuggingFace cache directories
    base_dir = os.environ.get("HF_BASE_DIR", "/research/d7/fyp25/yyyu2")
    
    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = f"{base_dir}/.cache/huggingface"
    if "TRANSFORMERS_CACHE" not in os.environ:
        os.environ["TRANSFORMERS_CACHE"] = f"{base_dir}/.cache/huggingface/transformers"
    if "HF_DATASETS_CACHE" not in os.environ:
        os.environ["HF_DATASETS_CACHE"] = f"{base_dir}/.cache/huggingface/datasets"
    if "HF_HUB_CACHE" not in os.environ:
        os.environ["HF_HUB_CACHE"] = f"{base_dir}/.cache/huggingface/hub"
    if "XET_CACHE" not in os.environ:
        os.environ["XET_CACHE"] = f"{base_dir}/.cache/huggingface/xet"
    
    # Create cache directories
    for cache_dir in [
        os.environ["HF_HOME"],
        os.environ["TRANSFORMERS_CACHE"],
        os.environ["HF_DATASETS_CACHE"],
        os.environ["HF_HUB_CACHE"],
        os.environ["XET_CACHE"],
    ]:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"HF_HOME: {os.environ['HF_HOME']}")
    logger.info(f"TRANSFORMERS_CACHE: {os.environ['TRANSFORMERS_CACHE']}")
    
    parser = argparse.ArgumentParser(description="Multi-GPU training for Qwen2.5-14B on Counsel Chat dataset")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/config_14b_8gpu.json",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-14B-Instruct",
        help="Model name or path"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="datasets/all_mental_health_combined",
        help="Path to processed dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/qwen2.5-14b-counsel-chat-8gpu",
        help="Output directory for trained model"
    )
    
    args = parser.parse_args()
    
    # Create default config optimized for 14B on 8 GPUs
    config = {
        "model_name": args.model_name,
        "dataset_path": args.dataset_path,
        "output_dir": args.output_dir,
        "batch_size": 1,  # Per device batch size (reduced for memory)
        "num_epochs": 3,
        "learning_rate": 1.5e-4,
        "use_wandb": False,
        "use_4bit": True,
        "lora_r": 8,  # Reduced from 16 to save memory
        "lora_alpha": 16,  # Reduced from 32 to save memory
        "lora_dropout": 0.1,
        "max_length": 512,  # Reduced from 1024 to save memory
        "gradient_accumulation_steps": 16,  # Increased to maintain effective batch size = 1 * 16 * 8 = 128
        "eval_batch_size": 1,
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
        "run_name": "qwen2.5-14b-counsel-chat-8gpu",
        "wandb_project": "qwen2.5-14b-counsel-chat",
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": True,
        "gradient_checkpointing": False,  # Disabled for BitsAndBytes compatibility
        "ddp_find_unused_parameters": False,
        "optim": "adamw_torch_fused",
        # Note: FSDP is incompatible with BitsAndBytes quantized models
        # DDP will be used automatically by accelerate launch
    }
    
    # Load config file if it exists
    config_path = args.config
    if not config_path.endswith('.json'):
        config_path = config_path + '.json'
    
    if os.path.exists(config_path):
        file_config = load_config(config_path)
        config.update(file_config)
    
    # Override with command line arguments
    if args.model_name != "Qwen/Qwen2.5-14B-Instruct":
        config["model_name"] = args.model_name
    if args.dataset_path != "datasets/all_mental_health_combined":
        config["dataset_path"] = args.dataset_path
    if args.output_dir != "models/qwen2.5-14b-counsel-chat-8gpu":
        config["output_dir"] = args.output_dir
    
    # Create output directory
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(os.path.join(config["output_dir"], "training_config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize and start training
    trainer = MultiGPUTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()

