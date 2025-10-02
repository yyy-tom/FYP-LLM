#!/usr/bin/env python3
"""
Colab-optimized training script for fine-tuning Qwen2.5 on Counsel Chat dataset.
This version is optimized for Google Colab's memory and time constraints.
"""

import os
import json
import logging
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)
from datasets import load_from_disk
import gc

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_model_and_tokenizer(model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    """Setup model and tokenizer with memory optimizations."""
    
    # Clear cache
    torch.cuda.empty_cache()
    gc.collect()
    
    logger.info(f"Loading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Loading model from {model_name}")
    
    # Use 4-bit quantization for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA with smaller parameters for memory efficiency
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # Smaller rank
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Fewer modules
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def format_prompt(example, tokenizer):
    """Format training example into prompt."""
    instruction = example["instruction"]
    output = example["output"]
    
    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": output}
    ]
    
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize dataset with memory optimization."""
    batch_size = len(examples["instruction"])
    prompts = []
    
    for i in range(batch_size):
        example = {
            "instruction": examples["instruction"][i],
            "output": examples["output"][i],
        }
        prompt = format_prompt(example, tokenizer)
        prompts.append(prompt)
    
    tokenized = tokenizer(
        prompts,
        truncation=True,
        padding=False,
        max_length=max_length,
        return_tensors=None,
    )
    
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

def main():
    """Main training function optimized for Colab."""
    
    # Configuration optimized for Colab
    config = {
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "dataset_path": "counsel_chat_processed",
        "output_dir": "qwen2.5-counsel-chat-finetuned",
        "batch_size": 1,
        "num_epochs": 1,
        "learning_rate": 5e-4,
        "max_length": 512,
        "gradient_accumulation_steps": 4,
        "save_steps": 100,
        "eval_steps": 50,
        "logging_steps": 5,
    }
    
    logger.info("Starting Colab-optimized training...")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config["model_name"])
    
    # Load dataset
    logger.info(f"Loading dataset from {config['dataset_path']}")
    dataset = load_from_disk(config["dataset_path"])
    logger.info(f"Dataset loaded. Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}")
    
    # Tokenize dataset
    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, config["max_length"]),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    
    # Setup data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=config["logging_steps"],
        eval_steps=config["eval_steps"],
        save_steps=config["save_steps"],
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to="none",
        run_name="qwen2.5-counsel-chat-colab",
        seed=42,
    )
    
    # Setup trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save model
    logger.info("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(config["output_dir"])
    
    # Save config
    with open(os.path.join(config["output_dir"], "training_config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info("Training completed successfully!")
    
    # Clear memory
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()
