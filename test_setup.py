#!/usr/bin/env python3
"""
Test script to verify the training setup works correctly.
This script tests dataset preparation and model loading without full training.
"""

import os
import sys
import json
import torch
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import transformers
        print(f"✓ transformers {transformers.__version__}")
    except ImportError as e:
        print(f"✗ transformers: {e}")
        return False
    
    try:
        import datasets
        print(f"✓ datasets {datasets.__version__}")
    except ImportError as e:
        print(f"✗ datasets: {e}")
        return False
    
    try:
        import peft
        print(f"✓ peft {peft.__version__}")
    except ImportError as e:
        print(f"✗ peft: {e}")
        return False
    
    try:
        import accelerate
        print(f"✓ accelerate {accelerate.__version__}")
    except ImportError as e:
        print(f"✗ accelerate: {e}")
        return False
    
    print("✓ All imports successful!")
    return True


def test_model_loading():
    """Test loading the Qwen2.5-0.5B model."""
    print("\nTesting model loading...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        print(f"Loading tokenizer from {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        print(f"✓ Tokenizer loaded. Vocab size: {len(tokenizer)}")
        
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        print(f"✓ Model loaded. Parameters: {model.num_parameters():,}")
        
        # Test a simple generation
        test_prompt = "Hello, how are you?"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        # Move inputs to the same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✓ Test generation successful: {response}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False


def test_dataset_preparation():
    """Test dataset preparation with a small sample."""
    print("\nTesting dataset preparation...")
    
    try:
        # Check if dataset file exists
        csv_path = "counsel-chat/data/20200325_counsel_chat.csv"
        if not os.path.exists(csv_path):
            print(f"✗ Dataset file not found: {csv_path}")
            return False
        
        print(f"✓ Dataset file found: {csv_path}")
        
        # Test with a small sample
        import pandas as pd
        df = pd.read_csv(csv_path, nrows=10)  # Load only first 10 rows
        
        print(f"✓ Loaded {len(df)} sample rows")
        print(f"✓ Columns: {list(df.columns)}")
        
        # Check required columns
        required_cols = ['questionText', 'answerText']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"✗ Missing required columns: {missing_cols}")
            return False
        
        print("✓ All required columns present")
        
        # Test data cleaning
        from prepare_counsel_dataset import clean_text, create_instruction_prompt
        
        sample_question = df.iloc[0]['questionText']
        sample_answer = df.iloc[0]['answerText']
        
        cleaned_question = clean_text(sample_question)
        cleaned_answer = clean_text(sample_answer)
        
        print(f"✓ Data cleaning successful")
        print(f"  Original question length: {len(sample_question)}")
        print(f"  Cleaned question length: {len(cleaned_question)}")
        
        # Test prompt creation
        prompt = create_instruction_prompt(cleaned_question)
        print(f"✓ Prompt creation successful (length: {len(prompt)})")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset preparation failed: {e}")
        return False


def test_config():
    """Test configuration file."""
    print("\nTesting configuration...")
    
    config_path = "config.json"
    if not os.path.exists(config_path):
        print(f"✗ Config file not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print("✓ Config file loaded successfully")
        
        # Check required keys
        required_keys = ['model_name', 'dataset_path', 'output_dir', 'batch_size']
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            print(f"✗ Missing required config keys: {missing_keys}")
            return False
        
        print("✓ All required config keys present")
        print(f"  Model: {config['model_name']}")
        print(f"  Batch size: {config['batch_size']}")
        print(f"  Epochs: {config['num_epochs']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("FYP LLM Training Setup Test")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Config Test", test_config),
        ("Dataset Preparation Test", test_dataset_preparation),
        ("Model Loading Test", test_model_loading),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * 40)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your setup is ready for training.")
        print("\nNext steps:")
        print("1. Run: python prepare_counsel_dataset.py --max_samples 100")
        print("2. Run: python train_qwen_counsel.py")
        print("3. Run: python inference.py --interactive")
    else:
        print(f"\n❌ {len(results) - passed} tests failed. Please fix the issues before training.")
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
