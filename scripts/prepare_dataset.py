from datasets import load_dataset

# Load from local path or convert JSONL to Dataset
dataset = load_dataset("json", data_files="counsel_chat.jsonl", split="train")

# Split into train/validation (90/10)
dataset = dataset.train_test_split(test_size=0.1)
dataset.save_to_disk("counsel_chat_split")

dataset = dataset.train_test_split(test_size=0.1, seed=42)
print(f"Training set size: {len(dataset['train'])}")
print(f"Validation set size: {len(dataset['test'])}")

# Save splits for reproducibility
dataset.save_to_disk("counsel_chat_split")