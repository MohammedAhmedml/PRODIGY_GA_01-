from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset

# Load pretrained GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Load text dataset
dataset = load_dataset("text", data_files={"train": "data.txt"})

# Tokenization function
def tokenize_function(examples):
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        padding='max_length',
        max_length=128,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

# Tokenize dataset
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

# Training configuration
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    logging_steps=10,
    save_strategy="no"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"]
)

# Train the model
trainer.train()

# Save the trained model
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")