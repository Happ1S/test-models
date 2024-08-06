# test-models

## code for LoRa 

import torch
import horovod.torch as hvd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import get_peft_model, LoraConfig

# Initialize Horovod
hvd.init()

# Pin GPU to local rank
torch.cuda.set_device(hvd.local_rank())

# Load the model and tokenizer
model_name = "LLaMA-8B"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Apply LoRa to the model
lora_config = LoraConfig(
    r=16, # Number of LoRa ranks
    lora_alpha=32, # LoRa alpha
    target_modules=["q", "v"], # Modules to apply LoRa
    lora_dropout=0.1, # Dropout rate for LoRa
)
model = get_peft_model(model, lora_config)

# Load the dataset
dataset = load_dataset('path_to_your_russian_history_dataset')

# Preprocess the dataset
def preprocess_function(examples):
    inputs = examples["text"]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=3,
    num_train_epochs=3,
    fp16=True, # Enable mixed precision training
    dataloader_num_workers=4,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# Start training
trainer.train()
