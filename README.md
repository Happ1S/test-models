# test-models

## code for LoRa 
```
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments
from peft import get_peft_model, LoraConfig
import glob

# Custom dataset class
class HistoryDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length=512, summary_max_length=128):
        self.text_files = glob.glob(os.path.join(data_dir, "*.txt"))
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.summary_max_length = summary_max_length
        self.texts, self.summaries = self.load_data()

    def load_data(self):
        texts = []
        summaries = []
        for file_path in self.text_files:
            if "_short" not in file_path:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                summary_path = file_path.replace(".txt", "_short.txt")
                if os.path.exists(summary_path):
                    with open(summary_path, 'r', encoding='utf-8') as file:
                        summary = file.read()
                    texts.append(text)
                    summaries.append(summary)
        return texts, summaries

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        summary = self.summaries[idx]

        inputs = self.tokenizer(text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(summary, max_length=self.summary_max_length, truncation=True, padding='max_length', return_tensors='pt')

        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()
        labels = labels["input_ids"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

def main():
    # Initialize the process group
    dist.init_process_group("nccl")
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)

    # Load the model and tokenizer
    model_name = "LLaMA-8B"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Apply LoRa to the model
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.1,
    )
    model = get_peft_model(model, lora_config).cuda()

    # Load the dataset
    data_dir = '/path/to/your/dataset'
    dataset = HistoryDataset(data_dir, tokenizer)
    
    # Split the dataset into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Data loaders with distributed sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=4, sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, num_workers=4)

    # Wrap the model with DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

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
        fp16=True,
        dataloader_num_workers=4,
    )

    # Define a custom training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)

    for epoch in range(training_args.num_train_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].cuda(non_blocking=True)
            attention_mask = batch['attention_mask'].cuda(non_blocking=True)
            labels = batch['labels'].cuda(non_blocking=True)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].cuda(non_blocking=True)
                attention_mask = batch['attention_mask'].cuda(non_blocking=True)
                labels = batch['labels'].cuda(non_blocking=True)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
        
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{training_args.num_train_epochs}, Validation Loss: {val_loss}")

    # Save the final model
    if local_rank == 0:
        model.module.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)

    # Clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```
