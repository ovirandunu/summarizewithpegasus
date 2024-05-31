import os
import torch
import nltk
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    DataCollatorForSeq2Seq, 
    TrainingArguments, 
    Trainer, 
    pipeline,
)
from datasets import load_dataset
import logging
import evaluate

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler('training.log'),
                        logging.StreamHandler()
                    ])

# Set training arguments
logging.info("Setting up training arguments")
trainer_args = TrainingArguments(
    output_dir='/content/drive/MyDrive/model/checkpoints',
    num_train_epochs=1,
    warmup_steps=500,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    weight_decay=0.01,
    logging_steps=10,
    evaluation_strategy='steps',
    eval_steps=500,
    save_steps=500,
    gradient_accumulation_steps=16
)

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Device set to {device}")

# Load the Pegasus model and tokenizer from the checkpoint
model_ckpt = "/path/to/your/checkpoint"  # Update this to your checkpoint path
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)
logging.info("Model and tokenizer loaded from checkpoint")

# Load medical conversations dataset
data_files = {"train": "path/to/train.json", "test": "path/to/test.json", "validation": "path/to/validation.json"}
dataset = load_dataset("json", data_files=data_files, split={"train": "train", "test": "test", "validation": "validation"})
logging.info(f"Dataset splits: {[len(dataset[split]) for split in dataset]}")
logging.info(f"Dataset features: {dataset['train'].column_names}")

# Convert examples to features
def convert_examples_to_features(example_batch):
    input_encodings = tokenizer(example_batch['dialogue'], max_length=1024, truncation=True)
    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(example_batch['soap'], max_length=128, truncation=True)
    target_encodings['labels'] = target_encodings['input_ids'].copy()
    target_encodings['input_ids'] = [[tokenizer.pad_token_id] + ids[:-1] for ids in target_encodings['input_ids']]
    return {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'labels': target_encodings['input_ids']
    }

dataset_preprocessed = dataset.map(convert_examples_to_features, batched=True)

# Initialize Trainer
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
trainer = Trainer(
    model=model,
    args=trainer_args,
    data_collator=data_collator,
    train_dataset=dataset_preprocessed["train"],
    eval_dataset=dataset_preprocessed["validation"],
    tokenizer=tokenizer
)

# Start training
logging.info("Starting training")
trainer.train()
logging.info("Training complete")

# Save model and tokenizer
model_dir = '/content/drive/MyDrive/model/fine_tuned'
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
logging.info("Model and tokenizer saved")

# Helper function to split data into batch-sized chunks
def generate_batch_sized_chunks(list_of_elements, batch_size):
    """Yield successive batch-sized chunks from list_of_elements."""
    for i in range(0, len(list_of_elements), batch_size):
        yield list_of_elements[i: i + batch_size]

# Evaluate using the ROUGE metric
rouge_metric = evaluate.load('rouge', trust_remote_code=True)
def calculate_metric_on_test_ds(dataset, metric, model, tokenizer, batch_size=8, device='cuda', column_text="dialogue", column_summary="soap"):
    model.eval()
    total_metric = []
    article_batches = list(generate_batch_sized_chunks(dataset[column_text], batch_size))
    target_batches = list(generate_batch_sized_chunks(dataset[column_summary], batch_size))

    for article_batch, target_batch in tqdm(zip(article_batches, target_batches), total=len(article_batches)):
        inputs = tokenizer(article_batch, max_length=1024, truncation=True, padding="max_length", return_tensors="pt").to(device)
        
        with torch.no_grad():
            summaries = model.generate(input_ids=inputs["input_ids"].to(device),
                                       attention_mask=inputs["attention_mask"].to(device),
                                       length_penalty=0.8, num_beams=8, max_length=128)

        decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True) for s in summaries]

        metric.add_batch(predictions=decoded_summaries, references=target_batch)

    score = metric.compute()
    return score

test_score = calculate_metric_on_test_ds(
    dataset['test'], rouge_metric, model, tokenizer, batch_size=2, 
    column_text='dialogue', column_summary='soap'
)
rouge_dict = {rn: test_score[rn].mid.fmeasure for rn in ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']}
dataframe_rouge = pd.DataFrame([rouge_dict], index=['Post-Training'])
logging.info("ROUGE scores after training:")
logging.info(dataframe_rouge)
