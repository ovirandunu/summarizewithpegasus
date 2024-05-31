import os
import torch
import nltk
import requests
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

# Define API utility functions
def fetch_dataset_rows():
    try:
        url = "https://datasets-server.huggingface.co/rows?dataset=omi-health%2Fmedical-dialogue-to-soap-summary&config=default&split=train&offset=0&length=100"
        response = requests.get(url)
        response.raise_for_status()  # Will raise an HTTPError for bad responses
        return response.json()
    except requests.RequestException as e:
        logging.error(f"Failed to fetch dataset rows: {e}")
        return None


def list_dataset_splits():
    """Lists available splits of the dataset."""
    url = "https://datasets-server.huggingface.co/splits?dataset=omi-health%2Fmedical-dialogue-to-soap-summary"
    response = requests.get(url)
    return response.json() if response.status_code == 200 else None

def download_parquet_files():
    """Downloads Parquet files for the dataset."""
    url = "https://huggingface.co/api/datasets/omi-health/medical-dialogue-to-soap-summary/parquet/default/train"
    response = requests.get(url)
    if response.status_code == 200:
        with open('dataset.parquet', 'wb') as f:
            f.write(response.content)
        logging.info("Parquet file downloaded successfully.")

# Training configurations
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

# Load the Pegasus model and tokenizer
model_ckpt = "/path/to/your/checkpoint"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)
logging.info("Model and tokenizer loaded from checkpoint")

# Helper function to split data into batch-sized chunks
def generate_batch_sized_chunks(list_of_elements, batch_size):
    """Yield successive batch-sized chunks from list_of_elements."""
    for i in range(0, len(list_of_elements), batch_size):
        yield list_of_elements[i: i + batch_size]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)


def calculate_metric_on_test_ds(dataset, rouge_metric, model, tokenizer, batch_size=2, column_text='dialogue', column_summary='soap'):
    model.eval()
    article_batches = list(generate_batch_sized_chunks(dataset[column_text], batch_size))
    target_batches = list(generate_batch_sized_chunks(dataset[column_summary], batch_size))

    for article_batch, target_batch in tqdm(zip(article_batches, target_batches), total=len(article_batches)):
        inputs = tokenizer(article_batch, max_length=1024, truncation=True, padding="max_length", return_tensors="pt").to(device)
        
        with torch.no_grad():
            summaries = model.generate(input_ids=inputs["input_ids"].to(device),
                                       attention_mask=inputs["attention_mask"].to(device),
                                       length_penalty=0.8, num_beams=8, max_length=128)

        decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True) for s in summaries]
        decoded_summaries = [d.replace("", " ") for d in decoded_summaries]

        metric.add_batch(predictions=decoded_summaries, references=target_batch)
        del inputs
        torch.cuda.empty_cache() if device == 'cuda' else None

    score = metric.compute()
    return score


# API Calls
dataset_rows = fetch_dataset_rows()
dataset_splits = list_dataset_splits()
download_parquet_files()

# Load medical conversations dataset
data_files = {"train": "path/to/train.json", "test": "path/to/test.json", "validation": "path/to/validation.json"}
dataset = load_dataset("json", data_files=data_files, split={"train": "train", "test": "test", "validation": "validation"})
logging.info(f"Dataset splits: {[len(dataset[split]) for split in dataset]}")
logging.info(f"Dataset features: {dataset['train'].column_names}")

# Convert examples to features and prepare for training
def convert_examples_to_features(example_batch):
    input_encodings = tokenizer(example_batch['dialogue'], max_length=1024, truncation=True)
    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(example_batch['soap'], max_length=128, truncation=True)
    target_encodings['labels'] = target_encodings['input_ids'].copy()
    target_encodings['input_ids'] = [[tokenizer.pad_token_id] + ids[:-1] for ids in target_encodings['input_ids']]
    return {'input_ids': input_encodings['input_ids'], 'attention_mask': input_encodings['attention_mask'], 'labels': target_encodings['input_ids']}
dataset_preprocessed = dataset.map(convert_examples_to_features, batched=True)

# Initialize Trainer and start training
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
trainer = Trainer(model=model, args=trainer_args, data_collator=data_collator, train_dataset=dataset_preprocessed["train"], eval_dataset=dataset_preprocessed["validation"], tokenizer=tokenizer)
logging.info("Starting training")
trainer.train()
logging.info("Training complete")

# Save model and tokenizer
model_dir = '/content/drive/MyDrive/model/fine_tuned'
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
logging.info("Model and tokenizer saved")

# Evaluate using the ROUGE metric and calculate scores
rouge_metric = evaluate.load('rouge', trust_remote_code=True)
test_score = calculate_metric_on_test_ds(dataset['test'], rouge_metric, model, tokenizer, batch_size=2, column_text='dialogue', column_summary='soap')
rouge_dict = {rn: test_score[rn].mid.fmeasure for rn in ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']}
dataframe_rouge = pd.DataFrame([rouge_dict], index=['Post-Training'])
logging.info("ROUGE scores after training:")
logging.info(dataframe_rouge)
# Token length histograms
dialogue_token_len = [len(tokenizer.encode(s)) for s in dataset['train']['dialogue']]
summary_token_len = [len(tokenizer.encode(s)) for s in dataset['train']['summary']]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(dialogue_token_len, bins=20, color='C0', edgecolor='C0')
axes[0].set_title("Dialogue Token Length")
axes[0].set_xlabel("Length")
axes[0].set_ylabel("Count")
axes[1].hist(summary_token_len, bins=20, color='C0', edgecolor='C0')
axes[1].set_title("Summary Token Length")
axes[1].set_xlabel("Length")
plt.tight_layout()
plt.show()



dataset_samsum_pt = dataset.map(convert_examples_to_features, batched=True)
seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)



# Summarization with fine-tuned model
sample_text = dataset["test"][0]["dialogue"]
reference = dataset["test"][0]["soap"]

gen_kwargs = {"length_penalty": 0.8, "num_beams": 8, "max_length": 128}

try:
    pipe = pipeline("summarization", model=model, tokenizer=tokenizer)  # Directly use the loaded model
    model_summary = pipe(sample_text, **gen_kwargs)[0]["summary_text"]
    logging.info("Dialogue:")
    logging.info(sample_text)
    logging.info("\nReference Summary:")
    logging.info(reference)
    logging.info("\nModel Summary:")
    logging.info(model_summary)
except Exception as e:
    logging.error(f"Error during summarization: {e}")


# Evaluate using ROUGE metric
rouge_metric = evaluate.load('rouge', trust_remote_code=True)
test_score = calculate_metric_on_test_ds(dataset['test'], rouge_metric, model, tokenizer, batch_size=2, column_text='dialogue', column_summary='soap')
rouge_dict = {rn: test_score[rn].mid.fmeasure for rn in ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']}
dataframe_rouge = pd.DataFrame([rouge_dict], index=['Post-Training'])
logging.info("ROUGE scores after training:\n" + str(dataframe_rouge))

# Plot token length histograms
dialogue_token_len = [len(tokenizer.encode(s)) for s in dataset['train']['dialogue']]
summary_token_len = [len(tokenizer.encode(s)) for s in dataset['train']['soap']]
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(dialogue_token_len, bins=20, color='C0', edgecolor='C0')
axes[0].set_title("Dialogue Token Length")
axes[0].set_xlabel("Length")
axes[0].set_ylabel("Count")
axes[1].hist(summary_token_len, bins=20, color='C0', edgecolor='C0')
axes[1].set_title("Summary Token Length")
axes[1].set_xlabel("Length")
plt.tight_layout()
plt.show()


