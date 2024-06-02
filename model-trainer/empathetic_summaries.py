# importing all the necessary models
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
    set_seed
)
from datasets import load_dataset, load_metric
import logging
from utils import calculate_metric_on_test_ds

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler('trainer.log'),
                        logging.StreamHandler()
                    ])

# download nltk punkt for tokenization
nltk.download("punkt")

# Check GPU availability and assign device
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Device set to {device}")

# Load the relevant model. 
# Checkpoints available for this pipeline: 
# google/pegasus-cnn_dailymail, ~/tm/tmgp/model-trainer/checkpoints/pegasus-samsum-model-2
model_ckpt = "google/pegasus-cnn_dailymail"
logging.info(f"Loading model and tokenizer from checkpoint {model_ckpt}")
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)
logging.info(f"Model loaded: {model_pegasus}")

# Load dataset
logging.info("Loading empathetic_dialogues_summary dataset")
dataset_empathic = load_dataset("jtatman/empathetic_dialogues_summary")

# Splitting the dataset into train, validation, and test sets using the slicing API
train_valid_test_split = {
    'train': 'train[:80%]',
    'validation': 'train[80%:90%]',
    'test': 'train[90%:]'
}
dataset_empathic = load_dataset("jtatman/empathetic_dialogues_summary", split=train_valid_test_split)

logging.info(f"Split lengths: {[len(dataset_empathic[split]) for split in dataset_empathic]}")
logging.info(f"Features: {dataset_empathic['train'].column_names}")

# Logging a sample prompt and summary
logging.info("\nUtterance:")
logging.info(dataset_empathic["test"][1]["prompt"])
logging.info("\nSummary:")
logging.info(dataset_empathic["test"][1]["summary"])

# Summarization pipeline
pipe = pipeline('summarization', model=model_ckpt)
pipe_out = pipe(dataset_empathic['test'][0]['prompt'])
logging.info(pipe_out)
logging.info(pipe_out[0]['summary_text'].replace(" .", ".\n"))

# Calculate ROUGE scores
rouge_metric = load_metric('rouge')
score = calculate_metric_on_test_ds(dataset_empathic['test'], rouge_metric, model_pegasus, tokenizer, column_text='prompt', column_summary='summary', batch_size=8)

rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
rouge_dict = dict((rn, score[rn].mid.fmeasure) for rn in rouge_names)
dataframe_rouge = pd.DataFrame(rouge_dict, index=['pegasus'])
logging.info(dataframe_rouge)


# Convert examples to features
def convert_examples_to_features(example_batch):
    input_encodings = tokenizer(example_batch['prompt'], max_length=1024, truncation=True)

    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(example_batch['summary'], max_length=128, truncation=True)

    target_encodings['labels'] = target_encodings['input_ids'].copy()
    target_encodings['input_ids'] = [[tokenizer.pad_token_id] + ids[:-1] for ids in target_encodings['input_ids']]

    return {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'labels': target_encodings['input_ids']
    }

# Tokenize and encode the dataset
dataset_empathic_pt = dataset_empathic.map(convert_examples_to_features, batched=True)
seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)

# Training arguments
logging.info("Setting up training arguments")
trainer_args = TrainingArguments(
    # directory for saving checkpoints
    output_dir=os.path.expanduser('~/tm/tmgp/model-trainer/empathetic'),
    # total number of training epochs
    num_train_epochs=6,
    # number of warmup steps for learning rate scheduler
    warmup_steps=500,
    # batch size for training
    per_device_train_batch_size=4,
    # batch size for evaluation
    per_device_eval_batch_size=4,
    # decay rate for learning rate
    weight_decay=0.01,
    # evaluate at every 500 steps
    eval_steps=500,
    evaluation_strategy='steps',
    # save at every epoch
    save_strategy='epoch',
    # optimize gpu memory usage with gradient accumulation
    gradient_accumulation_steps=16,
)
logging.info(f"Training arguments: {trainer_args}")

# Trainer initialization
logging.info("Initialising trainer")
trainer = Trainer(model=model_pegasus, args=trainer_args,
                  tokenizer=tokenizer, data_collator=seq2seq_data_collator,
                  train_dataset=dataset_empathic_pt["train"],
                  eval_dataset=dataset_empathic_pt["validation"])

logging.info("Starting training on the empathic dataset")
trainer.train()
logging.info("Training complete on the empathic dataset")

# Calculate ROUGE scores after training
score = calculate_metric_on_test_ds(
    dataset_empathic['test'], rouge_metric, trainer.model, tokenizer, batch_size=2, column_text='prompt', column_summary='summary'
)
rouge_dict = dict((rn, score[rn].mid.fmeasure) for rn in rouge_names)
logging.info(pd.DataFrame(rouge_dict, index=[f'pegasus']))

# Save model and tokenizer
try:
    model_pegasus.save_pretrained(os.path.expanduser('~/tm/empathetic/tmgp/model-trainer/empathetic/model-2/model-2'))
except Exception as e:
    logging.error(f"Error saving model stage 1: {e}")

# Save model and tokenizer
try:
    trainer.model.save_pretrained(os.path.expanduser('~/tm/empathetic/tmgp/model-trainer/empathetic/model-1'))
    logging.info("Model saved successfully after empathic finetuning.")
except Exception as e:
    logging.error(f"Error saving model: {e}")

try:
    tokenizer.save_pretrained(os.path.expanduser('~/tm/empathetic/tmgp/model-trainer/empathetic/tokenizer'))
    logging.info("Tokenizer saved successfully after empathic finetuning.")
except Exception as e:
    logging.error(f"Error saving tokenizer: {e}")

# Summarization pipeline with fine-tuned model
sample_text = dataset_empathic["test"][0]["prompt"]
reference = dataset_empathic["test"][0]["summary"]
gen_kwargs = {"length_penalty": 0.8, "num_beams": 8, "max_length": 128}

try:
    pipe = pipeline("summarization", model=os.path.expanduser('~/tm/empathetic/tmgp/model-trainer/empathetic/model-1'), tokenizer=tokenizer)
    model_summary = pipe(sample_text, **gen_kwargs)[0]["summary_text"]
    logging.info("Utterance:")
    logging.info(sample_text)
    logging.info("\nReference Summary:")
    logging.info(reference)
    logging.info("\nModel Summary:")
    logging.info(model_summary)
except Exception as e:
    logging.error(f"Error during summarization: {e}")
