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
    set_seed,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from datasets import load_dataset, load_metric
from transformers.trainer_callback import TrainerCallback
import logging
from utils import calculate_metric_on_test_ds


# Set up logging
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

# Load the Pegasus model and its tokenizer from HuggingFace. (In the second stage, we load the model from the checkpoint instead.)
model_ckpt = "google/pegasus-cnn_dailymail"
logging.info(f"Loading model and tokenizer from checkpoint {model_ckpt}")
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)
logging.info(f"Model loaded: {model_pegasus}")


# Load dataset
logging.info("Loading samsum dataset")
dataset_samsum = load_dataset("samsum")
logging.info(f"Split lengths: {[len(dataset_samsum[split]) for split in dataset_samsum]}")
logging.info(f"Features: {dataset_samsum['train'].column_names}")
logging.info("\nDialogue:")
logging.info(dataset_samsum["test"][1]["dialogue"])
logging.info("\nSummary:")
logging.info(dataset_samsum["test"][1]["summary"])

# Summarization pipeline - baseline model inference
pipe = pipeline('summarization', model=model_ckpt)
pipe_out = pipe(dataset_samsum['test'][0]['dialogue'])
logging.info(pipe_out)
logging.info(pipe_out[0]['summary_text'].replace(" .", ".\n"))

# Calculate ROUGE scores for baseline model
rouge_metric = load_metric('rouge')
score = calculate_metric_on_test_ds(dataset_samsum['test'], rouge_metric, model_pegasus, tokenizer, column_text='dialogue', column_summary='summary', batch_size=8)

# Assign rouge scores to dataframe for organization and display
rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
rouge_dict = dict((rn, score[rn].mid.fmeasure) for rn in rouge_names)
dataframe_rouge = pd.DataFrame(rouge_dict, index=['pegasus'])
logging.info(f"ROUGE scores for baseline model: {dataframe_rouge}")


# Convert examples to features
def convert_examples_to_features(example_batch):
    input_encodings = tokenizer(example_batch['dialogue'], max_length=1024, truncation=True)
    
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
dataset_samsum_pt = dataset_samsum.map(convert_examples_to_features, batched=True)
seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)


# Custom callback for ROUGE evaluation during training - 
# each dataset has a different column name for the text and summary, 
# hence the need for a unique callback for each pipeline
class RougeCallback(TrainerCallback):
    def __init__(self, rouge_metric, eval_dataset, tokenizer, device):
        self.rouge_metric = rouge_metric
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.device = device

    def on_evaluate(self, args, state, control, **kwargs):
        trainer = kwargs['model']
        score = calculate_metric_on_test_ds(
            self.eval_dataset, self.rouge_metric, trainer, self.tokenizer, batch_size=8, device=self.device, column_text='dialogue', column_summary='summary'
        )
        rouge_dict = dict((rn, score[rn].mid.fmeasure) for rn in rouge_names)
        logging.info(f"ROUGE scores after epoch {state.epoch}: {rouge_dict}")


# Training arguments
logging.info("Setting up training arguments")
trainer_args = TrainingArguments(
    # directory for saving checkpoints
    output_dir=os.path.expanduser('~/tm/tmgp/model-trainer/checkpoints'),
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

# Log the training arguments for the model.

logging.info(f"Training arguments: {trainer_args}")

# Initialize trainer
logging.info("Initializing trainer")
trainer = Trainer(
    model=model_pegasus,
    args=trainer_args,
    tokenizer=tokenizer,
    data_collator=seq2seq_data_collator,
    train_dataset=dataset_samsum_pt["train"],
    eval_dataset=dataset_samsum_pt["validation"],
    callbacks=[RougeCallback(rouge_metric, dataset_samsum['test'], tokenizer, device)]
)

logging.info("Starting training")
trainer.train()
logging.info("Training complete")

# Calculate ROUGE scores after training
score = calculate_metric_on_test_ds(dataset_samsum['test'], rouge_metric, trainer.model, tokenizer, column_text='dialogue', column_summary='summary', batch_size=8)

# Save model and tokenizer
try:
    trainer.model.save_pretrained(os.path.expanduser('~/tm/tmgp/model-trainer/checkpoints/pegasus-samsum-model-2'))
    logging.info("Model saved successfully.")
except Exception as e:
    logging.error(f"Error saving model: {e}")

try:
    tokenizer.save_pretrained(os.path.expanduser('~/tm/tmgp/model-trainer/checkpoints/pegasus-samsum-tokenizer'))
    logging.info("Tokenizer saved successfully.")
except Exception as e:
    logging.error(f"Error saving tokenizer: {e}")

# Inference pipeline with trained model
sample_text = dataset_samsum["test"][0]["dialogue"]
reference = dataset_samsum["test"][0]["summary"]
gen_kwargs = {"length_penalty": 0.8, "num_beams": 8, "max_length": 128}

try:
    pipe = pipeline("summarization", model=os.path.expanduser('~/tm/tmgp/model-trainer/checkpoints/pegasus-samsum-model-2'), tokenizer=tokenizer)
    model_summary = pipe(sample_text, **gen_kwargs)[0]["summary_text"]
    logging.info("Dialogue:")
    logging.info(sample_text)
    logging.info("\nReference Summary:")
    logging.info(reference)
    logging.info("\nModel Summary:")
    logging.info(model_summary)
except Exception as e:
    logging.error(f"Error during summarization: {e}")


# use reusable function to generate summaries for each checkpoint
from checkpoint_summarizer import generate_summaries_for_checkpoints

# Sample text and generation parameters
sample_text = dataset_samsum["test"][0]["dialogue"]
gen_kwargs = {"length_penalty": 0.8, "num_beams": 8, "max_length": 128}

# Directory where checkpoints are saved
checkpoint_dir = os.path.expanduser('~/tm/tmgp/model-trainer/checkpoints')

# Generate summaries for each checkpoint and save to file
generate_summaries_for_checkpoints(checkpoint_dir, sample_text, gen_kwargs)

