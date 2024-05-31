import os
import glob
import logging
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# Ensure logging is configured in the main script
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = "cuda" if torch.cuda.is_available() else "cpu"

def summarize_with_checkpoint(checkpoint_path, sample_text, gen_kwargs):
    try:
        logging.info(f"Loading model from {checkpoint_path}")
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
        summary = summarizer(sample_text, **gen_kwargs)[0]["summary_text"]
        return summary
    except Exception as e:
        logging.error(f"Error summarizing with checkpoint {checkpoint_path}: {e}")
        return None

def generate_summaries_for_checkpoints(checkpoint_dir, sample_text, gen_kwargs, output_file='summaries.txt'):
    try:
        checkpoint_paths = sorted(glob.glob(f"{checkpoint_dir}/checkpoint-*"))
        summaries = {}
        for checkpoint_path in checkpoint_paths:
            try:
                epoch = checkpoint_path.split('-')[-1]
                summary = summarize_with_checkpoint(checkpoint_path, sample_text, gen_kwargs)
                if summary:
                    summaries[epoch] = summary
                    logging.info(f"Epoch {epoch}:")
                    logging.info("Dialogue:")
                    logging.info(sample_text)
                    logging.info("\nModel Summary:")
                    logging.info(summary)
            except Exception as e:
                logging.error(f"Error processing checkpoint {checkpoint_path}: {e}")

        with open(os.path.join(checkpoint_dir, output_file), 'w') as f:
            for epoch, summary in summaries.items():
                f.write(f"Epoch {epoch}:\n")
                f.write(f"Model Summary: {summary}\n\n")

        logging.info(f"Summaries saved to {os.path.join(checkpoint_dir, output_file)}")
    except Exception as e:
        logging.error(f"Error generating summaries for checkpoints: {e}")

# Example usage in the main script
# from checkpoint_summarizer import generate_summaries_for_checkpoints
# sample_text = dataset_samsum["test"][0]["dialogue"]
# gen_kwargs = {"length_penalty": 0.8, "num_beams": 8, "max_length": 128}
# checkpoint_dir = os.path.expanduser('~/tm/tmgp/model-trainer/checkpoints')
# generate_summaries_for_checkpoints(checkpoint_dir, sample_text, gen_kwargs)

