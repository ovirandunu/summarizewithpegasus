# Investigating Dialogue Summarization with Pegasus

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![HuggingFace Transformers](https://img.shields.io/badge/Transformers-4.5.1-green)](https://github.com/huggingface/transformers)


## Authors

- **Ovindu Chakrawarthige** - [GitHub](https://www.github.com/ovirandunu)
- **Clàudia Domènech Farré** - [GitHub](https://www.github.com/cdomenechfarre)
- **Laure Hajislam** - [GitHub](https://www.github.com/laurexh77)


## Abstract

This project investigates how fine-tuning and knowledge transfer fine-tuning can enhance the Pegasus model's ability to summarize dialogues from diverse datasets. Using datasets like SamSum, NPC (Character Dialogue), and Empathetic Dialogues, the project applies advanced NLP techniques using the HuggingFace Transformers library. The study aims to improve summarization performance and provide insights into optimizing model performance for real-world applications such as customer service, healthcare, and project management.

## Research Questions

1. How can fine-tuning improve the Pegasus model's ability to summarize dialogues?
2. What is the impact of knowledge transfer fine-tuning on summarization performance across different datasets?
3. How do different conversational contexts affect the performance of the Pegasus model and its ability to adapt to different data contexts through fine-tuning?

## Dataset

The datasets used in this project include:

- **SamSum**: Simulates typical messenger-like conversations, testing the model's ability to summarize everyday dialogue.
- **NPC (Character Dialogue)**: Contains fictional character interactions, challenging the model to capture nuanced, story-driven exchanges.
- **Empathetic Dialogues**: Comprises emotion-rich conversations, assessing the model's capability to handle emotional subtleties in dialogues.

The datasets were chosen for their diversity in dialogue types, ensuring a comprehensive evaluation of the model's performance. Each dataset was processed and tokenized using the Pegasus tokenizer, and data was split into training, validation, and test sets as per standard practice.

### Division of Tasks

- **Ovindu**: Integrate and modularize code.
- **Clàudia**: Compile information and insights, started writing the report.
- **Laure**: Conduct statistical tests, analyze results.

## Documentation

### Directory Structure

```kotlin

Parent/
│
├── data/
│
├── model-trainer/
│   ├── checkpoints/
│   ├── empathetic/
│   ├── npc/
│   ├── checkpoint_summarizer.py
│   ├── empathetic_summaries.py
│   ├── empathetic.sh
│   ├── experiments.ipynb
│   ├── npc_summaries.py
│   ├── npc.sh
│   ├── trainer.py
│   ├── trainer.sh
│   ├── utils.py
│
├── venv/
├── .gitignore
├── notebook-experiment.ipynb
├── README.md
└── Report.pdf

```

### How to run the code

1. **Environment Setup**:
    - Clone the repository: `git clone <repository_url>`
    - Create a virtual environment: `python -m venv venv`
    - Activate the virtual environment: `source venv/bin/activate` (Linux/macOS) or `venv\Scripts\activate` (Windows)
    - Install the required packages: `pip install -r requirements.txt`
2. **Training**:
    - The code is optimized to be run on an HPC cluster with GPU availability for cuda compatibility. Once pushed to such an environment, this code can be executed using a SLURM workload manager interface.
    - Fine-tune the model on each dataset using the provided `.sh` scripts (e.g., `sbatch trainer.sh`).
    - Note that this process needs to be iteratively executed for evaluations with the correct checkpoints being provided to the code. Refer to our report for the entire methodology we followed in our investigation.

## Libraries and Configurations

### Libraries Used

The project uses several key libraries and frameworks to facilitate model training, evaluation, and processing:

- **Transformers**: Provided by HuggingFace, this library is used to access pre-trained language models and tokenizers, and to facilitate model fine-tuning.
- **Datasets**: Also from HuggingFace, this library is used to load and preprocess the dialogue datasets.
- **Torch**: PyTorch is used as the deep learning framework for model training and inference.
- **NLTK**: The Natural Language Toolkit is used for text processing and tokenization.
- **TQDM**: This library is used to provide progress bars during training and evaluation.
- **Pandas**: Used for data manipulation and organization.
- **Matplotlib**: Used for plotting and visualizations.
- **Logging**: Python's logging library is used for tracking the progress and results of training and evaluation.

### HuggingFace Transformers Configuration

The Transformers library by HuggingFace is central to this project, providing tools for model training, evaluation, and inference. Key configurations and features used include:

- **AutoModelForSeq2SeqLM**: Loads the Pegasus model specifically designed for sequence-to-sequence tasks such as summarization.
- **AutoTokenizer**: Tokenizes the input text to the format required by the Pegasus model.
- **DataCollatorForSeq2Seq**: Handles dynamic padding and collation of sequences during training.
- **TrainingArguments**: Configures the training process with parameters such as learning rate, batch size, and weight decay.
- **Trainer**: Facilitates the training loop, handles optimization, and manages checkpoints.

### Training Configurations

Key training configurations include:

- **Learning Rate Scheduling**: Dynamic adjustment of the learning rate during training to ensure optimal convergence.
- **Weight Decay**: Regularization technique applied to prevent overfitting by penalizing large weights.
- **Gradient Accumulation**: Technique to effectively increase the batch size by accumulating gradients over several steps before updating the model weights.
- **Evaluation Strategy**: Evaluation is performed at regular intervals (every 500 steps) during training to monitor progress and adjust as necessary.
- **Checkpointing**: Models are saved at the end of each epoch, allowing for later evaluation and further fine-tuning if needed.

### Example Configuration in `trainer.py`:

```python
pythonCopy code
trainer_args = TrainingArguments(
    output_dir=os.path.expanduser('~/tm/tmgp/model-trainer/checkpoints'),  # Directory for saving checkpoints
    num_train_epochs=6,  # Total number of training epochs
    warmup_steps=500,  # Number of warmup steps for learning rate scheduler
    per_device_train_batch_size=4,  # Batch size for training
    per_device_eval_batch_size=4,  # Batch size for evaluation
    weight_decay=0.01,  # Decay rate for learning rate
    eval_steps=500,  # Evaluate at every 500 steps
    evaluation_strategy='steps',  # Evaluation strategy
    save_strategy='epoch',  # Save at every epoch
    gradient_accumulation_steps=16,  # Optimize GPU memory usage with gradient accumulation
)

```

### Description of Scripts

- **trainer.py**: Main training script that sets up the model, tokenizer, and training pipeline for the SamSum dataset.
- **utils.py**: Utility functions for data processing, batching, and metric calculation.
- **checkpoint_summarizer.py**: Script to generate summaries from model checkpoints.
- **empathetic_summaries.py & npc_summaries.py**: Scripts adapted for respective datasets, utilizing the same libraries and pipeline.

### Utility Functions

- **calculate_metric_on_test_ds**: Evaluates the summarization model on the test data using ROUGE metrics.
- **generate_batch_sized_chunks**: Helper function to split data into batch-sized chunks.

### Checkpoint Summarization

The script `checkpoint_summarizer.py` is used to generate summaries from model checkpoints and includes:

- **summarize_with_checkpoint**: Summarizes the given text using a pre-trained model checkpoint.
- **generate_summaries_for_checkpoints**: Generates summaries for all checkpoints and saves them to a file.

These configurations and libraries work together to provide a robust framework for fine-tuning and evaluating the Pegasus model across different datasets, ensuring high performance and scalability.

### Logging

- Logging is configured to output both to a log file (`trainer.log`) and the SLURM output files.

## References

```
@misc{wolf2020huggingfaces,
      title={HuggingFace's Transformers: State-of-the-art Natural Language Processing}, 
      author={Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush},
      year={2020},
      eprint={1910.03771},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@misc{empathetic_dialogues_summary,
  title = {Empathetic Dialogues Summary},
  howpublished = {\url{https://huggingface.co/datasets/jtatman/empathetic_dialogues_summary}},
  note = {Accessed: 2024-06-02}
}

@misc{zhang2019pegasus,
    title={PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization},
    author={Jingqing Zhang and Yao Zhao and Mohammad Saleh and Peter J. Liu},
    year={2019},
    eprint={1912.08777},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}

@inproceedings{gliwa-etal-2019-samsum,
    title = "{SAMS}um Corpus: A Human-annotated Dialogue Dataset for Abstractive Summarization",
    author = "Gliwa, Bogdan  and
      Mochol, Iwona  and
      Biesek, Maciej  and
      Wawer, Aleksander",
    booktitle = "Proceedings of the 2nd Workshop on New Frontiers in Summarization",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-5409",
    doi = "10.18653/v1/D19-5409",
    pages = "70--79"
}

@misc{npc_engine_light_batch_2024,
  title = {LIGHT Dialogue Summarization Batch},
  howpublished = {\url{https://huggingface.co/datasets/npc-engine/light-batch-summarize-dialogue}},
  note = {Accessed: 2024-06-02}
}

```
