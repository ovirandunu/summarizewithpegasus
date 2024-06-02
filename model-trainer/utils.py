import torch
import tqdm


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Helper function to split data into batch-sized chunks
def generate_batch_sized_chunks(list_of_elements, batch_size):
    """
    Generate batch-sized chunks from a given list of elements.

    Args:
        list_of_elements (list): The list of elements to generate chunks from.
        batch_size (int): The size of each batch.

    Yields:
        list: A batch-sized chunk of elements from the input list.

    Example:
        >>> elements = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> batch_size = 3
        >>> for batch in generate_batch_sized_chunks(elements, batch_size):
        ...     print(batch)
        [1, 2, 3]
        [4, 5, 6]
        [7, 8, 9]
        [10]
    """
    for i in range(0, len(list_of_elements), batch_size):
        yield list_of_elements[i: i + batch_size]

# Function to evaluate the summarization model on the test data
def calculate_metric_on_test_ds(dataset, metric, model, tokenizer, batch_size=8, device=device, column_text="utterance", column_summary="summary"):
    """
    Calculates a metric on a test dataset using a given model and tokenizer.

    Args:
        dataset (pandas.DataFrame): The test dataset.
        metric: The metric object used to evaluate the model's performance.
        model: The trained model.
        tokenizer: The tokenizer used to preprocess the input data.
        batch_size (int, optional): The batch size for inference. Defaults to 8.
        device (str, optional): The device to run the model on. Defaults to 'cuda' if available, else 'cpu'.
        column_text (str, optional): The column name in the dataset that contains the input text. Defaults to "utterance".
        column_summary (str, optional): The column name in the dataset that contains the target summaries. Defaults to "summary".

    Returns:
        The computed score of the metric on the test dataset.
    """
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


