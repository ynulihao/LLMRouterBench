import pandas as pd
from utils import loadjson, get_embedding, savepkl, loadpkl
import yaml

def generate_unified_qa_dataset(output_path='data/unified_qa_data.csv',sample_size=60):
    """
    Generate a unified question-answering dataset from multiple data sources.

    Parameters:
    sample_size (int): Number of samples to extract from each dataset
    output_path (str): Path to save the output CSV file

    Returns:
    pandas.DataFrame: The generated unified dataset
    """
    # Initialize result DataFrame
    df = pd.DataFrame(columns=[
        'task_id', 'query', 'ground_truth', 'metric',
        'task_description'  # Added task description column
    ])

    # Define dataset paths and corresponding task names with descriptions
    dataset_configs = [
        {
            'task_name': 'alpaca_data',
            'path': 'data/alpaca_data/alpaca_data.json',
            'format': 'json',
            'query_fields': ['instruction', 'input'],
            'ground_truth_field': 'output',
            'metric': 'f1_score',
            'task_description': 'The Alpaca dataset is designed for instruction-following tasks, where the model is required to generate coherent and contextually appropriate responses to given instructions or prompts. It focuses on understanding diverse user requests and providing informative and accurate outputs based on those instructions.'
        },
        {
            'task_name': 'GSM8K',
            'path': 'data/GSM8K/GSM8K.json',
            'format': 'json',
            'query_fields': ['instruction', 'input'],
            'ground_truth_field': 'answer',
            'metric': 'GSM8K',
            'task_description': 'The GSM8K dataset is tailored for mathematical problem-solving tasks. It consists of natural language math problems that require the model to comprehend the problem statement, apply the correct mathematical operations, and provide the solution. The primary challenge lies in both parsing complex language and performing accurate calculations.'
        },
        {
            'task_name': 'multi_news',
            'path': 'data/multi_news/multi_news.json',
            'format': 'json',
            'query_fields': ['instruction', 'input'],
            'ground_truth_field': 'output',
            'metric': 'f1_score',
            'task_description': 'The Multi-News dataset is aimed at text summarization tasks. It contains multiple news articles on the same topic, and the model\'s objective is to generate a concise and comprehensive summary that integrates information from all the articles. The challenge is to distill key points while maintaining coherence and avoiding redundancy.'
        },
        {
            'task_name': 'SQUAD',
            'path': 'data/SQUAD/SQUAD.parquet',
            'format': 'parquet',
            'query_field': 'question',
            'ground_truth_field': 'answers',
            'ground_truth_subfield': 'text',
            'ground_truth_index': 0,
            'metric': 'f1_score',
            'task_description': 'The SQuAD dataset is focused on question-answering tasks, where the model is given a passage of text and needs to extract or generate a precise answer to a question based on the content of the passage. The dataset emphasizes comprehension, retrieval of relevant information, and concise answer generation.'
        }
    ]

    # Process each dataset
    for config in dataset_configs:
        # Load data
        if config['format'] == 'json':
            data = loadjson(config['path'])[:sample_size]

            # Process JSON formatted data
            for item in data:
                # Construct query text based on configuration
                if isinstance(config['query_fields'], list):
                    query = ''.join([item[field] for field in config['query_fields']])
                else:
                    query = item[config['query_fields']]

                # Get ground truth
                ground_truth = item[config['ground_truth_field']]

                # Add to dataset
                new_row = {
                    'task_id': config['task_name'],
                    'query': query,
                    'ground_truth': ground_truth,
                    'metric': config['metric'],
                    'task_description': config['task_description']  # Add task description
                }
                df = df._append(new_row, ignore_index=True)

        elif config['format'] == 'parquet':
            data = pd.read_parquet(config['path'])[:sample_size]

            # Process Parquet formatted data
            for item in data.itertuples():
                query = getattr(item, config['query_field'])

                # Handle complex ground truth structures
                if 'ground_truth_subfield' in config:
                    ground_truth_container = getattr(item, config['ground_truth_field'])
                    ground_truth = ground_truth_container[config['ground_truth_subfield']][config['ground_truth_index']]
                else:
                    ground_truth = getattr(item, config['ground_truth_field'])

                # add to dataset
                new_row = {
                    'task_id': config['task_name'],
                    'query': query,
                    'ground_truth': ground_truth,
                    'metric': config['metric'],
                    'task_description': config['task_description']  # Add task description
                }
                df = df._append(new_row, ignore_index=True)

    # Save results to CSV
    df.to_csv(output_path, index=False)

    return df


# Usage example
if __name__ == "__main__":
    # Open config file
    with open("configs/config.yaml", 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    # Generate dataset with default sample size
    unified_dataset = generate_unified_qa_dataset(config['unified_qa_data_path'])

    # Or specify custom sample size
    # unified_dataset = generate_unified_qa_dataset(config['unified_qa_data_path'],sample_size=100)