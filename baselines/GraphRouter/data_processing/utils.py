import json
import re
import string
import pickle
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from bert_score import score
import litellm

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')


# File I/O functions
def loadjson(filename: str) -> dict:
    """
    Load data from a JSON file.

    Args:
        filename: Path to the JSON file

    Returns:
        Dictionary containing the loaded JSON data
    """
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def savejson(data: dict, filename: str) -> None:
    """
    Save data to a JSON file.

    Args:
        data: Dictionary to save
        filename: Path where the JSON file will be saved
    """
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def loadpkl(filename: str) -> any:
    """
    Load data from a pickle file.

    Args:
        filename: Path to the pickle file

    Returns:
        The unpickled object
    """
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data


def savepkl(data: any, filename: str) -> None:
    """
    Save data to a pickle file.

    Args:
        data: Object to save
        filename: Path where the pickle file will be saved
    """
    with open(filename, 'wb') as pkl_file:
        pickle.dump(data, pkl_file)


# Text normalization and evaluation functions
def normalize_answer(s: str, normal_method: str = "") -> str:
    """
    Normalize text for evaluation.

    Args:
        s: String to normalize
        normal_method: Method for normalization ("mc" for multiple choice, "" for standard)

    Returns:
        Normalized string
    """

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    def mc_remove(text):
        a1 = re.findall('\([a-zA-Z]\)', text)
        if len(a1) == 0:
            return ""
        return re.findall('\([a-zA-Z]\)', text)[-1]

    if normal_method == "mc":
        return mc_remove(s)
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    """
    Calculate F1 score between prediction and ground truth.

    Args:
        prediction: Predicted text
        ground_truth: Ground truth text

    Returns:
        Tuple of (f1, precision, recall)
    """
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return ZERO_METRIC

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1, precision, recall


def exact_match_score(prediction: str, ground_truth: str, normal_method: str = "") -> bool:
    """
    Check if prediction exactly matches ground truth after normalization.

    Args:
        prediction: Predicted text
        ground_truth: Ground truth text
        normal_method: Method for normalization

    Returns:
        True if exact match, False otherwise
    """
    return (normalize_answer(prediction, normal_method=normal_method) ==
            normalize_answer(ground_truth, normal_method=normal_method))


def get_bert_score(generate_response: List[str], ground_truth: List[str]) -> float:
    """
    Calculate BERT score between generated responses and ground truths.

    Args:
        generate_response: List of generated responses
        ground_truth: List of ground truth texts

    Returns:
        Average BERT score (F1)
    """
    F_l = []
    for inter in range(len(generate_response)):
        generation = generate_response[inter]
        gt = ground_truth[inter]
        P, R, F = score([generation], [gt], lang="en", verbose=True)
        F_l.append(F.mean().numpy().reshape(1)[0])
    return np.array(F_l).mean()


# Embedding and dimensionality reduction
def reduce_embedding_dim(embed: np.ndarray, dim: int = 50) -> np.ndarray:
    """
    Reduce dimensionality of embeddings using PCA.

    Args:
        embed: Embedding vectors
        dim: Target dimension

    Returns:
        Reduced embeddings
    """
    pca = PCA(n_components=dim)
    reduced_embeddings = pca.fit_transform(embed)
    return reduced_embeddings


def get_embedding(instructions: List[str]) -> np.ndarray:
    """
    Get embeddings for a list of texts and optionally reduce dimensions.

    Args:
        instructions: List of texts to embed
        dim: Target dimension for embeddings

    Returns:
        Numpy array of embeddings
    """
    emb_list = model.encode(instructions)
    return emb_list



# LLM prompting
def model_prompting(
        llm_model: str,
        prompt: str,
        return_num: Optional[int] = 1,
        max_token_num: Optional[int] = 512,
        temperature: Optional[float] = 0.0,
        top_p: Optional[float] = None,
        stream: Optional[bool] = None,
) -> str:
    """
    Get a response from an LLM model using LiteLLM.

    Args:
        llm_model: Name of the model to use
        prompt: Input prompt text
        return_num: Number of completions to generate
        max_token_num: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        stream: Whether to stream the response

    Returns:
        Generated text response
    """
    completion = litellm.completion(
        model=llm_model,
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=max_token_num,
        n=return_num,
        top_p=top_p,
        temperature=temperature,
        stream=stream,
    )
    content = completion.choices[0].message.content
    return content