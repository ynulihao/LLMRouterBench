# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch

from typing import List, Dict
from transformers import PreTrainedTokenizer
from llm_blender.pair_ranker.collator import encode_texts

SOURCE_PREFIX = "<|source|>"


class NClassCollator(object):
    def __init__(
        self,
        source_maxlength: int,
        tokenizer: PreTrainedTokenizer,
        candidate_maxlength: int,
        source_prefix: str = None,
    ):
        """
        Initializes the Collator.

        Args:
            source_maxlength (int): The maximum length of the source input.
            tokenizer (PreTrainedTokenizer): The tokenizer object used for tokenization.
            candidate_maxlength (int): The maximum length of the candidate input.
            source_prefix (str): The prefix to be added to the source input. Defaults to None.
        """
        self.tokenizer = tokenizer
        self.source_maxlength = source_maxlength
        self.candidate_maxlength = candidate_maxlength
        self.max_length = min(
            self.tokenizer.model_max_length,
            self.source_maxlength + 2 * self.candidate_maxlength + 6,
        )

        self.sep_token = tokenizer.sep_token or tokenizer.eos_token
        assert self.sep_token, "sep_token is not found in the tokenizer"

        self.cls_token = tokenizer.cls_token or tokenizer.bos_token
        # Note: source_prefix is kept for compatibility but not used with GTE models
        # GTE uses last-token pooling instead of source prefix token pooling

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Encode a batch of the dataset.

        Args:
            batch (List[Dict]): A list of dictionaries representing the batch of the dataset.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the processed data.
        """
        assert batch, "batch cannot be empty"
        # For GTE models, we don't add source prefix token
        # The model will use last-token pooling instead
        batch_source = [
            b['source'].replace(self.sep_token, ' ')
            for b in batch
        ]
        scores = (
            torch.tensor([b["scores"] for b in batch])
            if "scores" in batch[0] and batch[0]["scores"]
            else None
        )
        costs = (
            torch.tensor([b["costs"] for b in batch])
            if "costs" in batch[0] and batch[0]["costs"]
            else None
        )

        source_ids, source_masks = encode_texts(
            batch_source, self.tokenizer, self.source_maxlength
        )

        return {
            "source_ids": source_ids,
            "source_attention_mask": source_masks,
            "scores": scores,
            "costs": costs,
        }
