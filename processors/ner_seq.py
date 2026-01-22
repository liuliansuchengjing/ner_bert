# -*- coding: utf-8 -*-
"""
NER data processor for BIO format (char/token + tag per line, blank line separates sentences).

Key guarantees:
- Uses fast tokenizer word alignment (word_ids) so labels align with wordpieces.
- Produces label_ids padded with -100 (ignore_index) for [CLS]/[SEP]/[PAD] and continuation wordpieces.
- Exposes `collate_fn` compatible with your DataLoader.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import os
import torch


@dataclass
class InputExample:
    guid: str
    words: List[str]
    labels: List[str]


@dataclass
class InputFeatures:
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: List[int]
    label_ids: List[int]
    input_len: int  # real length (incl. special tokens) before padding


def read_bio_file(path: str) -> List[Tuple[List[str], List[str]]]:
    """Return list of (tokens, labels) where tokens are already split (char-level is OK)."""
    sentences: List[Tuple[List[str], List[str]]] = []
    words: List[str] = []
    labels: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if words:
                    sentences.append((words, labels))
                    words, labels = [], []
                continue
            # allow either "tok label" or "tok\tlabel"
            parts = line.split()
            if len(parts) < 2:
                # skip malformed line
                continue
            tok, lab = parts[0], parts[-1]
            words.append(tok)
            labels.append(lab)
    if words:
        sentences.append((words, labels))
    return sentences


class NerProcessor:
    """Processor for BIO files: train.txt / dev.txt / test.txt under data_dir."""
    def __init__(self, data_dir: str, task_name: str = "myner"):
        self.data_dir = data_dir
        self.task_name = task_name

    def get_train_examples(self) -> List[InputExample]:
        return self._create_examples(read_bio_file(os.path.join(self.data_dir, "train.txt")), "train")

    def get_dev_examples(self) -> List[InputExample]:
        return self._create_examples(read_bio_file(os.path.join(self.data_dir, "dev.txt")), "dev")

    def get_test_examples(self) -> List[InputExample]:
        return self._create_examples(read_bio_file(os.path.join(self.data_dir, "test.txt")), "test")

    @staticmethod
    def _create_examples(lines: List[Tuple[List[str], List[str]]], set_type: str) -> List[InputExample]:
        examples: List[InputExample] = []
        for i, (words, labels) in enumerate(lines):
            guid = f"{set_type}-{i}"
            examples.append(InputExample(guid=guid, words=words, labels=labels))
        return examples


def convert_examples_to_features(
    examples: List[InputExample],
    label2id: Dict[str, int],
    max_seq_length: int,
    tokenizer,
) -> List[InputFeatures]:
    """
    Use fast tokenizer alignment to map word-level labels to wordpieces.
    Requires tokenizer to be a *fast* tokenizer (AutoTokenizer(..., use_fast=True)).
    """
    features: List[InputFeatures] = []
    for ex in examples:
        # Tokenize as pre-split tokens
        enc = tokenizer(
            ex.words,
            is_split_into_words=True,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_attention_mask=True,
            return_token_type_ids=True,
        )
        word_ids = enc.word_ids()  # list length == max_seq_length
        label_ids: List[int] = []
        prev_word_id = None
        for wi in word_ids:
            if wi is None:
                label_ids.append(-100)  # special tokens / padding
            elif wi != prev_word_id:
                # first piece of a word
                lab = ex.labels[wi] if wi < len(ex.labels) else "O"
                label_ids.append(label2id.get(lab, label2id["O"]))
            else:
                # subsequent wordpiece of same word: ignore
                label_ids.append(-100)
            prev_word_id = wi

        # compute real length (up to first padding) for later metric slicing
        attn = enc["attention_mask"]
        input_len = int(sum(attn))  # includes [CLS]/[SEP], excludes padding

        features.append(
            InputFeatures(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                token_type_ids=enc.get("token_type_ids", [0] * max_seq_length),
                label_ids=label_ids,
                input_len=input_len,
            )
        )
    return features


def collate_fn(batch: List[Tuple[torch.Tensor, ...]]):
    """
    Your run_ner_softmax builds TensorDataset of:
      (input_ids, attention_mask, token_type_ids, input_lens, label_ids)
    This collate keeps that order.
    """
    # Each element is already fixed-length tensors, so just stack.
    input_ids = torch.stack([x[0] for x in batch], dim=0)
    attention_mask = torch.stack([x[1] for x in batch], dim=0)
    token_type_ids = torch.stack([x[2] for x in batch], dim=0)
    input_lens = torch.stack([x[3] for x in batch], dim=0)
    label_ids = torch.stack([x[4] for x in batch], dim=0)
    return input_ids, attention_mask, token_type_ids, input_lens, label_ids
