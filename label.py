#!/usr/bin/env python3

"""
Functionalify for processing labels.
"""

import numpy as np

from collections import OrderedDict

from util import most_common


class Iob2TokenLabeler:
    """Labels tokens based on extension of Word IOB2 label."""
    def __init__(self, word_labels):
        self.word_labels = word_labels.copy()

    def labels(self):
        """Return token labels used by the class."""
        return self.word_labels

    def label_tokens(self, word_label, token_texts):
        """Returns token labels for given Word label and tokens."""
        num_tokens = len(token_texts)
        if word_label == 'O' or word_label.startswith('I'):
            return [word_label] * num_tokens
        else:
            try:
                assert word_label.startswith('B')
                tag, type_ = word_label.split('-')
            except:
                raise ValueError(f'invalid IOB2 label {word_label}')
            return [word_label] + [f'I-{type_}'] * (num_tokens-1)


class LabelEncoder:
    """Implements mapping from labels to integer values."""

    def __init__(self, labels, special_token_label='O'):
        self.labels = labels.copy()
        self.label_map = { k: v for v, k in enumerate(labels) }
        self.inv_label_map = { v: k for k, v in self.label_map.items() }
        # special Tokens ([SEP], [PAD], etc.) have the label None
        # and are mapped according to special_token_label.
        self.label_map[None] = self.label_map[special_token_label]
        
    def encode(self, labels):
        return [self.label_map[label] for label in labels]

    def decode(self, label_indices):
        return [self.inv_label_map[idx] for idx in label_indices]


def assign_labels_first_left(document, label_encoder):
    """Assign Word labels based on leftmost prediction for first Token."""
    for sentence in document.sentences:
        for word in sentence.words:
            sorted_probs = sorted(word.tokens[0].predictions)
            label_idx = np.argmax(sorted_probs[0][1], axis=-1)
            word.predicted_label = label_encoder.inv_label_map[label_idx]


def assign_labels_first_middle(document, label_encoder):
    """Assign Word labels based on middle prediction for first Token."""
    for sentence in document.sentences:
        for word in sentence.words:
            sorted_probs = sorted(word.tokens[0].predictions)
            mid = len(sorted_probs) // 2
            label_idx = np.argmax(sorted_probs[mid][1], axis=-1)
            word.predicted_label = label_encoder.inv_label_map[label_idx]


def assign_labels_first_right(document, label_encoder):
    """Assign Word labels based on rightmost prediction for first Token."""
    for sentence in document.sentences:
        for word in sentence.words:
            sorted_probs = sorted(word.tokens[0].predictions)
            label_idx = np.argmax(sorted_probs[-1][1], axis=-1)
            word.predicted_label = label_encoder.inv_label_map[label_idx]


def assign_labels_first_max(document, label_encoder):
    """Assign Word labels based on the maximum probability for first Token."""
    for sentence in document.sentences:
        for word in sentence.words:
            probs = np.stack([p[1] for p in word.tokens[0].predictions])
            label_idx = np.argmax(np.max(probs, axis=0), axis=-1)
            word.predicted_label = label_encoder.inv_label_map[label_idx]


def assign_labels_first_avg(document, label_encoder):
    """Assign Word labels based on the maximum probability for first Token."""
    for sentence in document.sentences:
        for word in sentence.words:
            probs = np.stack([p[1] for p in word.tokens[0].predictions])
            label_idx = np.argmax(np.mean(probs, axis=0), axis=-1)
            word.predicted_label = label_encoder.inv_label_map[label_idx]


def assign_labels_first_freq(document, label_encoder):
    """Assign Word labels based on most frequent prediction for first Token."""
    for sentence in document.sentences:
        for word in sentence.words:
            labels = [
                label_encoder.inv_label_map[np.argmax(p[1])]
                for p in word.tokens[0].predictions
            ]
            word.predicted_label = most_common(labels)


LABEL_ASSIGNERS = OrderedDict([
    ('avg', assign_labels_first_avg),
    ('freq', assign_labels_first_freq),
    ('left', assign_labels_first_left),
    ('middle', assign_labels_first_middle),
    ('right', assign_labels_first_right),
    ('max', assign_labels_first_max),
])
