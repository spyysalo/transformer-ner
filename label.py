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


def assign_labels_first(document, label_encoder):
    """Assign Word labels based on prediction summary for first Token."""
    for sentence in document.sentences:
        for word in sentence.words:
            label_idx = np.argmax(word.tokens[0].pred_summary, axis=-1)
            word.predicted_label = label_encoder.inv_label_map[label_idx]


LABEL_ASSIGNERS = OrderedDict([
    ('first', assign_labels_first),
])
