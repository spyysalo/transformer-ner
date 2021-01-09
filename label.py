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

    def label_tokens(self, word, token_texts):
        """Returns token labels for given Word and tokens."""
        num_tokens = len(token_texts)
        if word.label == 'O' or word.label.startswith('I'):
            return [word.label] * num_tokens
        else:
            try:
                assert word.label.startswith('B')
                tag, type_ = word.label.split('-', 1)
            except:
                raise ValueError(f'invalid IOB2 label {word.label}')
            return [word.label] + [f'I-{type_}'] * (num_tokens-1)


def _start_of_span(word):
    """Returns True if Word starts IOB2 or IOBES span, False otherwise."""
    if word.label == 'O':
        return False
    elif word.label[0] in 'BS':
        return True
    elif word.label[0] in 'IE':
        return False
    else:
        raise ValueError(word.label)


def _end_of_span(word):
    """Returns True if Word ends IOB2 or IOBES span, False otherwise."""
    if word.label == 'O':
        return False
    elif word.label[0] in 'ES':
        return True
    elif word.label[0] in 'BI':
        # Depends on the next word label for IOB2
        if word.next_word is None:
            return True
        elif word.next_word.label[0] in 'BSO':
            return True
        elif word.next_word.label[0] in 'IE':
            return False
        else:
            raise ValueError(word.next_word.label)
    else:
        raise ValueError(word.label)


def _single_word_span(word):
    return _start_of_span(word) and _end_of_span(word)


class IobesTokenLabeler:
    """Labels tokens with IOBES labels based on Word IOBES or IOB2 label."""

    def __init__(self, word_labels):
        self.word_labels = word_labels.copy()

    def labels(self):
        """Return token labels used by the class."""
        types = set(l.split('-', 1)[1] for l in self.word_labels if l != 'O')
        return ['O'] + [f'{t}-{l}' for t in 'BIES' for l in sorted(types)]

    def label_tokens(self, word, token_texts):
        if word.label == 'O':
            return ['O'] * len(token_texts)

        if _single_word_span(word):
            if len(token_texts) == 1:
                tags = ['S']
            else:
                tags = ['B'] + ['I'] * (len(token_texts)-2) + ['E']
        elif _start_of_span(word):
            tags = ['B'] + ['I'] * (len(token_texts)-1)
        elif _end_of_span(word):
            tags = ['I'] * (len(token_texts)-1) + ['E']
        else:
            assert word.label[0] == 'I'
            tags = ['I'] * len(token_texts)
        type_ = word.label.split('-', 1)[1]
        return [f'{tag}-{type_}' for tag in tags]


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
