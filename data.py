#!/usr/bin/env python3

"""
Functionality for loading and representing text documents.
"""

import sys
import logging
import numpy as np

from collections import OrderedDict
from argparse import ArgumentParser

from util import logger, pairwise


DOCUMENT_SEPARATOR = '-DOCSTART-'

METADATA_MARKER = '-META-'


def argparser():
    ap = ArgumentParser()
    ap.add_argument('--separator', default=None, help='TSV field separator')
    ap.add_argument('file', nargs='+')
    return ap


class Document:
    def __init__(self, sentences, metadata=None):
        for sentence in sentences:
            sentence.document = self
        self.sentences = sentences
        self.metadata = metadata

    def tokenize(self, tokenize_func, label_func):
        """Tokenize Words and assign token labels.

        Args:
            tokenize_func: function taking word text and returning list
                of token texts.
            label_func: function taking word label and token texts
                and returning list of of token labels.
        """
        for sentence in self.sentences:
            sentence.tokenize(tokenize_func, label_func)

    def sentence_count(self):
        return len(self.sentences)

    def word_count(self):
        return sum(s.word_count() for s in self.sentences)

    def token_count(self):
        return sum(s.token_count() for s in self.sentences)

    def __str__(self):
        return f'Document with {len(self.sentences)} sentences'


class Sentence:
    def __init__(self, words):
        self.words = words
        self.document = None

        for word in self.words:
            word.sentence = self
        for w1, w2 in pairwise(self.words):
            w1.next_word = w2
            w2.prev_word = w1

    def tokenize(self, tokenize_func, label_func):
        """Tokenize Words and assign token labels.

        Args:
            tokenize_func: function taking word text and returning list
                of token texts.
            label_func: function taking word label and token texts
                and returning list of of token labels.
        """
        for word in self.words:
            word.tokenize(tokenize_func, label_func)

    def word_count(self):
        return len(self.words)

    def token_count(self):
        return sum(w.token_count() for w in self.words)

    def __str__(self):
        return f'Sentence with {len(self.words)} words'


class Word:
    def __init__(self, text, label):
        self.text = text
        self.label = label
        self.tokens = None
        self.token_labels = None
        self.token_indices = None
        self.sentence = None
        self.predicted_label = None
        self.prev_word = None
        self.next_word = None

    @property
    def document(self):
        if self.sentence is None:
            return None
        else:
            return self.sentence.document

    def token_count(self):
        if self.tokens is None:
            raise ValueError('not tokenized')
        return len(self.tokens)

    def tokenize(self, tokenize_func, label_func):
        """Tokenize Word and assign token labels.

        Args:
            tokenize_func: function taking word text and returning list
                of token texts.
            label_func: function taking word label and token texts
                and returning list of of token labels.
        """
        if self.tokens is not None:
            logger.warning('Word tokenized repeatedly')
        token_texts = tokenize_func(self.text)
        if len(token_texts) == 0:
            unk_token = '[UNK]'    # TODO
            logger.warning(f'Word "{self.text}" tokenized to {token_texts}, '
                           f'replacing with {unk_token}')
            token_texts = [unk_token]
        token_labels = label_func(self, token_texts)
        self.tokens = [
            Token(t, l, self) for t, l in zip(token_texts, token_labels)
        ]

    def __str__(self):
        return f'Word({self.text}, {self.label}, {self.tokens})'


class Token:
    def __init__(self, text, label=None, word=None, is_special=False,
                 masked=False):
        self.text = text
        self.label = label
        self.word = word
        self.is_special = is_special
        self.masked = masked
        self.predictions = []
        self.pred_summary = None

    @property
    def sentence(self):
        if self.word is None:
            return None
        else:
            return self.word.sentence

    @property
    def document(self):
        if self.word is None:
            return None
        else:
            return self.word.document

    def __str__(self):
        return f'{self.text}/{self.label}'


def summarize_preds_token_avg(document):
    """Summarize token predictions using the average value."""
    for sentence in document.sentences:
        for word in sentence.words:
            for token in word.tokens:
                probs = np.stack([p[1] for p in token.predictions])
                token.pred_summary = np.mean(probs, axis=0)


def summarize_preds_token_max(document):
    """Summarize Token predictions using the maximum value."""
    for sentence in document.sentences:
        for word in sentence.words:
            for token in word.tokens:
                probs = np.stack([p[1] for p in token.predictions])
                token.pred_summary = np.max(probs, axis=0)


def summarize_preds_token_left(document):
    """Summarize Token predictions using leftmost prediction."""
    for sentence in document.sentences:
        for word in sentence.words:
            for token in word.tokens:
                sorted_probs = sorted(token.predictions)
                token.pred_summary = sorted_probs[0][1]


def summarize_preds_token_middle(document):
    """Summarize Token predictions using middle prediction."""
    for sentence in document.sentences:
        for word in sentence.words:
            for token in word.tokens:
                sorted_probs = sorted(token.predictions)
                mid = len(sorted_probs) // 2
                token.pred_summary = sorted_probs[mid][1]


def summarize_preds_token_right(document):
    """Summarize Token predictions using rightmost prediction."""
    for sentence in document.sentences:
        for word in sentence.words:
            for token in word.tokens:
                sorted_probs = sorted(token.predictions)
                token.pred_summary = sorted_probs[-1][1]


PREDICTION_SUMMARIZERS = OrderedDict([
    ('avg', summarize_preds_token_avg),
    ('max', summarize_preds_token_max),
    ('left', summarize_preds_token_left),
    ('middle', summarize_preds_token_middle),
    ('right', summarize_preds_token_right),
])


def get_prediction_summarizer(summary=None):
    if summary is None:
        return list(PREDICTION_SUMMARIZERS.keys())[0]    # default
    elif summary in PREDICTION_SUMMARIZERS:
        return PREDICTION_SUMMARIZERS[summary]
    else:
        raise ValueError(f'unknown summary strategy {summary}')


def write_conll(documents, separator='\t', include_gold_label=True,
                include_predicted_label=True, out=sys.stderr):
    for document in documents:
        fields = [DOCUMENT_SEPARATOR]
        if document.metadata is not None:
            fields += [METADATA_MARKER, document.metadata]
        else:
            fields += ['O'] * (include_gold_label+include_predicted_label)
        print(separator.join(fields), file=out)
        print(file=out)
        for sentence in document.sentences:
            for word in sentence.words:
                fields = [word.text]
                if include_gold_label:
                    fields.append(word.label)
                if include_predicted_label:
                    fields.append(word.predicted_label)
                print(separator.join(fields), file=out)
            print(file=out)


def load_labels(fn):
    labels = []
    with open(fn) as f:
        for ln, l in enumerate(f, start=1):
            fields = l.split()
            if len(fields) != 1:
                raise ValueError(f'cannot parse label on {fn} line {ln}: {l}')
            label = fields[0]
            if label in labels:
                raise ValueError(f'duplicate label on {fn} line {ln}: {l}')
            labels.append(label)
    logger.info(f'loaded labels from {fn}: {labels}')
    return labels


def load_conll(fn, separator=None, test=False):
    words, sentences = [], []
    at_document_start = False
    metadata = None
    with open(fn) as f:
        for ln, l in enumerate(f, start=1):
            l = l.rstrip('\n')
            if l.startswith(DOCUMENT_SEPARATOR):
                # Minor extension of CoNLL format: allow document metadata
                # on -DOCSTART- line
                rest = l[len(DOCUMENT_SEPARATOR):].strip()
                if rest.startswith(METADATA_MARKER):
                    metadata = rest[len(METADATA_MARKER):].strip()
                else:
                    metadata = None
                if words:
                    logger.warning(
                        f'missing sentence separator on line {ln} in {fn}')
                    sentences.append(Sentence(words))
                    words = []
                if sentences:
                    yield Document(sentences, metadata)
                    sentences = []
                at_document_start = True
            elif l and not l.isspace():
                if at_document_start:
                    logger.warning(
                        f'no empty after doc start on line {ln} in {fn}')
                fields = l.split(separator)
                text, label = fields[0], fields[-1]
                if test:
                    # blank out labels in test mode
                    label = 'O'
                words.append(Word(text, label))
                at_document_start = False
            else:
                # Blank lines separate sentences and document separator
                if words:
                    sentences.append(Sentence(words))
                    words = []
                elif not at_document_start:
                    logger.warning(
                        f'skipping empty sentence on line {ln} in {fn}')
                at_document_start = False
    if words:
        logger.warning(f'missing sentence separator on line {ln} in {fn}')
        sentences.append(Sentence(words))
    if sentences:
        yield Document(sentences, metadata)


class ConllLoader:
    """Loads and tokenizes data in CoNLL-like formats."""
    def __init__(self, tokenize_func, label_func, separator=None, test=False):
        self.tokenize_func = tokenize_func
        self.label_func = label_func
        self.separator = None
        self.test = test

    def load(self, path):
        for document in load_conll(path, separator=self.separator,
                                   test=self.test):
            document.tokenize(self.tokenize_func, self.label_func)
            yield document


def main(argv):
    args = argparser().parse_args(argv[1:])
    for fn in args.file:
        for document in load_conll(fn, args.separator):
            print(document)
            for sentence in document.sentences:
                print(sentence)
                for word in sentence.words:
                    print(word)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
