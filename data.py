#!/usr/bin/env python3

"""
Functionality for loading and representing text documents.
"""

import sys
import logging

from argparse import ArgumentParser
from logging import warning

from util import logger


DOCUMENT_SEPARATOR = '-DOCSTART-'


def argparser():
    ap = ArgumentParser()
    ap.add_argument('--separator', default=None, help='TSV field separator')
    ap.add_argument('file', nargs='+')
    return ap


class Document:
    def __init__(self, sentences):
        for sentence in sentences:
            sentence.document = self
        self.sentences = sentences

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
        for word in words:
            word.sentence = self
        self.words = words
        self.document = None
        
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
            warning('Word tokenized repeatedly')
        token_texts = tokenize_func(self.text)
        token_labels = label_func(self.label, token_texts)
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


def write_conll(documents, separator='\t', include_gold_label=True,
                include_predicted_label=True, out=sys.stderr):
    for document in documents:
        print(separator.join(
            [DOCUMENT_SEPARATOR] +
            ['O'] * (include_gold_label+include_predicted_label)),
            file=out
        )
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


def load_conll(fn, separator=None):
    words, sentences = [], []
    at_document_start = False
    with open(fn) as f:
        for ln, l in enumerate(f, start=1):
            l = l.rstrip('\n')
            if l.startswith(DOCUMENT_SEPARATOR):
                if words:
                    warning(f'missing sentence separator on line {ln} in {fn}')
                    sentences.append(Sentence(words))
                    words = []
                if sentences:
                    yield Document(sentences)
                    sentences = []
                at_document_start = True
            elif l and not l.isspace():
                if at_document_start:
                    warning(f'no empty after doc start on line {ln} in {fn}')
                fields = l.split(separator)
                text, label = fields[0], fields[-1]
                words.append(Word(text, label))
                at_document_start = False
            else:
                # Blank lines separate sentences and document separator
                if words:
                    sentences.append(Sentence(words))
                    words = []
                elif not at_document_start:
                    warning(f'skipping empty sentence on line {ln} in {fn}')
                at_document_start = False
    if words:
        warning(f'missing sentence separator on line {ln} in {fn}')
        sentences.append(Sentence(words))
    if sentences:
        yield Document(sentences)


class ConllLoader:
    """Loads and tokenizes data in CoNLL-like formats."""
    def __init__(self, tokenize_func, label_func, separator=None):
        self.tokenize_func = tokenize_func
        self.label_func = label_func
        self.separator = None

    def load(self, path):
        for document in load_conll(path, separator=self.separator):
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
