"""
Support for generating Examples from Documents.
"""

import sys
import numpy as np

from collections import OrderedDict
from itertools import cycle


class Example:
    def __init__(self, tokens, token_ids, label_ids, input_weights,
                 segment_ids):
        self.tokens = tokens
        self.token_ids = np.asarray(token_ids)
        self.label_ids = np.asarray(label_ids)
        self.input_weights = np.asarray(input_weights)
        self.segment_ids = np.asarray(segment_ids)
        
    def __str__(self):
        token_texts = [t.text for t in self.tokens]
        token_labels = [t.label for t in self.tokens]
        return (
            f'Example:\n'
            f'token texts:\n{token_texts}\n'
            f'token labels:\n{token_labels}\n'
            f'token_ids:\n{self.token_ids}\n'
            f'label_ids:\n{self.label_ids}\n'
            f'input_weights:\n{self.input_weights}\n'
            f'segment_ids:\n{self.segment_ids}\n'
        )


class ExampleGenerator:
    """Base class for example generators."""
    def __init__(self, seq_len, cls_token, sep_token, pad_token,
                 encode_tokens, encode_labels, align='left'):
        self.seq_len = seq_len
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.encode_tokens = encode_tokens
        self.encode_labels = encode_labels
        self.align = align
        self.max_tokens = self.seq_len - 1    # default: one for cls_token

    def check_document(self, document, doc_id=None):
        if not document.sentences:
            raise ValueError(f'no sentences in document {doc_id}')
        for sent_idx, sentence in enumerate(document.sentences, start=1):
            if not sentence.words:
                raise ValueError(
                    f'empty sentence {sent_idx} in document {doc_id}')
                for word_idx, word in enumerate(sentence, words, start=1):
                    if not word.tokens:
                        raise ValueError(
                            f'no tokens for word {word_idx} ({word.text}) '
                            f'in sentence {sent_idx} in document {doc_id}')
                    if len(word.tokens) > self.max_tokens:
                        # TODO consider trimming excessively long tokens
                        raise ValueError(
                            f'word token length ({len(word.tokens)}) '
                            f'exceeds max sequence length ({max_len}): '
                            f'{word.tokens}'
                        )

    def make_example(self, tokens):
        if not tokens:
            raise ValueError(f'empty token sequence')
        if len(tokens) > self.max_tokens:
            raise ValueError(f'number of tokens ({len(token_texts)}) '
                             f'exceeds max_tokens ({self.max_tokens})')
    
        if self.align == 'left':
            # First token is CLS, then text tokens, and finally pad
            padded = (
                [self.cls_token] +
                tokens +
                [self.pad_token] * (self.seq_len-len(tokens)-1)
            )
        elif self.align == 'right':
            # First token is CLS, pad, and finally tokens
            padded = (
                [self.cls_token] +
                [self.pad_token] * (self.seq_len-len(tokens)-1) +
                tokens
            )

        # Encode tokens and labels
        token_texts = [t.text for t in padded]
        token_labels = [t.label for t in padded]
        token_ids = self.encode_tokens(token_texts)
        label_ids = self.encode_labels(token_labels)
        input_weights = [int(not t.masked) for t in padded]
        segment_ids = [t.segment_id for t in padded]
    
        return Example(padded, token_ids, label_ids, input_weights,
                       segment_ids)

    def fill(self, tokens, sentences, current_idx):
        raise NotImplementedError    # use a subclass

    def examples(self, documents):
        for doc_idx, document in enumerate(documents):
            self.check_document(document, doc_idx)
            for sent_idx, sentence in enumerate(document.sentences):
                tokens = []
                for word_idx, word in enumerate(sentence.words):
                    if len(tokens) + len(word.tokens) > self.max_tokens:
                        # sentence doesn't fit fully, split at word boundary
                        yield self.make_example(tokens)
                        tokens = []
                    tokens.extend(word.tokens)
                if tokens:
                    self.fill(tokens, document.sentences, sent_idx)
                    yield self.make_example(tokens)


class SingleSentenceExampleGenerator(ExampleGenerator):
    """Generates examples containing single sentence each."""
    def __init__(self, *args):
        super().__init__(*args)

    def fill(self, tokens, sentences, current_idx):
        pass    # no fill


class FillSentenceExampleGenerator(ExampleGenerator):
    """Generates example for each sentence, filled from next sentences."""
    def __init__(self, *args):
        super().__init__(*args)

    def fill(self, tokens, sentences, current_idx):
        for sentence in sentences[current_idx+1:]:
            if len(tokens) < self.max_tokens:
                tokens.append(self.sep_token)
            for word in sentence.words:
                if len(tokens) + len(word.tokens) + 1 > self.max_tokens:
                    return tokens    # word doesn't fit, don't include partial
                tokens.extend(word.tokens)
        return tokens


class WrapSentenceExampleGenerator(ExampleGenerator):
    """Generates example for each sentence, filled from next sentences
    with wrapping back to document start."""
    def __init__(self, *args):
        super().__init__(*args)

    def fill(self, tokens, sentences, current_idx):
        sentences = cycle(sentences)
        for i in range(current_idx+1):
            next(sentences)    # skip past current
        for sentence in sentences:
            if len(tokens) < self.max_tokens:
                tokens.append(self.sep_token)
            for word in sentence.words:
                if len(tokens) + len(word.tokens) + 1 > self.max_tokens:
                    return tokens    # word doesn't fit, don't include partial
                tokens.extend(word.tokens)
        return tokens


EXAMPLE_GENERATORS = OrderedDict([
    ('wrap', WrapSentenceExampleGenerator),    # tends to be the best option
    ('fill', FillSentenceExampleGenerator),
    ('single', SingleSentenceExampleGenerator),
])


def examples_to_inputs(examples):
    """Stack Example data, return X and Y for model input."""
    x = {
        'input_ids': np.stack([e.token_ids for e in examples]),
        'token_type_ids': np.stack([e.segment_ids for e in examples]),
        'attention_mask': np.stack([e.input_weights for e in examples]),
    }
    y = np.stack([e.label_ids for e in examples])
    return x, y


def argparser():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument('--model_name', default=None, required=True,
                    help='pretrained model name')
    ap.add_argument('--max_seq_length', type=int, default=128,
                    help='maximum input sequence length')
    ap.add_argument('--labels', metavar='FILE', required=True,
                    help='file with labels (one per line)')
    ap.add_argument('--cache_dir', default=None,
                    help='transformers cache directory')
    ap.add_argument('conll_data', nargs='+')
    return ap


def main(argv):
    # test example generation
    from data import Token, ConllLoader, load_labels
    from label import Iob2TokenLabeler, LabelEncoder
    from transformers import AutoConfig, AutoTokenizer

    options = argparser().parse_args(argv[1:])
    seq_len = options.max_seq_length
    
    word_labels = load_labels(options.labels)
    token_labeler = Iob2TokenLabeler(word_labels)    # TODO add argument
    token_labels = token_labeler.labels()
    label_func = token_labeler.label_tokens
    label_encoder = LabelEncoder(token_labels, padding_label='O')    # TODO
    encode_labels = label_encoder.encode

    config = AutoConfig.from_pretrained(
        options.model_name, cache_dir=options.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        options.model_name, config=config, cache_dir=options.cache_dir)
    tokenize_func = tokenizer.tokenize
    encode_tokens = lambda t: tokenizer.encode(t, add_special_tokens=False)

    document_loader = ConllLoader(
        tokenize_func,
        label_func
    )
    example_generator = WrapSentenceExampleGenerator(
        seq_len,
        Token(tokenizer.cls_token, is_special=True, masked=False),
        Token(tokenizer.sep_token, is_special=True, masked=False),
        Token(tokenizer.pad_token, is_special=True, masked=True),
        encode_tokens,
        encode_labels
    )

    for fn in options.conll_data:
        documents = list(document_loader.load(fn))
        examples = list(example_generator.examples(documents))
        for i, example in enumerate(examples):
            print(f'example {i}')
            print(example)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
