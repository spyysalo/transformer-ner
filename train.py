import sys
import numpy as np

from collections import OrderedDict
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from data import ConllLoader, Token, load_labels, write_conll
from data import PREDICTION_SUMMARIZERS
from label import LabelEncoder, Iob2TokenLabeler, LABEL_ASSIGNERS
from example import EXAMPLE_GENERATORS, examples_to_inputs
from model import load_pretrained, get_optimizer, build_ner_model
from model import save_ner_model
from evaluation import conlleval_report, evaluate_assign_labels_funcs
from evaluation import evaluate_viterbi
from util import logger, timed, unique
from util import LRHistory, log_examples, log_dataset_statistics
from viterbi import ViterbiDecoder

from defaults import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_SEQ_LEN,
    DEFAULT_LR,
    DEFAULT_WARMUP_PROPORTION
)


def argparser():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument('--model_name', default=None, required=True,
                    help='pretrained model name')
    ap.add_argument('--max_seq_length', type=int, default=128,
                    help='maximum input sequence length')
    ap.add_argument('--labels', metavar='FILE', required=True,
                    help='file with labels (one per line)')
    ap.add_argument('--lr', '--learning_rate', metavar='FLOAT', type=float,
                    default=DEFAULT_LR, help='learning rate')
    ap.add_argument('--num_train_epochs', metavar='INT', type=int, default=1,
                    help='number of training epochs')
    ap.add_argument('--batch_size', metavar='INT', type=int,
                    default=DEFAULT_BATCH_SIZE,
                    help='batch size for training')
    ap.add_argument('--warmup_proportion', metavar='FLOAT', type=float,
                    default=DEFAULT_WARMUP_PROPORTION,
                    help='warmup proportion of training steps')
    ap.add_argument('--train_data', metavar='FILE', required=True,
                    help='training data')
    ap.add_argument('--dev_data', metavar='FILE', required=True,
                    help='development data')
    ap.add_argument('--examples', choices=EXAMPLE_GENERATORS.keys(),
                    default=list(EXAMPLE_GENERATORS.keys())[0],
                    help='example generation strategy')
    ap.add_argument('--summarize_preds', choices=PREDICTION_SUMMARIZERS.keys(),
                    default=list(PREDICTION_SUMMARIZERS.keys())[0],
                    help='prediction summarization strategy')
    ap.add_argument('--assign_labels', choices=LABEL_ASSIGNERS.keys(),
                    default=list(LABEL_ASSIGNERS.keys())[0],
                    help='label assignment strategy')
    ap.add_argument('--separator', default=None,
                    help='CoNLL format field separator')
    ap.add_argument('--output_file', default=None,
                    help='file to write predicted outputs to')
    ap.add_argument('--ner_model_dir', default=None,
                    help='Trained NER model directory')
    ap.add_argument('--cache_dir', default=None,
                    help='transformers cache directory')
    return ap


def check_predictions(documents):
    """Checks that each Token has at least one prediction."""
    tokens, predictions = 0, 0
    for document in documents:
        for sentence in document.sentences:
            for word in sentence.words:
                for token in word.tokens:
                    if not token.predictions:
                        raise ValueError(f'missing predictions for {token}')
                    tokens += 1
                    predictions += len(token.predictions)
    logger.info(f'{predictions} predictions for {tokens} tokens '
                f'({predictions/tokens:.1f} per token)')


def main(argv):
    options = argparser().parse_args(argv[1:])
    logger.info(f'train.py arguments: {options}')

    # word_labels are the labels assigned to words in the original
    # data, token_labeler.labels() the labels assigned to tokens in
    # the tokenized data. The two are differentiated to allow distinct
    # labels to be added e.g. to continuation wordpieces.
    word_labels = load_labels(options.labels)
    token_labeler = Iob2TokenLabeler(word_labels)
    num_labels = len(token_labeler.labels())
    label_encoder = LabelEncoder(token_labeler.labels())

    logger.info('loading pretrained model')
    pretrained_model, tokenizer, config = load_pretrained(
        options.model_name,
        cache_dir=options.cache_dir
    )
    logger.info('pretrained model config:')
    logger.info(config)

    if options.max_seq_length > config.max_position_embeddings:
        raise ValueError(f'--max_seq_length {options.max_seq_length} not '
                         f'supported by model')
    seq_len = options.max_seq_length

    encode_tokens = lambda t: tokenizer.encode(t, add_special_tokens=False)

    document_loader = ConllLoader(
        tokenizer.tokenize,
        token_labeler.label_tokens,
        options.separator
    )

    example_generator = EXAMPLE_GENERATORS[options.examples](
        seq_len,
        Token(tokenizer.cls_token, is_special=True, masked=False),
        Token(tokenizer.sep_token, is_special=True, masked=False),
        Token(tokenizer.pad_token, is_special=True, masked=True),
        encode_tokens,
        label_encoder.encode
    )

    train_documents = document_loader.load(options.train_data)
    dev_documents = document_loader.load(options.dev_data)
    # containers instead of generators for statistics
    train_documents = list(train_documents)
    dev_documents = list(dev_documents)
    log_dataset_statistics('train', train_documents)
    log_dataset_statistics('dev', dev_documents)

    decoder = ViterbiDecoder(label_encoder.label_map)
    decoder.estimate_probabilities(train_documents)
    logger.info(f'init_prob:\n{decoder.init_prob}')
    logger.info(f'trans_prob:\n{decoder.trans_prob}')

    train_examples = example_generator.examples(train_documents)
    dev_examples = example_generator.examples(dev_documents)
    # containers instead of generators for len() and logging
    train_examples = list(train_examples)
    dev_examples = list(dev_examples)
    num_train_examples = len(train_examples)
    log_examples(train_examples, count=2)

    optimizer, lr_schedule = get_optimizer(
        options.lr,
        options.num_train_epochs,
        options.batch_size,
        options.warmup_proportion,
        num_train_examples,
    )

    ner_model = build_ner_model(
        pretrained_model,
        num_labels,
        seq_len
    )
    ner_model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        sample_weight_mode='temporal',    # TODO is this necessary?
        metrics=['sparse_categorical_accuracy']
    )
    logger.info('ner model:')
    ner_model.summary(print_fn=logger.info)

    train_x, train_y = examples_to_inputs(train_examples)
    dev_x, dev_y = examples_to_inputs(dev_examples)

    lr_history = LRHistory(lr_schedule)
    history = ner_model.fit(
        train_x,
        train_y,
        epochs=options.num_train_epochs,
        batch_size=options.batch_size,
        validation_data=(dev_x, dev_y),
        callbacks=[lr_history]
    )
    for k, v in history.history.items():
        logger.info(f'{k} history: {v}')
    logger.info(f'lr history: {lr_history.by_epoch}')

    dev_predictions = ner_model.predict(
        dev_x,
        verbose=1,
        batch_size=options.batch_size
    )
    assert len(dev_examples) == len(dev_predictions)
    for example, preds in zip(dev_examples, dev_predictions):
        assert len(example.tokens) == len(preds)
        for pos, (token, pred) in enumerate(zip(example.tokens, preds)):
            token.predictions.append((pos, pred))

    documents = unique(
        t.document for e in dev_examples for t in e.tokens if not t.is_special
    )
    check_predictions(documents)

    for n, r in evaluate_assign_labels_funcs(documents, label_encoder).items():
        print(f'{n}: prec {r.prec:.2%} rec {r.rec:.2%} f {r.fscore:.2%}')

    summarize_predictions = PREDICTION_SUMMARIZERS[options.summarize_preds]
    assign_labels = LABEL_ASSIGNERS[options.assign_labels]
    for document in documents:
        summarize_predictions(document)
        assign_labels(document, label_encoder)

    for n, r in evaluate_viterbi(documents, decoder.init_prob,
                                 decoder.trans_prob, label_encoder).items():
        print(f'{n}: prec {r.prec:.2%} rec {r.rec:.2%} f {r.fscore:.2%}')

    for document in documents:
        assign_labels(document, label_encoder)    # greedy

    print(conlleval_report(documents))

    if options.output_file is not None:
        with open(options.output_file, 'w') as out:
            write_conll(documents, out=out)

    if options.ner_model_dir is not None:
        save_ner_model(
            options.ner_model_dir,
            ner_model,
            decoder,
            tokenizer,
            word_labels,
            config
        )

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
