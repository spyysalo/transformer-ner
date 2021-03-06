import sys

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from model import load_ner_model
from data import ConllLoader, Token, PREDICTION_SUMMARIZERS
from data import write_conll
# from label import LabelEncoder, Iob2TokenLabeler, LABEL_ASSIGNERS
from label import LabelEncoder, IobesTokenLabeler, LABEL_ASSIGNERS
from label import iobes_to_iob2
from example import EXAMPLE_GENERATORS, examples_to_inputs
from util import logger, unique
from evaluation import conlleval_report


def argparser():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument('--ner_model_dir', default=None, required=True,
                    help='Trained NER model directory')
    ap.add_argument('--data', default=None, required=True,
                    help='Test data')
    ap.add_argument('--separator', default=None,
                    help='CoNLL format field separator')
    return ap


def main(argv):
    options = argparser().parse_args(argv[1:])

    ner_model, decoder, tokenizer, word_labels, config = load_ner_model(
        options.ner_model_dir)

    token_labeler = IobesTokenLabeler(word_labels)
    label_encoder = LabelEncoder(token_labeler.labels())

    encode_tokens = lambda t: tokenizer.encode(t, add_special_tokens=False)

    document_loader = ConllLoader(
        tokenizer.tokenize,
        token_labeler.label_tokens,
        options.separator,
        #test=True
    )

    example_generator = 'wrap'    # TODO read from config
    seq_len = 512    # TODO read from config
    example_generator = EXAMPLE_GENERATORS[example_generator](
        seq_len,
        Token(tokenizer.cls_token, is_special=True, masked=False),
        Token(tokenizer.sep_token, is_special=True, masked=False),
        Token(tokenizer.pad_token, is_special=True, masked=True),
        encode_tokens,
        label_encoder.encode
    )

    test_documents = document_loader.load(options.data)
    test_examples = example_generator.examples(test_documents)
    test_examples = list(test_examples)    # TODO stream
    test_x, test_y = examples_to_inputs(test_examples)

    test_predictions = ner_model.predict(test_x)
    for example, preds in zip(test_examples, test_predictions):
        assert len(example.tokens) == len(preds)
        for pos, (token, pred) in enumerate(zip(example.tokens, preds)):
            token.predictions.append((pos, pred))

    documents = unique(
        t.document for e in test_examples for t in e.tokens if not t.is_special
    )

    summarize_preds = 'avg'    # TODO read from config
    assign_labels = 'first'    # TODO read from config
    summarize_predictions = PREDICTION_SUMMARIZERS[summarize_preds]
    assign_labels = LABEL_ASSIGNERS[assign_labels]

    for document in documents:
        summarize_predictions(document)
        assign_labels(document, label_encoder)

    with open('greedy.tsv', 'w') as out:
        write_conll(documents, out=out)

    print(conlleval_report(documents))

    for document in documents:
        for sentence in document.sentences:
            tokens = [t for w in sentence.words for t in w.tokens]
            cond_prob = [t.pred_summary for t in tokens]
            path = decoder.viterbi_path(cond_prob, weight=4)
            assert len(path) == len(tokens)
            for idx, token in zip(path, tokens):
                label = label_encoder.inv_label_map[idx]
                label = iobes_to_iob2(label)
                token.viterbi_label = label
            for word in sentence.words:
                word.predicted_label = word.tokens[0].viterbi_label

    with open('viterbi.tsv', 'w') as out:
        write_conll(documents, out=out)

    print(conlleval_report(documents))


if __name__ == '__main__':
    sys.exit(main(sys.argv))
