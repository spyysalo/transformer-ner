"""
Evaluation support functionality.
"""

from io import StringIO
from conlleval import evaluate, report, metrics

from data import write_conll
from collections import OrderedDict


def conlleval_evaluate(documents):
    """Return conlleval evaluation results for Documents as counts."""
    # conlleval.py has a file-based API, so use StringIO
    conll_string = StringIO()
    write_conll(documents, out=conll_string)
    conll_string.seek(0)
    return evaluate(conll_string)


def conlleval_report(documents):
    """Return conlleval evaluation report for Documents as string."""
    # conlleval.py has a file-based API, so use StringIO
    counts = conlleval_evaluate(documents)
    report_string = StringIO()
    report(counts, out=report_string)
    return report_string.getvalue()


def conlleval_overall_results(documents):
    """Return overall conlleval results for Documents."""
    counts = conlleval_evaluate(documents)
    overall, by_type = metrics(counts)
    return overall


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


def evaluate_assign_labels_funcs(documents, label_encoder):
    from label import LABEL_ASSIGNERS

    results = OrderedDict()
    for name, assign_labels in LABEL_ASSIGNERS.items():
        for document in documents:
            assign_labels(document, label_encoder)
        results[name] = conlleval_overall_results(documents)
    return results
