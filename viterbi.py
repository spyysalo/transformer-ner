import json
import numpy as np

from itertools import tee
from logging import error


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    # from https://docs.python.org/3/library/itertools.html
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def _label_dict_to_array(label_dict, tag_map):
    """Map values in dict keyed by labels to numpy array in tag_map order.
    >>> _label_dict_to_array({'O': 0.3, 'B': 0.7 }, { 'O': 0, 'B': 1 })
    array([0.3, 0.7])
    """
    idx_dict = { tag_map[k]: v for k, v in label_dict.items() }
    return np.array([v for k, v in sorted(idx_dict.items())])


def save_viterbi_probabilities(init_prob, trans_prob, inv_tag_map, path):
    # Map numpy arrays to dictionaries
    init_prob = { inv_tag_map[i]: v for i, v in enumerate(init_prob) }
    trans_prob = {
        inv_tag_map[i]: { inv_tag_map[j]: v for j, v in enumerate(p) }
        for i, p in enumerate(trans_prob)
    }
    probs = {
        'initial': init_prob,
        'transition': trans_prob,
    }
    with open(path, 'w') as out:
        json.dump(probs, out, indent=4)


def load_viterbi_probabilities(path, tag_map):
    with open(path) as f:
        probs = json.load(f)
    init_prob, trans_prob = probs['initial'], probs['transition']

    # Map dictionaries to numpy arrays
    init_prob = _label_dict_to_array(init_prob, tag_map)
    trans_prob = {
        k: _label_dict_to_array(v, tag_map) for k, v in trans_prob.items()
    }
    trans_prob = _label_dict_to_array(trans_prob, tag_map)
    return init_prob, trans_prob


def viterbi_probabilities(documents, tag_map, lambda_=0.001):
    """Return initial state and transition probabilities estimated
    from Token labels."""
    sentence_labels = []
    for document in documents:
        for sentence in document.sentences:
            labels = []
            for word in sentence.words:
                labels.extend([t.label for t in word.tokens])
            sentence_labels.append(labels)
    return _viterbi_probabilities(sentence_labels, tag_map, lambda_)


def _viterbi_probabilities(sentence_labels, tag_map, lambda_=0.001):
    """Return initial state and transition probabilities estimated
    from given list of lists of labels."""

    num_labels = len([k for k in tag_map if k is not None])
    init_count = np.zeros(num_labels) + lambda_
    trans_count = np.zeros((num_labels, num_labels)) + lambda_

    for labels in sentence_labels:
        init_count[tag_map[labels[0]]] += 1
        for prev, curr in pairwise(labels):
            try:
                trans_count[tag_map[prev],tag_map[curr]] += 1
            except Exception as e:
                error(f'VERROR: {e} {tag_map}')

    init_prob = init_count / np.sum(init_count)
    trans_prob = []
    for l_count in trans_count:
        total = np.sum(l_count)
        if total:
            l_prob = l_count/total
        else:
            # TODO warn?
            l_prob = np.ones(num_labels)/num_labels
        trans_prob.append(l_prob)
    trans_prob = np.array(trans_prob)

    return init_prob, trans_prob


def viterbi_path(init_prob, trans_prob, cond_prob, weight=1):
    # Calculate viterbi path for given initial, transition, and conditional
    # probabilities. Operates in log-space to avoid underflow.
    init_prob = np.log(init_prob)
    trans_prob = np.log(trans_prob)
    cond_prob = np.log(cond_prob)

    seq_length, num_states = cond_prob.shape
    prob = np.zeros((seq_length, num_states))
    prev = np.zeros((seq_length, num_states), dtype=int)

    for s in range(num_states):
        prob[0,s] = init_prob[s] + cond_prob[0,s]

    for t in range(1, seq_length):
        for s in range(num_states):
            p_probs = prob[t-1,:] + trans_prob[:,s]
            p = np.argmax(p_probs)
            prob[t,s] = p_probs[p] + cond_prob[t,s] * weight
            prev[t,s] = p

    path = [np.argmax(prob[seq_length-1,:])]
    for t in reversed(range(seq_length-1)):
        path.append(prev[t+1,path[-1]])
    return list(reversed(path))
