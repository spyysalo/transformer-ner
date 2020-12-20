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


def viterbi_probabilities(sentence_labels, tag_map, lambda_=0.001):
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
                error(f'ERROR: {e} {tag_map}')

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


class ViterbiDecoder:
    def __init__(self, label_map, init_prob=None, trans_prob=None):
        self.label_map = { 
            k: v for k, v in label_map.items() if k is not None
        }
        self.init_prob = init_prob
        self.trans_prob = trans_prob

    def estimate_probabilities(self, documents, lambda_=0.001):
        """Estimate initial and transition probabilities from data."""
        sentence_labels = []
        for document in documents:
            for sentence in document.sentences:
                labels = []
                for word in sentence.words:
                    labels.extend([t.label for t in word.tokens])
                sentence_labels.append(labels)
        init, trans =  viterbi_probabilities(
            sentence_labels, self.label_map, lambda_)
        self.init_prob = init
        self.trans_prob = trans

    def viterbi_path(self, cond_prob, weight=1):
        return viterbi_path(self.init_prob, self.trans_prob, cond_prob, weight)

    def save(self, path):
        # Map numpy arrays to dictionaries
        inv_label_map = { 
            v: k for k, v in self.label_map.items() if k is not None
        }
        init_prob = { 
            inv_label_map[i]: v for i, v in enumerate(self.init_prob)
        }
        trans_prob = {
            inv_label_map[i]: { inv_label_map[j]: v for j, v in enumerate(p) }
            for i, p in enumerate(self.trans_prob)
        }
        data = {
            'initial': init_prob,
            'transition': trans_prob,
            'labels': self.label_map
        }
        with open(path, 'w') as out:
            json.dump(data, out, indent=4)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        label_map = data['labels']
        init_prob = data['initial']
        trans_prob = data['transition']

        # Map dictionaries to numpy arrays
        init_prob = _label_dict_to_array(init_prob, label_map)
        trans_prob = {
            k: _label_dict_to_array(v, label_map)
            for k, v in trans_prob.items()
        }
        trans_prob = _label_dict_to_array(trans_prob, label_map)
        
        return cls(label_map, init_prob, trans_prob)
