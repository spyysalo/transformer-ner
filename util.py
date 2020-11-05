import sys
import logging

from collections import Counter
from functools import wraps
from time import time


# logger setup
logging.basicConfig()
logger = logging.getLogger('ner')
logger.setLevel(logging.INFO)


def timed(f, out=sys.stderr):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        print('{} completed in {:.1f} sec'.format(f.__name__, time()-start),
              file=out)
        return result
    return wrapper


def unique(sequence):
    """Return unique items in sequence, preserving order."""
    # https://www.peterbe.com/plog/fastest-way-to-uniquify-a-list-in-python-3.6
    return list(dict.fromkeys(sequence))


def most_common(sequence):
    """Return most common item in sequence."""
    return Counter(sequence).most_common(1)[0][0]


def log_examples(examples, count=1, log=logger.info):
    for i, e in enumerate(examples):
        if i >= count:
            break
        log(f'example {i}:')
        log(e)


def log_dataset_statistics(name, documents, log=logger.info):
    sents = sum(d.sentence_count() for d in documents)
    words = sum(d.word_count() for d in documents)
    tokens = sum(d.token_count() for d in documents)
    log(f'{name}: {len(documents)} docs, {sents} sentences, '
        f'{words} words, {tokens} tokens')
