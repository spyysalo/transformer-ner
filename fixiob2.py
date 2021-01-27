#!/usr/bin/env python

# Fixes errors in IOB2 sequences in CoNLL-style column-formatted
# data.

import sys
import re


EMPTY_LINE_RE = re.compile(r'^\s*$')

IOB_TAG_RE = re.compile(r'^([IOB])((?:-\S+)?)$')


def argparser():
    import argparse
    ap = argparse.ArgumentParser(
        description='Fix B-I-O sequence errors in CoNLL-style data.'
    )
    ap.add_argument(
        '-m', '--mode', default='last', choices=('first', 'last', 'split'),
        help='How to resolve BI+ sequences with multiple types'
    )
    ap.add_argument(
        '-i', '--indices', default=None, 
        help='Indices of fields to fix (comma-separated)'
    )
    ap.add_argument(
        '-v', '--verbose', default=False, action='store_true',
        help='Verbose output.'
    )
    ap.add_argument(
        'files', nargs='+',
        help='Target file(s) ("-" for STDIN)'
    )
    return ap


class ParseError(Exception):
    def __init__(self, line, linenum, message=None, filename=None):
        self.line = line
        self.linenum = linenum
        self.message = message
        self.file = file

        if self.message is None:
            self.message = "Parse error"

    def __str__(self):
        return (self.message +
                ("on line %d" % self.linenum) + 
                ("" if self.file is None else " in file %s" % self.file) +
                (": '%s'" % self.line))


def parse_IOB_tag(tag):
    """Parse given string as IOB tag.

    The expected format is "[IOB]-TYPE", where TYPE is any non-empty
    nonspace string, and the "-TYPE" part is optional.

    Args:
        tag (string): tag to parse.
    Returns:
        string pair: tag ("B", "I" or "O") and TYPE.
    """

    m = re.match(r'^([IOB])((?:-\S+)?)$', tag)
    assert m, "ERROR: failed to parse tag '%s'" % tag
    ttag, ttype = m.groups()

    # Strip off starting "-" from tagged type, if any.
    if len(ttype) > 0 and ttype[0] == "-":
        ttype = ttype[1:]

    return ttag, ttype


def make_IOB_tag(ttag, ttype):
    """Inverse of parse_IOB_tag."""

    if ttype is None:
        return ttag
    else:
        return ttag+'-'+ttype


def is_IOB_tag(s):
    """Return True if given string is a valid IOB tag."""
    return IOB_TAG_RE.match(s)


def is_docstart(s):
    return s == '-DOCSTART-'


def IOB_indices(blocks):
    """Return indices of fields containing IOB tags.

    Expects output of parse_conll() (or similar) as input.

    Args:
        blocks (list of lists of lists of strings): parsed CoNLL-style input.
    Returns:
        list of integers: indices of valid IOB tags in data.
    """

    valid = None
    for block in blocks:
        for line in block:
            # Initialize candidates on first non-empty
            if valid is None:
                valid = range(len(line))

            valid = [
                i for i in valid if i < len(line) and 
                is_IOB_tag(line[i]) or is_docstart(line[0])
            ]

            # Short-circuit
            if len(valid) == 0:
                return valid

    if valid is None:
        return []

    return valid


def _fix_IOB_index(blocks, index, mode, verbose):
    """Implement fix_IOB() for single index."""
    # Fix errors where non-"O" sequence begins with "I" instead of "B"
    for block in blocks:
        prev_tag = None
        for line in block:
            if is_docstart(line[0]):
                continue

            ttag, ttype = parse_IOB_tag(line[index])

            if (prev_tag is None or prev_tag == "O") and ttag == "I":
                if verbose:
                    print("Rewriting initial \"I\" -> \"B\" (%s)" % ttype,
                          file=sys.stderr)
                line[index] = make_IOB_tag("B", ttype)

            prev_tag = ttag

    # Fix errors where type changes without a "B" at the boundary
    for block in blocks:
        prev_tag, prev_type = None, None
        for ln, line in enumerate(block):
            if is_docstart(line[0]):
                continue

            ttag, ttype = parse_IOB_tag(line[index])

            if prev_tag in ("B", "I") and  ttag == "I" and prev_type != ttype:

                if mode == 'first':
                    # Propagate first type to whole sequence
                    if verbose:
                        print("Rewriting multi-type sequence to first type (%s->%s)" % (ttype, prev_type), file=sys.stderr)
                    i = ln
                    while i < len(block):
                        itag, itype = parse_IOB_tag(block[i][index])
                        if itag != "I":
                            break
                        block[i][index] = make_IOB_tag(itag, prev_type)
                        i += 1
                    # Current was changed
                    ttype = prev_type

                elif mode == 'last':
                    # Propagate last type to whole sequence
                    if verbose:
                        print("Rewriting multi-type sequence to last type (%s->%s)" % (prev_type, ttype), file=sys.stderr)
                    i = ln - 1
                    while i >= 0:
                        itag, itype = parse_IOB_tag(block[i][index])
                        if itag not in ("B", "I"):
                            break
                        block[i][index] = make_IOB_tag(itag, ttype)
                        i -= 1

                elif mode == 'split':
                    # Split sequence
                    if verbose:
                        print("Rewriting \"I\" -> \"B\" to split at type switch (%s->%s)" % (prev_type, ttype), file=sys.stderr)
                    line[index] = make_IOB_tag("B", ttype)

                else:
                    assert False, "INTERNAL ERROR"
            
            prev_tag, prev_type = ttag, ttype

    return blocks


def fix_IOB(blocks, indices, mode, verbose):
    """Corrects IOB tag sequence errors in given data.

    Expects output of parse_conll() (or similar) as input.
    NOTE: Modifies given blocks.

    Args:
        blocks (list of lists of lists of strings): parsed CoNLL-style input.
        indices (list of ints): indices of fields containing IOB tags.
    Returns:
        given blocks with fixed IOB tag sequence.
    """

    assert len(indices) > 0, "Error: fix_IOB() given empty indices"

    for i in indices:
        blocks = _fix_IOB_index(blocks, i, mode, verbose)

    return blocks


def _line_is_empty(l):
    return EMPTY_LINE_RE.match(l)


def parse_conll(stream, filename=None, separator='\t', is_empty=_line_is_empty):
    """Parse CoNLL-style input.

    Input should consist of blocks of lines separated by empty lines
    (is_empty), each non-empty line consisting of fields separated by
    the given separator.

    Returns:
        list of lists of lists: blocks, lines, fields.
    """

    li, l = 0, None
    try:
        blocks = []
        current_block = []
        for l in stream:
            l = l.rstrip()
            li += 1
            if is_empty(l):
                blocks.append(current_block)
                current_block = []
            else:
                current_block.append(l.split(separator))
    except Exception:
        # whatever goes wrong
        raise ParseError(l, li)

    return blocks


def process_stream(stream, indices, mode, verbose, out=None):
    blocks = parse_conll(stream)

    if indices is None:
        # Fix all valid unless specific indices given
        indices = IOB_indices(blocks)
    assert len(indices) > 0, "Error: no valid IOB fields"

    if out is None:
        out = sys.stdout

    blocks = fix_IOB(blocks, indices, mode, verbose)

    # Output
    for block in blocks:
        for line in block:
            print('\t'.join(line), file=out)
        print(file=out)


def process_file(fn, indices, mode, verbose):
    with open(fn, 'r') as f:
        return process_stream(f, indices, mode, verbose)

        
def main(argv):
    options = argparser().parse_args(argv[1:])

    # Resolve indices to fix
    if options.indices is None:
        indices = None
    else:
        try:
            indices = [int(i) for i in options.indices.split(",")]
        except Exception:
            raise ValueError('Argument "-i" value should be a comma-separated list of integers')

    # Primary processing
    for fn in options.files:
        try:
            if fn == '-':
                # Special case to read STDIN
                process = process_stream
                fn = sys.stdin
            else:
                process = process_file
            process(fn, indices, options.mode, options.verbose)
        except Exception:            
            print('Error processing %s' % fn, file=sys.stderr)
            raise

    return 0


if __name__ == '__main__':
    try:
        sys.exit(main(sys.argv))
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)
