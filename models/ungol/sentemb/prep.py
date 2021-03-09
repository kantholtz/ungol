#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  sentence preprocessing
#
#     >> inb4 refactoring...
#

from ungol.common import logger
from ungol.common import util as ucu
from ungol.sentemb import common as usc

from tqdm import tqdm as _tqdm
from nltk.tokenize import sent_tokenize as tok_sent
from nltk.tokenize import word_tokenize as tok_wrd

import pickle
import pathlib
import argparse
import functools
import multiprocessing as mp
from collections import defaultdict

dumpr = ucu.load_module('dumpr.common')


log = logger.get('sentemb.prep')
tqdm = functools.partial(_tqdm, ncols=80)


#
#  use dumpr to produce sentence files
#


class Writer:

    def __init__(self, f_out: str):
        self._f_out = f_out

    def __enter__(self):
        self._fd_out = open(self._f_out, mode='w')
        self._fd_disc = open(self._f_out + '.disc', mode='w')
        return self

    def __exit__(self, *args):
        self._fd_out.close()
        self._fd_disc.close()

    def write(self, sent: str, accepted: bool = True):
        fd = self._fd_out if accepted else self._fd_disc
        fd.write(sent)
        fd.write('\n')


def _filter(sent: str):
    if len(sent) < 1:
        return False

    if sent.startswith('"') and sent[1].isalpha() and sent[1].isupper():
        return True

    if sent[0].isalpha() and sent[0].isupper():
        return True

    return False


def from_dumpr(f_in: str, f_out: str, append: bool = False):
    with dumpr.BatchReader(f_in) as reader, Writer(f_out) as writer:
        for i, doc in tqdm(enumerate(reader)):

            if doc.content is None:
                continue

            for sent in tok_sent(doc.content):
                sent = sent.replace('\n', ' ').strip()
                writer.write(sent, accepted=_filter(sent))


# --------------------


class WorkerActor(usc.Actor):

    def update(self, msg):
        toks = [t.lower() for t in tok_wrd(msg)]
        return tuple(toks)


class WriterActor(usc.Actor):

    def __init__(self, f_out: str):
        self.pth = pathlib.Path(f_out)

    def before(self):
        self.ids = 0
        self.vocab = {}
        self.counts = defaultdict(lambda: 0)
        self.fd_out_arr = (self.pth / usc.F_ARRS).open(mode='w')
        self.fd_out_tok = (self.pth / usc.F_TOKS).open(mode='w')

    def update(self, msg):
        for tok in msg:
            if tok not in self.vocab:
                self.vocab[tok] = self.ids
                self.ids += 1

            self.counts[tok] += 1

        arr = [self.vocab[tok] for tok in msg]
        self.fd_out_tok.write(' '.join(msg) + '\n')
        usc.write_arr(self.fd_out_arr, arr)

    def after(self):
        log.info('persisting vocabulary')

        with(self.pth / usc.F_VOCAB).open(mode='wb') as fd:
            pickle.dump(self.vocab, fd)

        with(self.pth / usc.F_COUNTS).open(mode='wb') as fd:
            pickle.dump(dict(self.counts), fd)

        self.fd_out_arr.close()
        self.fd_out_tok.close()


def _tok_reader(f_in: str, n=None, rq: mp.Queue = None):
    count = 0
    with open(f_in, 'r') as fd_in:
        for line in tqdm(fd_in, desc='reader', position=1):
            rq.put(line)
            count += 1

            if n is not None and count >= n:
                break


def from_file(f_in: str, f_out: str, processes: int):
    print()

    reader = functools.partial(_tok_reader, f_in)
    worker = WorkerActor()
    writer = WriterActor(f_out)

    usc.multiprocess((reader, worker, writer), processes, 1)
    print('done')


# --------------------


def prune(f_in: str, n: int):
    print('loading vocabularies')
    vocab, counts = usc.get_vocabs(f_in)

    print('building list of undesired tokens')
    undesired = set(tok for tok in counts if counts[tok] <= n)

    _perc = int(len(undesired) / len(counts) * 100)
    print(f'pruning {len(undesired)}/{len(counts)} ({_perc}%) tokens')

    c_total = 0
    c_short = 0
    c_long = 0

    prefix = f'pruned-{n}.'

    pth = pathlib.Path(f_in)
    fd_in = open(pth / usc.F_ARRS, mode='r')
    fd_out_arr = open(pth / (prefix + usc.F_ARRS), mode='w')
    fd_out_tok = open(pth / (prefix + usc.F_TOKS), mode='w')

    lookup = {v: k for k, v in vocab.items()}

    for arr in tqdm(usc.read_arr_file(fd_in)):
        c_total += 1
        pruned_arr = [idx for idx in arr if lookup[idx] not in undesired]

        if len(pruned_arr) < usc.SENT_MIN_LEN:
            c_short += 1
        elif len(pruned_arr) > usc.SENT_MAX_LEN:
            c_long += 1
        else:
            usc.write_arr(fd_out_arr, pruned_arr)

            token = [lookup[idx] for idx in pruned_arr]
            fd_out_tok.write(' '.join(token) + '\n')

    fd_in.close()
    fd_out_arr.close()
    fd_out_tok.close()

    for tok in undesired:
        del vocab[tok]
        del counts[tok]

    with(pth / (prefix + usc.F_VOCAB)).open(mode='wb') as fd:
        pickle.dump(vocab, fd)

    with(pth / (prefix + usc.F_COUNTS)).open(mode='wb') as fd:
        pickle.dump(dict(counts), fd)

    _perc = int((c_short + c_long) / c_total * 100)
    print(f'pruned {c_short + c_long} sentences ({_perc}%)')
    print(f'  too short: {c_short}\n  too long: {c_long}')

# --------------------


def main(args):
    # i know, misuse of assert :)

    if args.command == 'create-file':
        assert args.dumpr, 'provide --dumpr'
        assert args.out, 'provide --out'
        assert dumpr, 'install dumpr first'
        from_dumpr(args.dumpr, args.out)

    elif args.command == 'from-file':
        assert args.file, 'provide --file'
        assert args.out, 'provide --out'
        assert args.processes, 'provide --processes'
        from_file(args.file, args.out, processes=args.processes)

    elif args.command == 'prune':
        assert args.folder, 'provide --folder'
        assert args.threshold, 'provide --threshold'
        prune(args.folder, args.threshold)

    else:
        raise Exception('command not recognized')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'command', type=str,
        help='one of create-file|from-file|prune')

    parser.add_argument(
        '--out', type=str,
        help='output file or directory')

    parser.add_argument(
        '--dumpr', type=str,
        help='use dumpr to create a file with sentences')

    parser.add_argument(
        '--file', type=str,
        help='use a file with sentences to build a training set')

    parser.add_argument(
        '--folder', type=str,
        help='folder containing vocabularies, arrays etc.')

    parser.add_argument(
        '--threshold', type=int,
        help='for "prune", minimum occurence count of a word')

    parser.add_argument(
        '--processes', type=int, default=5,
        help='number of processes to use')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
