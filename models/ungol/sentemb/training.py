#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  create datasets for training embcompr
#  (processes files as produced by prep.py)
#


from ungol.common import logger
from ungol.common import embed as uce
from ungol.sentemb import common as usc
from ungol.sentemb import redux as usr

import attr
import h5py
from tqdm import tqdm as _tqdm

import pickle
import pathlib
import argparse
import functools

from typing import Set
from typing import Dict

log = logger.get('sentemb.training')
tqdm = functools.partial(_tqdm, ncols=80)


# FIXME: make an option
BATCH_SIZE = 2048


def btqdm(*args, step: int = 1, **kwargs):
    bar = tqdm(*args, **kwargs)
    for x in bar:
        yield x
        bar.update(step)


@attr.s
class Stats:

    count:      int = attr.ib(default=0)
    pruned:     int = attr.ib(default=0)
    # lens: List[int] = attr.ib(default=attr.Factory(list))

    def __str__(self):
        # lens = np.array(self.lens)
        s = (
            'statistics:',
            f'  total sentences: {self.count}',
            f'  pruned: {self.pruned}', )
        #   f'  lengths: min={lens.min()}, max={lens.max()}',
        #   f'  average: {lens.mean():.3f} (σ={lens.std():.3f})', )

        return '\n'.join(map(str, s))


# ---


def _sent_filter(idxs):
    return len(idxs) >= usc.SENT_MIN_LEN


def _generate_sentence_batches(
        f_in: str, prefix: str,
        s_common: Set[str],
        batch_size: int = None):

    gen = usc.read_toks_batches(
        f_in, prefix=prefix,
        batch_size=batch_size)  # , n=2e5):

    for batch in gen:
        pruned = [[t for t in toks if t in s_common] for toks in batch]
        yield [s for s in pruned if _sent_filter(s)]


# copy on write for 'sentences'
def Reader(creator, prune: bool = True, batch_size: int = None, rq=None):
    pos = 0

    gen_args = creator.f_in, creator.prefix, creator.common
    gen = _generate_sentence_batches(*gen_args, batch_size=BATCH_SIZE)
    progress = btqdm(gen, desc='reader', position=1, step=BATCH_SIZE)

    for batch in progress:
        mapped = creator.redux.map_batch(batch)
        rq.put((pos, batch, mapped))
        pos += 1


class WorkerActor(usc.Actor):
    """
    Simply a proxy class for a concrete
    redux.Redux implementation
    """

    # invoked by main process
    def __init__(self, creator):
        self.creator = creator

    # invoked in own process
    def update(self, msg):
        pos, batch, mapped = msg
        sentences = [' '.join(toks) for toks in batch]
        reduced = self.creator.redux.reduce_batch(mapped)
        return pos, sentences, reduced


# invoked in main process
class WriterActor(usc.Actor):

    @property
    def redux(self) -> usr.Redux:
        return self._redux

    def __init__(self, creator):
        if creator.prefix is None:
            f_name = f'{creator.redux.name}.'
        else:
            f_name = f'{creator.prefix}.{creator.redux.name}.'

        pth_out = pathlib.Path(creator.f_out)
        pth_txt = (pth_out / (f_name + 'sentences.txt'))
        self.fd_sents = pth_txt.open(mode='w')

        pth_h5 = pth_out / (f_name + 'embedding.h5')
        amount = creator.stats.count

        shape = amount, creator.dimensions

        log.info('writer: opening file handles')
        self.fd_h5 = h5py.File(str(pth_h5), mode='w')
        self.ds = self.fd_h5.create_dataset('embedding', shape)

        self._position = 0
        self._redux = creator.redux

    def update(self, msg):
        _, sentences, vecs = msg
        lower = self._position
        upper = lower + len(vecs)

        self.ds[lower:upper] = vecs
        for sentence in sentences:
            self.fd_sents.write(sentence + '\n')

        self._position = upper

    def after(self):
        self.redux.finish(self.ds)

        log.info('writer: closing file handles')
        self.fd_sents.close()
        self.fd_h5.close()


@attr.s
class Creator:

    f_in:    str = attr.ib()
    f_out:   str = attr.ib()
    f_redux: str = attr.ib()

    # optional

    embed: uce.Embed = attr.ib(default=None)

    processes:   int = attr.ib(default=4)
    prefix:      str = attr.ib(default=None)  # usually 'pruned-10' or the like

    # upstream specific parameters

    sif_alpha: float = attr.ib(default=None)  # redux: SIF (10^-3 - 10^-4)
    f_sent2vec:  str = attr.ib(default=None)  # redux: Sent2Vec (.bin)
    f_infersent: str = attr.ib(default=None)  # redux: Infersent (.pickle)

    # initialized in post-init

    @property
    def redux(self) -> usr.Redux:
        return self._redux

    @property
    def vocab(self) -> Dict[str, int]:
        """
        string to ungol index
        """
        return self._vocab

    @property
    def counts(self) -> Dict[str, int]:
        """
        string to term count
        """
        return self._counts

    @property
    def lookup(self) -> Dict[int, str]:
        """
        ungol index to string
        """
        return self._lookup

    @property
    def common(self) -> Set[str]:
        return self._common

    @property
    def stats(self) -> Stats:
        return self._stats

    @property
    def dimensions(self) -> int:
        if self.f_infersent:
            return self.redux.DIMS

        if self.embed:
            return self.embed.dimensions

        if self.f_sent2vec:
            return self.redux.model.get_emb_size()

    # initialization

    def __attrs_post_init__(self):
        self._init_vocab()
        self._init_stats()
        self._init_redux()

    def _init_vocab(self):
        print('loading vocabularies')
        self._vocab, self._counts = usc.get_vocabs(
            self.f_in, prefix=self.prefix)

        print(f'\nread vocabulary of size {len(self.vocab)}')

        self._lookup = {v: k for k, v in self.vocab.items()}
        self._common = set(self.vocab.keys())
        if self.embed:
            self._common &= set(self.embed.vocab.keys())
            print(f'embedding:   {len(self.embed)} embedding vectors')

        print(f'vocabulary:  {len(self.vocab)} training tokens')
        print(f'common:      {len(self.common)}\n')

    def _init_stats(self):
        print('gathering file statistics')
        self._stats = Stats()

        prune = True if self.embed else False

        gen_args = self.f_in, self.prefix, self.common
        gen = _generate_sentence_batches(*gen_args, batch_size=BATCH_SIZE)

        for batch in btqdm(gen, step=BATCH_SIZE):
            if prune:
                batch = [s for s in batch if _sent_filter(s)]

            self.stats.pruned += BATCH_SIZE - len(batch)
            self.stats.count += len(batch)

        print(f'will transform {self.stats.count} samples')
        return self.stats

    def _init_redux(self):

        if self.f_redux == 'bow':
            assert self.embed, 'provide a word embedding'
            self._redux = usr.BoW(self.embed)

        elif self.f_redux == 'mbow':
            assert self.embed, 'provide a word embedding'
            self._redux = usr.MBoW(self.embed)

        # elif self.f_redux == 'wmbow':
        #     assert self.embed, 'provide a word embedding'
        #     self._redux = usr.WMBow(self.counts)

        elif self.f_redux == 'sif':
            assert self.sif_alpha, 'provide alpha value for SIF'
            self._redux = usr.SIF(self.counts, self.sif_alpha)

        elif self.f_redux == 'sent2vec':
            assert self.f_sent2vec, 'provide a Sent2Vec model'
            print('loading sent2vec model')
            self._redux = usr.Sent2Vec(self.f_sent2vec)

        elif self.f_redux == 'infersent':
            assert self.f_infersent, 'provide an InferSent Model'
            assert self.embed, 'provide a word embedding'
            print('loading infersent model')
            self._redux = usr.Infersent(self.f_infersent, self.embed)

        else:
            raise Exception(f'unknown redux "{self.f_redux}"')

    # interface

    def run(self):
        print('creating training data set')

        prune = True if self.embed else False

        reader = functools.partial(Reader, self, prune=prune)
        worker = WorkerActor(self)
        writer = WriterActor(self)

        usc.multiprocess((reader, worker, writer), self.processes, BATCH_SIZE)
        print('done')

    # factories

    @staticmethod
    def from_args(args: argparse.Namespace):

        assert args.folder, 'provide --folder'
        assert args.out, 'provide --out'
        assert args.redux, 'provide --redux'

        print('loading embeddings')
        embed = None
        if args.embed_provider:
            embed = uce.create(uce.Config.from_args(args))

        print('initialize creator')
        return Creator(
            f_in=args.folder,
            f_out=args.out,
            f_redux=args.redux,
            embed=embed,
            prefix=args.prefix,
            processes=args.processes,
            sif_alpha=args.sif_alpha,
            f_sent2vec=args.sent2vec_model,
            f_infersent=args.infersent_model, )


# ---


def create_vocab(f_in: str):
    assert f_in.endswith('sentences.txt')
    f_out = f_in.replace('sentences.txt', 'vocab.pickle')

    duplicates = 0

    vocab = {}
    with open(f_in, mode='r') as fd_in:
        for i, line in tqdm(enumerate(fd_in)):
            line = line.strip()

            # handle duplicate sentences
            #
            # This might happen under the following circumstances: if
            # a word is removed when creating the training set because
            # it cannot be found in the word embedding space,
            # sentences might turn to duplicates. To detect and
            # eliminate duplicates, the whole set of sentences must be
            # loaded into ram (if not using any database system) and
            # this is not always possible - hence this little
            # work-around:
            if line in vocab:
                duplicates += 1
                line = f'[{duplicates}] {line}'

            vocab[line] = i

    print(f'encountered {duplicates} duplicate sentences')
    print(f'writing "{f_out}"')
    with open(f_out, mode='wb') as fd_out:
        pickle.dump(vocab, fd_out)


def main(args):
    if args.cmd == 'create':
        creator = Creator.from_args(args)
        creator.run()

    elif args.cmd == 'vocab':
        assert args.file, 'provide --file'
        create_vocab(args.file)

    else:
        print(f'unknown command "{args.cmd}"')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'cmd', type=str,
        help='one of (create|vocab)')

    parser.add_argument(
        '--prefix', type=str,
        help='for subsets of the whole data (e.g. "pruned-10")')

    parser.add_argument(
        '--folder', type=str,
        help='folder containing vocabularies, arrays etc.')

    parser.add_argument(
        '--redux', type=str,
        help='one of (bow|mbow|sif|sent2vec)')

    parser.add_argument(
        '--file', type=str,
        help='input file')

    parser.add_argument(
        '--out', type=str,
        help='output file')

    parser.add_argument(
        '--processes', type=int, default=4,
        help='number of processors to use')

    # redux specifics

    parser.add_argument(
        '--sif-alpha', type=float, default=None,
        help='free weight parameter')

    parser.add_argument(
        '--sent2vec-model', type=str, default=None,
        help='path to a pre-trained sent2vec model')

    parser.add_argument(
        '--infersent-model', type=str, default=None,
        help='path to a pre-trained infersent model')

    uce.add_parser_args(parser)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
