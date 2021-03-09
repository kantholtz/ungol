# -*- coding: utf-8 -*-

import math
import pickle
import pathlib
import argparse

import attr
import h5py
import torch
import configobj
import numpy as np

from typing import Dict
from typing import Tuple
from typing import Generator

from ungol.common import logger
from ungol.common import util as ucu


log = logger.get('common.embed')


class Embed:
    """
    The embedding provider interface to be implemented.

    """

    CHUNK_SIZE = 8192

    @property
    def name(self) -> str:
        raise NotImplementedError()

    @property
    def shape(self) -> Tuple[int, int]:
        # not using self.whole.shape to avoid loading the whole
        # embedding space for implementations with lazy loading
        return len(self), self.dimensions

    @property
    def lookup(self) -> Dict[int, str]:
        """
        Returns an index to string mapping
        """
        try:
            return self._lookup
        except AttributeError:
            self._lookup = {v: k for k, v in self.vocab.items()}
            return self.lookup

    def _chunks_from_sliceable(self, a, size):
        lower = 0
        upper = size

        while lower < len(a):
            yield a[lower:upper]
            lower = upper
            upper = lower + size

    # ---

    def __len__(self) -> int:
        return len(self.vocab)

    def __str__(self) -> str:
        n, dims = self.shape
        return f'ungol embedding provider\n  {n} words\n  {dims} dimensions'

    def __getitem__(self, key: str) -> np.array:
        """

        Get embedding vector either by word or index.

        """
        try:
            return self.whole[self._vocab[key]]
        except (KeyError, TypeError):
            return self.whole[key]

        raise KeyError(f'"{key}" is not a proper embedding selector')

    # --- to be implemented

    @property
    def dimensions(self) -> int:
        raise NotImplementedError()

    @property
    def vocab(self) -> Dict[str, int]:
        """
        Returns an word to index mapping
        """
        raise NotImplementedError()

    @property
    def whole(self) -> np.ndarray:
        """
        Returns a tensor with all embeddings
        """
        raise NotImplementedError()

    def chunks(self, size: int) -> Generator[np.ndarray, None, None]:
        """
        Yields embedding batches

        Args:
          size: batch size

        Returns:
          A generator of embedding batches
        """
        raise NotImplementedError()


#
#  to work with h5py files
#
class H5PYEmbed(Embed):
    """

    Currently "dumb" implementation; use to implement lazy
    loading later.

    """

    @property
    def name(self) -> str:
        return self._name

    @property
    def dimensions(self) -> int:
        return self._ds.shape[1]

    @property
    def vocab(self) -> Dict[str, int]:
        return self._vocab

    @property
    def whole(self) -> np.ndarray:
        return self._ds

    def chunks(self, size: int = Embed.CHUNK_SIZE):
        return self._chunks_from_sliceable(self._ds, size)

    def __len__(self) -> int:
        return self.whole.shape[0]

    # ---

    def __init__(self, f_h5: str, f_vocab: str = None):
        super().__init__()
        self._name = pathlib.Path(f_h5).stem

        # not using lazy loading atm because of
        # problems with advanced indexing
        fd = h5py.File(f_h5, mode='r')
        self._ds = fd['embedding']

        if f_vocab is not None:
            with open(f_vocab, mode='rb') as fd:
                self._vocab = pickle.load(fd)

            assert len(self.vocab) == self._ds.shape[0]

        else:
            self._vocab = None


#
#  to work with text files
#
class FileEmbed(Embed):
    """

    Expects one text file with the following format:

    word1 1.231 2.3452 ...
    word2 0.231 0.1232 ...
    ...

    """

    @property
    def name(self) -> str:
        return self._name

    @property
    def dimensions(self) -> int:
        return self._buf.shape[1]

    @property
    def vocab(self) -> Dict[str, int]:
        return self._vocab.copy()

    @property
    def whole(self) -> np.ndarray:
        return self._buf.copy()

    def chunks(self, size: int = Embed.CHUNK_SIZE):
        return self._chunks_from_sliceable(self.whole, size)

    # ---

    def _map_to_np(self, line):
        word, raw = line.split(' ', maxsplit=1)
        vec = map(float, raw.strip().split(' '))
        return word, np.fromiter(vec, dtype=np.float)

    def _load_from_file(self, f_name):
        log.info(f'loading embeddings from file: "{f_name}"')
        self._name = pathlib.Path(f_name).stem

        vecs = []
        self._skipped = []

        with open(f_name, 'r') as fd:
            for line in fd:
                word, vec = self._map_to_np(line)

                if word in self._vocab:
                    # there are very strange "words" (looks like decoding
                    # errors) in glove files. They appear multiple times.
                    # Skipping those...
                    self._skipped.append(word)
                    continue

                self._vocab[word] = len(vecs)
                vecs.append(vec)

        log.info(f'skipped {len(self._skipped)} words (multiple)')

        self._buf = np.array(vecs)
        assert len(self._buf.shape) == 2
        assert len(vecs) == len(self._vocab)
        assert sorted(self._vocab.values())[-1] == len(vecs) - 1

    def __init__(self, f_name: str):
        super().__init__()

        self._vocab = {}
        self._load_from_file(f_name)


#
#  to work with word2vec binary format
#
class BinaryEmbed(Embed):

    pass


#
#  to work with numpy npz files
#
class NPEmbed(Embed):

    pass


#
#  creation
#
def _raise_no_provider(name):
    raise Exception(f'provider "{name}" not supported')


@attr.s
class Config:
    """
    This class exists to allow for a definition independent
    creation of an embedding provider. The workflow is as
    follows:

    1. Create Config (manually, from_conf, from_args, ...)
    2. Invoke embed.create(config)

    """
    provider:   str = attr.ib()
    file_name:  str = attr.ib(default=None)
    vocabulary: str = attr.ib(default=None)

    @staticmethod
    def from_conf(conf: configobj.Section) -> 'Config':
        """
        Expects the [embedding] section of a config file
        """

        def _check_property(prop_name: str):
            if prop_name in conf:
                return

            msg = f'no "{prop_name}" provided in configuration'
            raise Exception(msg)

        log.info('loading embedding from configuration')
        _check_property('provider')

        if conf['provider'] == 'file':
            _check_property('file name')
        elif conf['provider'] == 'h5py':
            _check_property('file name')
        else:
            _raise_no_provider(conf["provider"])

        vocab = None if 'vocabulary' not in conf else conf['vocabulary']

        return Config(
            provider=conf['provider'],
            file_name=conf['file name'],
            vocabulary=vocab)

    @staticmethod
    def from_args(args: argparse.Namespace, required=True) -> Embed:
        """
        Expects an argparse namespace with the options defined
        in add_parser_args.
        """

        def _raise(name, arg):
            raise Exception(f'no {name} given ({arg})')

        if not args.embed_provider:
            if not required:
                return None

            _raise('provider', '--embed-provider')

        if args.embed_provider == 'file':
            if not args.embed_file_name:
                _raise('file name', '--embed-file-name')
        elif args.embed_provider == 'h5py':
            if not args.embed_file_name:
                _raise('file name', '--embed-file-name')
            if not args.embed_vocabulary:
                _raise('vocabulary', '--embed-vocabulary')
        elif required:
            _raise_no_provider(args.embed_provider)

        return Config(
            provider=args.embed_provider,
            file_name=args.embed_file_name,
            vocabulary=args.embed_vocabulary, )


def create(conf: Config) -> Embed:
    # the check for consistency and correctness
    # must already been conducted to create appropriate
    # error messages (hence the use of asserts here)

    embed = None

    if conf.provider == 'file':
        assert conf.file_name
        embed = FileEmbed(conf.file_name)

    if conf.provider == 'h5py':
        assert conf.file_name
        embed = H5PYEmbed(conf.file_name, conf.vocabulary)

    assert embed is not None
    return embed


def add_parser_args(parser: argparse.ArgumentParser) -> None:
    """
    Used to register the necessary command line arguments to
    an argument parser instance.
    """

    parser.add_argument(
        '--embed-provider', type=str,
        help='how to load the embeddings (see ungol.common.embed)')

    parser.add_argument(
        '--embed-file-name', type=str, default=None,
        help='file with embeddings')

    parser.add_argument(
        '--embed-vocabulary', type=str, default=None,
        help='pickled vocabulary')


# ---


@attr.s
class Loader:
    """

    FIXME docs

    """

    batch_size:     int = attr.ib()

    ram_chunk_size: int = attr.ib()
    gpu_chunk_size: int = attr.ib()

    embed:         Embed = attr.ib()
    device: torch.device = attr.ib()

    # ---

    # whether each chunk is shuffled before yielding
    shuffle: bool = attr.ib(default=True)

    # ---

    def __attrs_post_init__(self):
        assert torch is not None, 'you need to install torch'

        assert self.gpu_chunk_size <= self.ram_chunk_size, (
            'gpu chunks must be smaller than ram chunks')

        def fit(n, x):
            return math.ceil(n / x) * x

        self.gpu_chunk_size = fit(self.gpu_chunk_size, self.batch_size)
        self.ram_chunk_size = fit(self.ram_chunk_size, self.gpu_chunk_size)

        log.info(f'setting ram chunk size to {self.ram_chunk_size}')
        log.info(f'setting gpu chunk size to {self.gpu_chunk_size}')

        assert self.ram_chunk_size % self.gpu_chunk_size == 0, (
            'gpu chunk size must divide ram chunk size')

        assert self.gpu_chunk_size % self.batch_size == 0, (
            'batch size must divide gpu chunk size')

    # ---

    @property
    def batch_count(self):
        return math.ceil(len(self.embed) / self.batch_size)

    # ---

    def _batcher(self, step, limit):
        for lower in range(0, limit, step):
            upper = min(limit, lower + step)
            yield lower, upper

    # ---

    def _ram_load_chunk(self, lower, upper):
        # multi-dimensional arrays are only shuffled along the first axis
        chunk = self.embed.whole[lower:upper]
        np.random.shuffle(chunk)
        return chunk

    def _ram_from_cache(self):
        try:
            yield self._ram_chunk
            return True
        except AttributeError:
            if len(self.embed) <= self.ram_chunk_size:
                log.info('RAM chunks fit into memory, only loading once')
                self._ram_chunk = self.embed.whole[:]
                yield self._ram_chunk
                return True

        return False

    def _ram_gen_chunks(self):
        if (yield from self._ram_from_cache()):
            return

        gen = self._batcher(self.ram_chunk_size, len(self.embed))
        for lower, upper in gen:

            msg = 'moving chunk {lower}-{upper} to RAM took {delta}s'
            done = ucu.ts(log.info, msg)
            chunk = self._ram_load_chunk(lower, upper)
            done(lower=lower, upper=upper)
            yield chunk

    # ---

    def _gpu_load_chunk(self, ram_chunk, lower, upper):
        t = torch.from_numpy(ram_chunk[lower:upper])

        gpu_chunk = t.to(device=self.device, dtype=torch.float)
        assert not gpu_chunk.requires_grad

        return gpu_chunk

    def _gpu_from_cache(self, ram_chunk):
        try:
            yield self._gpu_chunk
            return True

        except AttributeError:
            if len(ram_chunk) <= self.gpu_chunk_size:
                log.info('GPU chunks fit into memory, only loading once')
                chunk = self._gpu_load_chunk(ram_chunk, 0, len(ram_chunk))
                self._gpu_chunk = chunk
                yield chunk
                return True

        return False

    def _gpu_gen_chunks(self, ram_chunk):
        if (yield from self._gpu_from_cache(ram_chunk)):
            return

        gen = self._batcher(self.gpu_chunk_size, len(ram_chunk))
        for lower, upper in gen:
            chunk = self._gpu_load_chunk(ram_chunk, lower, upper)
            yield chunk

    # ---

    def gen(self) -> Generator[np.ndarray, None, None]:
        for ram_chunk in self._ram_gen_chunks():
            for gpu_chunk in self._gpu_gen_chunks(ram_chunk):

                gen = self._batcher(self.batch_size, len(gpu_chunk))
                for lower, upper in gen:
                    yield gpu_chunk[lower:upper]
