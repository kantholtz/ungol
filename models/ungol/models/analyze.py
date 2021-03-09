#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   NEURAL CODES AND EMBEDDING ANALYZER
#


from ungol.common import logger
from ungol.common import embed as uce
from ungol.common.util import ts

import h5py
import attr
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm as _tqdm

import gc
import sys
import pickle
import pathlib
import argparse
import functools
import multiprocessing as mp

from typing import List
from typing import Dict
from typing import Union
from typing import Tuple
from typing import Generator


# ---


log = logger.get('models.analyze')
tqdm = functools.partial(_tqdm, ncols=80, disable=False)


# ---

DEV_CPU = torch.device('cpu')
DEV_GPU = torch.device('cuda')


RETAINED = 2000  # the k in k-NN
BUF_SIZE = 4000  # resulting file chunk size (BUF_SIZE * RETAINED * 4 Byte)

# ---


@attr.s(frozen=True)
class Neighbour:

    word: str = attr.ib()
    index: int = attr.ib()
    dist: float = attr.ib()


@attr.s
class Neighbours:
    """

    Works chunk-wise as raw data consumes to much space
    for most systems. Offers as much raw data as possible
    for efficient computation.

    Usual computation requires about 6-12GB or RAM (based
    on the chunk size).

    """

    chunk_size: int = attr.ib()  # controls how much RAM is consumed
    vocabulary: Dict[str, int] = attr.ib()
    fd: h5py.File = attr.ib()

    def get_dists(self) -> Generator[np.array, None, None]:
        for bucket in range(self._mapping.shape[0] // self.chunk_size):
            a = bucket * self.chunk_size
            b = a + self.chunk_size
            yield self._dists[a:b]

    def get_mappings(self) -> Generator[np.array, None, None]:
        for bucket in range(self._mapping.shape[0] // self.chunk_size):
            a = bucket * self.chunk_size
            b = a + self.chunk_size
            yield self._mapping[a:b]

    def get_chunks(self) -> Generator[Tuple[np.array, np.array], None, None]:
        for dists, mappings in zip(self.get_dists(), self.get_mappings()):
            yield dists, mappings

    @property
    def lookup(self) -> Dict[int, str]:
        return self._lookup

    def __str__(self) -> str:
        return 'Neighbours: {} words'.format(len(self.vocabulary))

    def __getitem__(self, word: str) -> Tuple[Neighbour]:
        """
        Creates an enumeration of k neighbours for the
        provided word. The whole chunks which contain the
        neighbour rows are cached. To avoid high memory
        consumption, call .free()...

        """
        ref_idx = self.vocabulary[word]
        dists = self._dists[ref_idx]
        mapping = self._mapping[ref_idx]

        neighbours = []
        for pos, nn_idx in enumerate(mapping):

            try:
                nn = Neighbour(
                    word=self.lookup[nn_idx],
                    index=nn_idx,
                    dist=dists[pos])
            except KeyError:
                nn = Neighbour(
                    word='<UNKNOWN>',
                    index=nn_idx,
                    dist=dists[pos])

            neighbours.append(nn)

        return tuple(neighbours)

    def close(self):
        self.fd.close()

    def __attrs_post_init__(self):
        self._dists = self.fd['dists']
        self._mapping = self.fd['mapping']

        assert len(self.vocabulary) == self._dists.shape[0]
        assert len(self.vocabulary) == self._mapping.shape[0]

        self._lookup = {v: k for k, v in self.vocabulary.items()}

    @staticmethod
    def from_file(f_data: str, f_vocab: str, chunk_size: int = int(4e3)):
        fd_h5 = h5py.File(f_data, mode='r')
        with open(f_vocab, mode='rb') as fd_vocab:
            vocabulary = pickle.load(fd_vocab)

        return Neighbours(
            chunk_size=chunk_size,
            vocabulary=vocabulary,
            fd=fd_h5)


# DEPRECATED
# this will only stay here until migrated data was validated
@attr.s
class Neighbours__:
    """

    Works chunk-wise as raw data consumes to much space
    for most systems. Offers as much raw data as possible
    for efficient computation.

    Usual computation requires about 6-12GB or RAM (based
    on the chunk size).

    """

    chunk_size: int = attr.ib()
    vocabulary: Dict[str, int] = attr.ib()
    buckets: List[np.lib.npyio.NpzFile] = attr.ib()

    @property
    def lookup(self) -> Dict[int, str]:
        return self._lookup

    # ---

    def _init_cache(self):
        self._mapping_cache = {}
        self._dist_cache = {}

    def _get_bucket_dist(self, bucket: int) -> np.ndarray:
        try:
            return self._dist_cache[bucket]
        except KeyError:
            self._dist_cache[bucket] = self.buckets[bucket]['dists']
            return self._get_bucket_dist(bucket)

    def _get_bucket_mapping(self, bucket: int) -> np.ndarray:
        try:
            return self._mapping_cache[bucket]
        except KeyError:
            self._mapping_cache[bucket] = self.buckets[bucket]['mapping']
            return self._get_bucket_mapping(bucket)

    # ---

    def __getitem__(self, word: str) -> Tuple[Neighbour]:
        """
        Creates an enumeration of k neighbours for the
        provided word. The whole chunks which contain the
        neighbour rows are cached. To avoid high memory
        consumption, call .free()...

        """

        idx = self.vocabulary[word]
        bucket = idx // self.chunk_size
        bucket_idx = idx - bucket * self.chunk_size

        dists = self._get_bucket_dist(bucket)
        mapping = self._get_bucket_mapping(bucket)

        v_dist = dists[bucket_idx]
        v_mapping = mapping[bucket_idx]

        neighbours = []
        for i in range(len(v_dist)):
            n_idx = v_mapping[i]
            n = Neighbour(
                word=self._lookup[n_idx],
                index=n_idx,
                dist=v_dist[i])

            neighbours.append(n)

        return list(neighbours)

    def __len__(self) -> int:
        return len(self.buckets)

    # ---
    # generators for raw data

    def get_dists(self) -> Generator[np.array, None, None]:
        for bucket in range(len(self.buckets)):
            yield self._get_bucket_dist(bucket)

    def get_mappings(self) -> Generator[np.array, None, None]:
        for bucket in range(len(self.buckets)):
            yield self._get_bucket_mapping(bucket)

    def get_chunks(self) -> Generator[Tuple[np.array, np.array], None, None]:
        for dists, mappings in zip(self.get_dists(), self.get_mappings()):
            yield dists, mappings

    # ---

    def __str__(self) -> str:
        fmt = '{}:\n  buckets: {}\n  chunk size: {}\n  vocabulary: {}'
        return fmt.format(
            'Neighbours',
            len(self.buckets),
            self.chunk_size,
            len(self.buckets) * self.chunk_size)

    # ---

    def __attrs_post_init__(self):
        self._init_cache()
        self._lookup = {v: k for k, v in self.vocabulary.items()}

    # ---

    def free(self, bucket: Union[None, int] = None) -> int:
        """
        Release allocated memory for raw mapping files loaded
        either when using the raw data iterators or __getitem__.
        """

        amount = 0

        if bucket is None:
            amount = len(self._dist_cache) + len(self._mapping_cache)
            self._init_cache()

        else:
            if bucket in self._dist_cache:
                del self._dist_cache[bucket]
                amount += 1
            if bucket in self._mapping_cache:
                del self._mapping_cache[bucket]
                amount += 1

        gc.collect()
        return amount

    def close(self):
        """
        When loading from files, numpy keeps the descriptors open.
        Using many Neighbours instances, this tends to exceed OS
        file-descriptor limits (ulimit -n on ubuntu/debian).

        """
        for bucket in self.buckets:
            bucket.close()

    @staticmethod
    def from_files(pathname: str, f_vocab: str):

        path = pathlib.Path(pathname)
        buckets = []

        for glob in sorted(path.parents[0].glob(path.parts[-1])):
            buckets.append(np.load(str(glob)))

        assert len(buckets), 'no data loaded'

        # --- checks
        # contiguous?
        ranges = [(b['start'].item(), b['end'].item()) for b in buckets]
        assert all(a[1] + 1 == b[0] for a, b in zip(ranges[::2], ranges[1::2]))

        # chunk sizes equal?
        chunk_sizes = [d[1] - d[0] for d in ranges]
        assert all([chunk_sizes[0] == c for c in chunk_sizes])

        with open(f_vocab, mode='rb') as fd:
            vocabulary = pickle.load(fd)

        n = Neighbours(
            buckets=buckets,
            chunk_size=ranges[0][1] - ranges[0][0] + 1,
            vocabulary=vocabulary)

        return n

# ---


def load_embeddings(conf: uce.Config):
    ember = uce.create(conf)
    t = torch.from_numpy(ember.whole[:])
    t_emb = t.to(device=DEV_GPU, dtype=torch.float)

    return t_emb


# ---


def nn_writer(q: mp.Queue, shape: Tuple[int, int], fname: str):
    """

    Handles intermediate buffering and persisting to disk.
    Executed in separate process. Data is transmitted through
    the queue to this process.

    """

    height, width = shape

    size = (BUF_SIZE, width)
    buf_dists = np.zeros(size, dtype=np.float)
    buf_mapping = np.zeros(size, dtype=np.int)

    total = shape[0]
    iterations = total // BUF_SIZE
    # fmt = fname + '.{:0' + str(len(str(iterations - 1))) + 'd}'

    path = pathlib.Path(fname)
    path.parents[0].mkdir(parents=True, exist_ok=True)

    # prepare h5 file

    fd = h5py.File(fname, 'w')
    ds_dists = fd.create_dataset('dists', shape)
    ds_mapping = fd.create_dataset('mapping', shape)

    # accept data from queue

    with tqdm(total=iterations, position=2) as bufbar:
        for i in tqdm(range(total), total=total, position=1):
            dists, mapping = q.get()

            assert dists.shape == (RETAINED, )
            assert mapping.shape == (RETAINED, )

            buf_dists[i % BUF_SIZE] = dists
            buf_mapping[i % BUF_SIZE] = mapping

            pos = i + 1
            if pos % BUF_SIZE == 0:

                a = pos - BUF_SIZE
                ds_dists[a:pos] = buf_dists
                ds_mapping[a:pos] = buf_mapping

                # part = i // BUF_SIZE
                # np.savez_compressed(
                #     fmt.format(part),
                #     start=(i + 1) - BUF_SIZE,
                #     end=i,
                #     dists=buf_dists,
                #     mapping=buf_mapping)

                bufbar.update(1)

    fd.close()


# ---


class NNStats:

    bins = int(1e3)

    @property
    def minimum(self):
        return self._minimum

    @minimum.setter
    def minimum(self, minimum):
        if self._minimum is None or minimum < self._minimum:
            self._minimum = minimum

    @property
    def maximum(self):
        return self._maximum

    @maximum.setter
    def maximum(self, maximum):
        if self._maximum is None or self._maximum < maximum:
            self._maximum = maximum

    def __init__(self):
        self._maximum = None
        self._minimum = None
        self.histogram = np.zeros(NNStats.bins, dtype=np.int)
        self.bin_edges = None


class NNGen:

    def __init__(self, X, fn, retained=RETAINED):
        N, E = X.shape

        assert N >= retained
        print('got {} embeddings with {} dimensions'.format(N, E))

        self.X = X
        self.fn = fn
        self.retained = retained
        self.stats = NNStats()

    def __iter__(self):
        N, _ = self.X.shape
        for i in tqdm(range(N)):

            result = self.fn(self.X, self.X[i])
            dists, mapping = (a.cpu().numpy() for a in result)

            self.stats.minimum = dists.min()
            self.stats.maximum = dists.max()

            yield dists[:self.retained], mapping[:self.retained]


# ---

# Distance functions accept a matrix X and a vector y
# and compute the distance of the vector to all rows
# of the matrix. A tuple containing the distance and
# the respective indexes sorted ascending by distance.


_cossim = torch.nn.CosineSimilarity()


def _cosine(X, y):
    Y = y.expand((X.shape[0], ) + y.shape)
    y = _cossim(X, Y)
    dists, mapping = torch.sort(y, descending=True)
    return dists, mapping


def _pnorm(X, y, p=2):
    norm = (X - y).abs().norm(dim=-1, p=p)
    dists, mapping = torch.sort(norm)
    return dists, mapping


def _hamming(X, y):
    _, E = X.shape
    hamming = E - (X - y == 0).sum(dim=1)
    dists, mapping = torch.sort(hamming, dim=-1)
    return dists, mapping


# ---


# see rant in _gen_histogram
_shitq = mp.Queue()


def _buffer_stats():
    """
    Accepts data of _update_stats. Alive as long
    as there are jobs in the queue. Exits upon
    None as message.
    """
    global _shitq
    stats = NNStats()

    while True:
        data = _shitq.get()
        if data is None:
            _shitq.put(stats)
            break

        hist, bin_edges = data
        stats.histogram += hist
        stats.bin_edges = bin_edges


def _calc_stats(dists: np.array, bins: int, rg: Tuple[float, float]):
    """
    Calculates the histogram and chooses min/max.
    Sends data to _buffer_stats. Called for every
    dist array concurrently. Order is not important
    as min/max/+ are commutative.
    """
    global _shitq

    hist, bin_edges = np.histogram(dists, bins=bins, range=rg)
    _shitq.put((hist, bin_edges))


def _gen_histogram(fd: h5py.File, bins: int):
    # Alright, this has been frustrating:

    # (I) I cannot use a locally scoped queue as pickle
    # is not able to pickle the locally bound function (why would
    # it even do that? It only has to pickle the return value).
    # (II) I cannot use a manager, because pytorch fails miserably
    # as for some reason it thinks it has to reinitialize
    # CUDA even though no tensors are ever used in the forked
    # process.
    # (III) I cannot switch the process start_method to 'spawn' because
    # it was already initialized by modules imported by this module.
    # Also: changing some global state which has side effect even
    # regarding other modules is probably a _very_ bad idea.
    # (IV) When using pathos as a replacement for multiprocessing
    # it spewed a whole new class of different errors and for the moment
    # I simply give that up and use global variables.

    # I hate pickle so much.

    minimum = fd['dists'].attrs['minimum']
    maximum = fd['dists'].attrs['maximum']

    rg = int(minimum - 1), int(maximum + 1)

    print('creating histogram in range [{}, {}]'.format(*rg))

    pool = mp.Pool()
    proc = mp.Process(target=_buffer_stats)
    proc.start()

    chunk_size = BUF_SIZE

    print()
    # it is not using map or async_map to be able to slow
    # down the reader on systems with little RAM
    for chunk in tqdm(range(fd['dists'].shape[0] // chunk_size)):
        a = chunk * chunk_size
        b = a + chunk_size
        dists = fd['dists'][a:b]
        pool.apply_async(_calc_stats, (dists, bins, rg))

    print('\n', 'waiting for workers to finish')
    pool.close()  # harbl
    pool.join()

    print('awaiting result')
    # feed some cyanide
    _shitq.put(None)
    proc.join()
    stats = _shitq.get()

    ds_hist = fd.create_dataset('histogram', stats.histogram.shape)
    ds_hist[:] = stats.histogram
    ds_hist.attrs['bin_edges'] = stats.bin_edges

    print('finished creating histogram')


def _nn_proxy(gen: NNGen, fname: str):
    print('creating nearest neighbour map', '\n')

    shape = (gen.X.shape[0], gen.retained)

    q = mp.Queue()
    p = mp.Process(target=nn_writer, args=(q, shape, fname))
    p.start()

    for tup in gen:
        q.put(tup)

    p.join()
    print('\n' * 3)

    print('writing metadata information')
    fd = h5py.File(fname, 'a')

    ds_dist = fd['dists']
    ds_dist.attrs['minimum'] = gen.stats.minimum
    ds_dist.attrs['maximum'] = gen.stats.maximum

    print('distances are in between [{}, {}]'.format(
        gen.stats.minimum, gen.stats.maximum))

    _gen_histogram(fd, NNStats.bins)
    fd.close()
    print('done')


def nn_manhattan(ember_conf: uce.Config, fname: str):
    print('creating manhattan distances from embeddings')

    t_emb = load_embeddings(ember_conf)
    p1 = functools.partial(_pnorm, p=1)
    gen = NNGen(t_emb, p1)

    _nn_proxy(gen, fname)


def nn_euclidean(ember_conf: uce.Config, fname: str):
    print('creating euclidean distances from embeddings')

    t_emb = load_embeddings(ember_conf)
    p2 = functools.partial(_pnorm, p=2)
    gen = NNGen(t_emb, p2)

    _nn_proxy(gen, fname)


def nn_cosine(ember_conf: uce.Config, fname: str):
    print('creating cosine similarity distances from embeddings')

    t_emb = load_embeddings(ember_conf)
    gen = NNGen(t_emb, _cosine)

    _nn_proxy(gen, fname)


def nn_hamming(_, in_file: str, out_file: str):
    # e.g. nn_hamming(None, 'codemap.model-2000.h5', 'hamming.h5')
    print('creating hamming distances from codes')

    with h5py.File(in_file, 'r') as fd:

        print('loading data into memory')
        conv = torch.from_numpy(fd['codes'][:].astype(np.int))

        print('moving data to GPU')
        codes = conv.to(device=DEV_GPU)

    print('w o r k i n g')
    gen = NNGen(codes, _hamming)
    _nn_proxy(gen, out_file)


# ---


"""
def _distance_histogram(
        neighbours: Neighbours,
        codomain: Tuple[int, int],
        bins: int,
        ranges: Tuple[int],
        q: mp.Queue = None):

    assert len(codomain) == 2, codomain
    assert bins > 0, bins

    index = np.arange(bins) / bins * codomain[1]
    data = {k: np.zeros(bins, dtype=np.int) for k in ranges}

    df = pd.DataFrame(data, index=index)

    # retrieve distances chunk-wise
    gen = enumerate(neighbours.get_dists())
    for i, chunk in tqdm(gen, disable=(q is not None)):

        for k in data.keys():
            selection = chunk[:, 1:k + 1].flat
            df[k] += np.histogram(selection, bins=bins, range=codomain)[0]

        neighbours.free()
        if q is not None:
            q.put(i)

    if q is not None:
        q.put(df)

    return df


def __distance_histogram(glob_neighbours: str):
    neighbours = Neighbours.from_files(glob_neighbours)

    q = mp.Queue()
    args = (neighbours, (0, 140), 1800, q)
    p = mp.Process(target=_distance_histogram, args=args)

    p.start()
    with tqdm(total=len(neighbours)) as bar:
        for _ in range(len(neighbours)):
            q.get()
            bar.update(1)

        print('awaiting result')
        df = q.get()

    print('join processes')
    p.join()
    df
"""


def distance_histogram(
        f_neighbours: str, ):  # file to be loaded by Neighbours

    # FIXME: .h5 files produced by nn-*
    pass


# ---


def _gen_isect_dataframes(
        n_ref: Neighbours,
        n_cmp: Neighbours,
        ranges: Tuple[int], ):

    assert n_ref.chunk_size == n_cmp.chunk_size
    chunk_size = n_ref.chunk_size

    # k-independent, chunk-wise iteration
    # used for chunk-independent, k-bound, row-wise iteration
    def gen_rows():
        chunks = zip(n_ref.get_chunks(), n_cmp.get_chunks())
        for chunk, (chunk_ref, chunk_cmp) in enumerate(chunks):
            for k in ranges:

                # crop to obtain k-nn
                # chunks are concatenated (dist, mapping), (dist, mapping)
                aggregate = chunk_ref + chunk_cmp
                cropped = tuple(a[:, 1:k + 1] for a in aggregate)
                assert len(cropped) == 4

                yield from (
                    (k, i, i + chunk * chunk_size, t)
                    for i, t in enumerate(zip(*cropped)))

    # k-bound, row-wise iteration
    # idx_local is the chunk-local index
    # idx_global is the vocabulary index
    for k, idx_local, idx_global, aggregate in gen_rows():
        dist_ref, map_ref, dist_cmp, map_cmp = aggregate

        # Compute intersections of mapping indexes per word
        # to obtain shared nearest neighbours;
        # 'return_indices' requires numpy >= 1.15;
        # Obvious but important: order is not preserved
        # in isect - order by respective distances...
        isect, idx_ref, idx_cmp = np.intersect1d(
            map_ref, map_cmp,
            assume_unique=True, return_indices=True)

        # write to dataframes
        word = n_ref.lookup[idx_global]

        pd_ref = pd.DataFrame({word: dist_ref[sorted(idx_ref)]})
        pd_cmp = pd.DataFrame({word: dist_cmp[sorted(idx_cmp)]})

        yield k, pd_ref, pd_cmp


def _aggregate_common_nn(n_ref, n_cmp, ranges) -> '{k: ([], [])':  # noqa
    """

    see _gen_common_nn(...)

    """
    done = ts(print, 'iteration took {delta}s')
    total = len(n_ref.vocabulary) * len(ranges)
    gen = enumerate(_gen_isect_dataframes(n_ref, n_cmp, ranges))

    agg = {k: ([], []) for k in ranges}
    for i, (k, pd_ref, pd_cmp) in tqdm(gen, total=total):
        ls_ref, ls_cmp = agg[k]
        ls_ref.append(pd_ref)
        ls_cmp.append(pd_cmp)

    print()
    done()

    return agg


def _compile_common_nn(agg):
    """

    see _gen_common_nn(...)

    """
    done = ts(print, 'concatenation took {delta}s')
    for k in tuple(agg.keys()):

        assert len(agg[k][0]) == len(agg[k][1])
        print('compiling k-{} from {} units'.format(k, len(agg[k][0])))

        chunk_size = int(1e4)
        total = len(agg[k][0]) // chunk_size
        assert len(agg[k][0]) % chunk_size == 0, len(agg[k][0])

        # reverse list for nice pop() usage:
        # must pop to free memory (cannot change data structure
        # while iterating...)

        agg1 = agg[k][0]
        agg2 = agg[k][1]
        agg1.reverse()
        agg2.reverse()

        # compile chunks

        comp = [[], []]

        print()
        for chunk in tqdm(range(total)):
            amount = min(len(agg1), chunk_size)

            agg1_chunk = [agg1.pop() for _ in range(amount)]
            agg2_chunk = [agg2.pop() for _ in range(amount)]

            comp[0] = [pd.concat(comp[0] + agg1_chunk, axis=1)]
            comp[1] = [pd.concat(comp[1] + agg2_chunk, axis=1)]

        print()
        yield k, (comp[0][0], comp[1][0])

    done()


def _gen_common_nn(
        n_ref: Neighbours,
        n_cmp: Neighbours,
        ranges: Tuple[int]) -> Generator[
            Tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """
    Create a mapping of k -> (ref: pd.DataFrame, hamming: pd.DataFrame).
    While iteration, for each word two DataFrames are created containing
    the respective indexes of common nearest neighbours bound by k.
    DataFrames are compiled at the end by concatenating the aggregated ones.

    https://tomaugspurger.github.io/modern-4-performance
    http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

    Make sure to provide some juicy RAM :)

    """

    print('\n', 'start searching for common nearest neighbours', '\n')

    # intermed = pathlib.Path('.tmp.pickle')
    # if intermed.is_file():
    #     print('loading intermediary file')
    #     agg = pickle.load(intermed.open('rb'))
    # else:
    #     agg = _aggregate_common_nn(n_ref, n_cmp, ranges)
    #     print('saving intermediary')
    #     pickle.dump(agg, intermed.open('wb'))

    agg = _aggregate_common_nn(n_ref, n_cmp, ranges)

    print('compiling data frames')
    yield from _compile_common_nn(agg)

    # print('deleting intermediary file')
    # intermed.unlink()


def correlate(
        ember_conf: uce.Config,
        f_reference: str,
        f_compare: str,
        out_folder: str,
        *range_argv):
    """
    Creates correlation data frames. These frames
    contain for each word the common nearest neighbours
    of both provided embedding spaces. The data consists
    of both the respective indexes and each distance.

    You will need k * len(vocabulary) * 4 Bytes * 2 Bytes of RAM
    - e.g. for k=4e3, N=4e5: 12.8GB
    plus all intermediary memory required for loaded chunks

    Convention for produced files:
    opt/correlate/<EMBEDDING>_<REF>-<CMP>/corr-[k].h5

    The following files are writen to out_folder:
    corr-[k].h5 for every k in range_argv.

    :param f_reference: reference distances (usually cosine)
    :param f_compare: correlating distances (usually hamming)
    :param f_vocab: vocabulary file str -> int
    :param out_folder: place to write results to
    :param *range_argv: the k in k-NN (use multiple for efficiency)

    """

    path = pathlib.Path(out_folder)
    assert path.is_dir(), str(path)
    assert len(range_argv) > 0

    n_ref = Neighbours.from_file(f_reference, ember_conf.f_vocab)
    print('loaded reference nn:\n', str(n_ref))

    n_cmp = Neighbours.from_file(f_compare, ember_conf.f_vocab)
    print('loaded hamming nn:\n', str(n_cmp))

    ranges = tuple(map(int, range_argv))
    assert all([rg > 0 for rg in ranges]), ranges

    for k, dfs in _gen_common_nn(n_ref, n_cmp, ranges):
        assert len(dfs) == 2, len(dfs)
        fname = str(path / 'corr-{}.h5'.format(k))
        print('writing {}'.format(fname))

        for name, df in zip(('ref', 'cmp'), dfs):
            print('  writing "{}" subgroup'.format(name))
            df.to_hdf(fname, name + '/common', mode='a')

    print('done')

# ---


CMD_MAPPING = {

    # create nearest neighbour mappings
    'nn-manhattan': nn_manhattan,
    'nn-euclidean': nn_euclidean,
    'nn-cosine': nn_cosine,
    'nn-hamming': nn_hamming,

    # create distance histograms from
    # files produced by nn-* for plotting etc.
    # 'distance-histogram': distance_histogram,

    # correlation of different distances
    'correlate': correlate,

}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('cmd')
    parser.add_argument('argv', nargs='*')
    uce.add_parser_args(parser)

    return parser.parse_args()


def main():
    print('\n', 'welcome to the analyzer'.upper(), '\n')
    args = parse_args()
    ember_conf = uce.Config.from_args(args, required=False)

    if args.cmd not in CMD_MAPPING:
        print('unexpected argument: {}'.format(args.cmd))
        print('choose one of: ' + '\n\t' + '\n\t'.join(CMD_MAPPING.keys()))
        sys.exit(2)

    CMD_MAPPING[args.cmd](ember_conf, *args.argv)


if __name__ == '__main__':
    main()
