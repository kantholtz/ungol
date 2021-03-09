#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  reduction methods for word vector collections
#


from ungol.common import logger
from ungol.common import util as ucu
from ungol.common import embed as uce

import attr
import torch
import numpy as np

from typing import Any
from typing import Set
from typing import Dict
from typing import Tuple
from typing import Collection


# conditional imports

skdecomp = ucu.load_module('sklearn.decomposition')
sent2vec = ucu.load_module('sent2vec')
infersent = ucu.load_module('infersent.models')

log = logger.get('sentemb.redux')


class Redux:

    @property
    def name(self) -> str:
        raise NotImplementedError()

    @property
    def dimensions(self) -> int:
        """
        Output dimensionality.
        """
        raise NotImplementedError

    def do_batch(self, batch: Tuple[Tuple[str]]):
        mapped = self.map_batch(batch)
        reduced = self.reduce_batch(mapped)
        return reduced

    # invoked in main process per sentence batch
    # return value determines what is handed to reduce()
    def map_batch(self, batch: Collection[Tuple[str]]) -> Collection[Any]:
        raise NotImplementedError()

    # must be thread/process safe!
    def reduce_batch(self, batch) -> Collection[np.array]:
        """
        Accepts what is returned by self.map_batch().
        """
        return tuple(self.reduce(args) for args in batch)
        raise NotImplementedError()

    # invoked once
    def finish(self, X):
        """
        X is array like -> (n, d)
          n: sentences
          d: embedding dimensionality

        """
        pass


class EmbedRedux(Redux):
    """
    Reduction with pre-trained word embeddings

    """
    # no name() implementation - sort of an abstract class

    @property
    def dimensions(self) -> int:
        return self.X.shape[1]

    @property
    def vocab(self) -> Dict[str, int]:
        return self._vocab

    @property
    def X(self) -> np.ndarray:
        return self._X

    @property
    def common(self) -> Set[str]:
        return self._common

    def __init__(self, embed: uce.Embed, *args, **kwargs):
        super().__init__(*args, **kwargs)

        log.info('loading embeddings into memory')
        self._X = embed.whole[:]
        self._vocab = embed.vocab

    def map_batch(self, batch: Collection[Tuple[str]]):
        idx_batch = (
            [self.vocab[tok] for tok in toks if tok in self.vocab]
            for toks in batch)

        return [self._X[idxs] for idxs in idx_batch]


class BoW(EmbedRedux):
    """

    Bag of words.
    Σ_{t \in S} t

    """

    @property
    def name(self):
        return 'bow'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reduce_batch(self, batch):
        return [np.sum(vecs, axis=0) for vecs in batch]


class MBoW(EmbedRedux):
    """

    Mean bag of words.
    1\|S| * Σ_{t \in S} t

    """

    @property
    def name(self):
        return 'mbow'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reduce_batch(self, batch):
        return [np.mean(vecs, axis=0) for vecs in batch]


class SIF(EmbedRedux):
    """
    Arora, S., Liang, Y., & Ma, T. (2016).
    A simple but tough-to-beat baseline for sentence embeddings.
    """

    @property
    def name(self):
        return 'sif'

    def __init__(self, counts, alpha):
        raise NotImplementedError('implement batch wise')

        assert skdecomp is not None, 'install sklearn'

        self._alpha = alpha
        self._counts = counts
        self._total_count = sum(counts.values())

    def map(self, toks: Tuple[str]):
        vecs = super().map(toks)
        return vecs, toks

    def reduce(self, args):
        vecs, toks = args

        counts = [self._counts[tok] for tok in toks]
        tf = np.array(counts) / self._total_count
        alpha = np.repeat(self._alpha, len(vecs))

        w = alpha / (alpha + tf)
        y = (np.array(vecs).T * w).mean(axis=1)

        assert y.shape == vecs[0].shape
        return y

    def finish(self, X):
        # TODO handle data which does not fit in memory
        # TODO implement this using torch:
        #        - https://github.com/pytorch/pytorch/issues/8049

        log.info('computing truncated SVD')
        Z = X[:]

        svd = skdecomp.TruncatedSVD(n_components=1, n_iter=7, random_state=0)
        svd.fit(Z)
        comps = svd.components_

        log.info('transforming embeddings')
        X[:] = Z - Z.dot(comps.T) * comps


# --------------------


class Sent2Vec(Redux):
    """
    Pagliardini, M., Gupta, P., & Jaggi, M. (2017).
    Unsupervised learning of sentence embeddings using
    compositional n-gram features.
    arXiv preprint arXiv:1703.02507.
    """

    @property
    def name(self):
        return 'sent2vec'

    @property
    def dimensions(self):
        return self.model.get_emb_size()

    @property
    def model(self):
        return self._model

    def __init__(self, f_model: str):
        assert sent2vec is not None, 'install sent2vec'

        self._f_model = f_model
        self._model = sent2vec.Sent2vecModel()
        self._model.load_model(f_model, inference_mode=True)

    def map_batch(self, toks: Tuple[str]):
        return toks

    def reduce_batch(self, batch):
        emb = self.model.embed_sentences([' '.join(toks) for toks in batch])
        return emb

    def finish(self, X):
        self.model.release_shared_mem(self._f_model)


# --------------------


# inject own embedding mechanic
def _infersent_hack_build_vocab(self, embed: uce.Embed):
    log.info(f'mapping {len(embed)} word embedding vectors')
    space = embed.whole[:]
    self.word_vec = {word: space[idx] for word, idx in embed.vocab.items()}


class Infersent(Redux):
    """
    Conneau, A., Kiela, D., Schwenk, H., Barrault, L., & Bordes, A. (2017).
    Supervised learning of universal sentence representations from natural
    language inference data.
    arXiv preprint arXiv:1705.02364.
    """

    DIMS = 4096  # FIXME: is this constant?

    @attr.s
    class SplitProxy:

        toks: Tuple[str] = attr.ib()

        def split(self):
            return self.toks

    # ---

    @property
    def name(self):
        return 'infersent'

    @property
    def dimensions(self):
        return Infersent.DIMS

    @property
    def model(self):
        return self._model

    def __init__(
            self,
            f_model: str,
            embed: uce.Embed,
            cuda: bool = True,
            **kwargs):

        assert infersent is not None, 'install infersent'

        default_params = {
            'bsize': 64,
            'word_emb_dim': 300,
            'enc_lstm_dim': 2048,
            'pool_type': 'max',
            'dpout_model': 0.0,
            'version': 1}  # GloVe Models

        params_model = {**default_params, **kwargs}

        log.info(f'loading infersent model from {f_model}')
        self._model = infersent.InferSent(params_model)
        self.model.load_state_dict(torch.load(f_model))

        if cuda:
            log.info('moving infersent model to CUDA')
            self.model.cuda()

        _infersent_hack_build_vocab(self.model, embed)

    def map_batch(self, batch: Tuple[Tuple[str]]):

        # if tokenize is set to False, a simple .split() is invoked
        proxy = [Infersent.SplitProxy(toks) for toks in batch]
        encoded = self.model.encode(proxy, tokenize=False)
        return encoded.squeeze()

    def reduce(self, vec):
        return vec


# --- Proxy Classes


def binarize(parent_redux):
    class Binarized(parent_redux):

        @property
        def name(self):
            parent = super().name
            return f'binarized_{parent}'

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def finish(self, X):
            super().finish(X)

            # thresholding each feature dimension
            # by the median of all observations

            m = np.median(X, dim=0)
            X[X < m] = 0
            X[X >= m] = 1

    return Binarized
