# -*- coding: utf-8 -*-

"""
This file contains all clients that compete with each other.

"""

from ungol.common import logger
from ungol.index import index as uii
from ungol.similarity import stats
from ungol.retrieval import common

import attr
import time
import pickle
import random
import multiprocessing as mp
from collections import defaultdict

import elasticsearch as es
from wmd import WMD as WMDR
from gensim.models.fasttext import FastText

from typing import Any
from typing import List
from typing import Dict
from typing import Tuple
from typing import Callable


log = logger.get('retrieval.clients')


# ---


@attr.s
class Result:

    doc_id:  str = attr.ib()
    score: float = attr.ib()


class Client:

    @property
    def time(self) -> float:
        return sum(self._times) / len(self._times)

    @property
    def times(self) -> List[float]:
        return self._times

    def __init__(self):
        self._times = []

    def search(self, text: str, k=None, only=None) -> List[Result]:
        raise NotImplementedError()

    def retrieve(self, top_id: str, k=None, only=None) -> List[Result]:
        raise NotImplementedError()


# ---


class Elasticsearch(Client):

    @property
    def conf(self) -> Dict[str, Any]:
        return self._conf

    @property
    def es_client(self) -> es.Elasticsearch:
        return self._es_client

    def _query(self, content: str, size: int):
        return {'query': {'match': {'content': content}}, 'size': size}

    def _search_wrapper(self, query):
        """
        ES is not returning results reliably. This wrapper retries
        if not results were returned. I have not found a way to replicate
        this behaviour so far...

        """

        retries = 3
        while retries > 0:
            res = self.es_client.search(body=query)['hits']['hits']
            if len(res):
                return res

            retries -= 1
            log.warning(f'no results - retrying ({retries} tries left)')
            time.sleep(2)

        raise Exception('retried multiple times')

    # use common.get_client(...)
    def __init__(self, es_conf: Dict[str, Any]):
        Client.__init__(self)

        self._conf = es_conf
        self._es_client = common.get_client(es_conf)
        assert self.es_client.ping(), 'cannot reach cluster'
        log.info('initialized client')

    def search(self, text: str, k=None, only=None) -> List[Result]:
        query = self._query(text, k)

        results = []
        for es_res in self._search_wrapper(query):
            result = Result(doc_id=es_res['_id'], score=es_res['_score'])
            results.append(result)

        return results

    def retrieve(self, top_id: str, **kwargs) -> List[Result]:
        delta = common.timer()
        topic = common.Topic.from_es(self.es_client, top_id)
        results = Elasticsearch.search(self, topic.description, **kwargs)
        self._times.append(delta())

        return results


# ---


# globalized ro; in case Ungol(multi=True, ...)
_ungol_index: uii.Index = None


def _ungol_worker(args):
    global _ungol_index
    assert _ungol_index is not None, 'there\'s no index, mate!'

    top_id, doc_id, fn = args
    score = fn(_ungol_index, top_id, doc_id)
    return doc_id, score


@attr.s
class Ungol(Client):

    name: str = attr.ib()  # can by anything, used for writing reports
    index_file: str = attr.ib()
    fn: Callable[[uii.Index, str, str], float] = attr.ib()

    verbose:         bool = attr.ib(default=False)
    multiprocessing: bool = attr.ib(default=False)

    # initialized post-init if None
    index: uii.Index = attr.ib(default=None)

    # ---

    @property
    def pool(self) -> mp.Pool:
        assert self.multiprocessing
        return self._pool

    @property
    def report(self) -> Dict[str, stats.ScoreData]:
        assert self.verbose
        return self._report

    # ---

    def __attrs_post_init__(self):
        Client.__init__(self)
        assert not all((self.multiprocessing, self.verbose)), 'mutex'

        if self.index is None:
            log.info('loading index into memory')
            with open(self.index_file, 'rb') as fd:
                self.index = pickle.load(fd)

            log.info('loaded index:\n{}'.format(str(self.index)))

        if self.multiprocessing:
            self._mp_init()
            self._retrieve = self._mp_retrieve

        else:
            self._retrieve = self._sp_retrieve
            if self.verbose:
                self._report = defaultdict(dict)

    # --- mp related

    def _mp_init(self):
        log.info('initializing multiprocessing')

        # globalize index pre-fork
        global _ungol_index
        assert _ungol_index is None, 'you cannot use multiple clients with mp'
        _ungol_index = self.index

        self._pool = mp.Pool()
        log.info('initialized worker pool')

    def _mp_retrieve(self, top_id: str, k=None, only=None):
        assert only, 'not searching anything'

        # make sure to pass only primitive ro values
        work = ((top_id, doc_id, self.fn) for doc_id in only)
        raw = self.pool.map(_ungol_worker, work)
        return self._mapreduce_and_sort(raw, k)

    def _mp_close(self):
        self.pool.close()
        global _ungol_index
        _ungol_index = None
        log.info('pool\'s closed')  # harbl

    # --- sp related

    def _sp_retrieve(self, top_id: str, k=None, only=None):
        assert only, 'not searching anything'

        raw = []
        for doc_id in only:
            if doc_id not in self.index.mapping:
                raw.append((doc_id, -1))
                continue

            scoredata = score = self.fn(self.index, top_id, doc_id)

            # overwrites 'score'
            if self.verbose:
                self._report[top_id][doc_id] = scoredata
                score = scoredata.score

            raw.append((doc_id, score))

        return self._mapreduce_and_sort(raw, k)

    # ---

    def _mapreduce_and_sort(
            self,
            raw: List[Tuple['doc_id', 'score']],
            k: int) -> List[Result]:

        assert len(raw)

        results = list(map(lambda t: Result(*t), raw))
        results.sort(key=lambda r: r.score, reverse=True)

        # for result in results[:20]:
        #     print(result)

        return results[:k]

    # ---

    def search(self, text: str, k=None, only=None) -> List[Result]:
        raise NotImplementedError()  # FIXME
        delta = common.timer()
        res = self._search(text, k, only)
        self._times.append(delta())
        return res

    def retrieve(self, top_id: str, k=None, only=None) -> List[Result]:
        delta = common.timer()
        res = self._retrieve(top_id, k=k, only=only)
        self._times.append(delta())
        return res

    # ---

    def close(self):
        if self.multiprocessing:
            self._mp_close()

# ---


class UngolReranking:

    @property
    def es(self) -> Elasticsearch:
        return self._es

    @property
    def un(self) -> Ungol:
        return self._un

    def __init__(self,
                 un: Ungol,
                 es: Elasticsearch,
                 preselect: int = None, ):

        self._un = un
        self._es = es
        self._preselect = preselect

    def search(self, text: str, k=None):
        results = self.es.search(text, k=self._preselect)
        preselection = [result.doc_id for result in results]
        return self.un.search(text, k=k, only=preselection)

    def retrieve(self, top_id: str, k=None, only=None):
        results = self.es.retrieve(top_id, k=self._preselect)
        preselection = [result.doc_id for result in results]

        # just temporarily!
        random.shuffle(preselection)

        return self.un.retrieve(top_id, k=k, only=preselection)


# ---


class WMD(Client):

    @property
    def emb(self):
        return self._emb

    def __init__(self, f_emb: str, f_db):
        super().__init__()

        log.info('loading embeddings')
        self._emb = FastText.load_fasttext_format(f_emb)

        log.info('loading ungol index')
        with open(f_db, 'rb') as fd:
            self.db = pickle.load(fd)

    def search(self, text: str, k=None):
        raise NotImplementedError()

    def retrieve(self, top_id: str, k=None, only=None):
        assert only, 'not searching anything'
        index = self.db.mapping
        delta = common.timer()

        raw = []
        for doc_id in only:
            if doc_id not in index:
                raw.append((doc_id, -1))
                continue

            q, d = index[top_id], index[doc_id]

            score = self.emb.wv.wmdistance(q.tokens, d.tokens)
            raw.append((doc_id, score))

        results = list(map(lambda t: Result(*t), raw))
        results.sort(key=lambda r: r.score)

        self._times.append(delta())
        return results[:k]


class WMDReranking:

    @property
    def es(self) -> Elasticsearch:
        return self._es

    @property
    def wmd(self) -> WMD:
        return self._wmd

    def __init__(self, wmd: WMD, es: Elasticsearch, preselect: int = None):
        self._wmd = wmd
        self._es = es
        self._preselect = preselect

    def search(self, text: str, k=None):
        results = self.es.search(text, k=self._preselect)
        preselection = [result.doc_id for result in results]
        return self.emd.search(text, k=k, only=preselection)

    def retrieve(self, top_id: str, k=None, only=None):
        results = self.es.retrieve(top_id, k=self._preselect)
        preselection = [result.doc_id for result in results]

        # just temporarily!
        random.shuffle(preselection)

        return self.emd.retrieve(top_id, k=k, only=preselection)


class WMDRelax(Client):

    @property
    def db(self):
        return self._db

    @property
    def emb(self):
        return self._emb

    def __init__(self, f_emb: str, f_db):
        super().__init__()

        log.info('loading embeddings')
        with open(f_emb, 'rb') as fd:
            self._emb = pickle.load(fd)

        log.info('loading ungol index')
        with open(f_db, 'rb') as fd:
            self._db = pickle.load(fd)

    def search(self, text: str, k=None):
        raise NotImplementedError()

    def retrieve(self, top_id: str, k=None, only=None):
        assert only, 'not searching anything'
        index = self.db.mapping
        delta = common.timer()

        def to_nbow(doc_id):
            # transform to the nbow model used by wmd.WMD:
            # ('human readable name', 'item identifiers', 'weights')
            doc = index[doc_id]
            return (doc_id, doc.idx, doc.freq)

        docs = {d: to_nbow(d) for d in only + [top_id]}
        calc = WMDR(self.emb, docs, vocabulary_min=2)

        calc.cache_centroids()
        nn = calc.nearest_neighbors(top_id, k=k)

        self._times.append(delta())

        assert len(nn) == k, f'{len(nn)} not {k}'
        return [Result(*n) for n in nn]
