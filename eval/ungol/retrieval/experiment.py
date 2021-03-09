#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ungol.common import logger
from ungol.retrieval import stats
from ungol.retrieval import common
from ungol.retrieval import clients

import requests
import numpy as np
from tabulate import tabulate
from tqdm import tqdm as _tqdm

import time
import json
from pprint import pformat
from collections import defaultdict

from typing import Callable
from typing import Collection


# ---


log = logger.get('retrieval.experiment')


def tqdm(*args, **kwargs):
    yield from _tqdm(*args, ncols=80, **kwargs)


# --- experiments


class Task(dict):
    """
    Maps    doc_id -> bool
    Named:    task -> flag

    Currently depending on elasticsearch - could also
    pickle the topics...
    """

    @property
    def correct(self) -> int:
        return len([v for v in self.values() if v])

    @property
    def incorrect(self) -> int:
        return len([v for v in self.values() if not v])


class Truth(defaultdict):
    """
    Maps     top_id -> doc_id -> bool
    Named:    truth ->   task -> flag
    """

    def __init__(self):
        # use Task as default factory for self (the defaultdict)
        super().__init__(Task)

    @staticmethod
    def from_file(f_truth: str):
        with open(f_truth, mode='r', encoding='ascii') as fd:
            truth_raw = [line for line in fd.readlines()
                         if len(line.strip()) > 0]

            log.info(f'read {len(truth_raw)} ground truth items')

        truth = Truth()
        sample_count = len(truth_raw)

        for line in truth_raw:
            top_id, _, doc_id, val = line.split()
            assert val == '1' or val == '0'
            truth[top_id][doc_id] = True if val == '1' else False

        assert sample_count == sum([len(v) for v in truth.values()])
        log.info(f'imported {len(truth)} ground truth topics')

        return truth


#
#  generic entry points
#

Hook = Callable[['top_id'], Collection[clients.Result]]


def _run_one(top_id: str, task: Task, hook: Hook) -> stats.TopicStats:
    StatFactory = stats.TopicStats.from_search_results

    delta = common.timer()
    results = hook(top_id)
    return StatFactory(results, task, delta=delta())


def _run(truth: Truth, hook: Hook, name: str) -> stats.ExperimentStats:
    stats_agg = {}

    for top_id, task in tqdm(truth.items(), leave=False, desc=name):
        stats_agg[top_id] = _run_one(top_id, task, hook)

    return stats.ExperimentStats(stats=stats_agg)


def run(
        client: clients.Client,
        truth: Truth,
        k: int = None,
        pooled: bool = False,
        hook=None, ) -> stats.ExperimentStats:
    """
    Generic entry point for searching with an arbitrary
    common.Client instance.

    :param client: for searching
    :param truth: ground truth for evaluation
    :param k: amount of search results
    :param pooled: whether to only use pooled documents

    """

    def _hook(top_id) -> Collection[clients.Result]:
        only = truth[top_id].keys() if pooled else None
        return client.retrieve(top_id, k=k, only=only)

    hook = hook if hook else _hook
    return _run(truth, hook, client.name)


#
#  specific entry points
#
#  - adapted from (jf)
def _es_set_bm25_params(url: str, k1: float = 1.2, b: float = 0.75):
    log.info('setting BM25 parameters to k1={k1}, b={b}')

    headers = {"Content-Type": 'application/json'}
    query = json.dumps({
            'index': {'similarity': {'default': {
                'type': 'BM25',
                'k1': k1,
                'b': b,
            }, }, },
        })

    endpoint = '{url}/{index}'.format(**{'url': url, **common.CTX_ARTICLES})

    # It is dangerous to eval data retrieved from some distant service.
    # I leave it in because I trust my local elasticsearch ;)
    # - it is optional though, just delete it if you like.

    res = requests.post(endpoint + '/_close', headers=headers)
    assert res.status_code == 200, 'close failed:\n' + pformat(eval(res.text))

    res = requests.put(endpoint + '/_settings', query, headers=headers)
    assert res.status_code == 200, 'update failed:\n' + pformat(eval(res.text))

    res = requests.post(endpoint + '/_open', headers=headers)
    assert res.status_code == 200, 'open failed:\n' + pformat(eval(res.text))

    time.sleep(3)


def es_gridsearch(client: clients.Elasticsearch, truth: Truth) -> str:
    url = 'http://{host}:{port}'.format(**client.conf)

    K = 250

    all_k1 = [1.2, 1.32, 1.44, 1.56, 1.68, 1.8, 1.92, 2.0]
    all_b = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0]

    combinations = [(x, y)
                    for x in enumerate(all_k1)
                    for y in enumerate(all_b)]

    results = np.zeros((len(all_k1), len(all_b)))

    # results = defaultdict(list)
    for (i, k1), (j, b) in combinations:
        _es_set_bm25_params(url, k1=k1, b=b)
        stats = run(client, truth, k=K)
        results[i, j] = stats.mean_avg_precision

    argmax = np.argmax(results, axis=None)
    max_idx = np.unravel_index(argmax, results.shape)

    tab_data = []
    for k1, row in zip(all_k1, results):
        tab_data.append((k1, ) + tuple(row))

    tab_headers = ['k1 \\ b'] + [str(b) for b in all_b]
    table = tabulate(tab_data, headers=tab_headers)

    sbuf = ['elasticsearch parameter gridsearch', '']

    sbuf.append('found maximum ({:.5f}) at ({}, {}): k1={}, b={}'.format(
        results[max_idx], max_idx[0], max_idx[1],
        all_k1[max_idx[0]], all_b[max_idx[1]]))

    sbuf.append('')
    sbuf.append(table)

    return '\n'.join(sbuf)


def es_find_k(client: clients.Elasticsearch, truth: Truth):
    K = 2000

    print_selection = [
        1, 2, 3, 4, 5,
        10, 20, 30, 40, 50,
        100, 200, 300, 400, 500,
        1000, 2000]

    tab_data = []
    exp = run(client, truth, k=K)

    for k in print_selection:
        tab_headers, data = exp.tab_data_at(k)
        tab_data.append([k] + list(data))

    table = tabulate(tab_data, ('k', ) + tab_headers)
    print(table, '\n')
    return exp
