# -*- coding: utf-8 -*-

"""
Gather and process statistical data for evaluation.
"""

from ungol.retrieval import clients

import enum
import attr
import numpy as np
from tabulate import tabulate

from typing import Dict
from typing import Tuple


# ---


@attr.s
class Grade:
    """
    Each result is either correct or wrong. Wrong candidates
    can either belong to the pool or not be in the ground
    truth at all. Thus, State.MISSED denotes samples that cannot
    be found in the ground truth.

    """

    class State(enum.Enum):
        TRUE = enum.auto()     # both in truth & correct
        WRONG = enum.auto()    # in truth & incorrect
        MISSED = enum.auto()   # not in truth

    state: State = attr.ib()

    def __bool__(self) -> bool:
        return self.state == Grade.State.TRUE

    def __str__(self) -> str:
        return self.state.name


@attr.s
class Hit:

    doc_id:      str = attr.ib()
    score:     float = attr.ib()
    correct:   Grade = attr.ib()

    def __bool__(self) -> bool:
        return self.correct.__bool__()

    @staticmethod
    def from_result(result: clients.Result, task: Dict[str, bool]):
        state: Grade.State = Grade.State.TRUE

        if result.doc_id not in task:
            state = Grade.State.MISSED
        elif not task[result.doc_id]:
            state = Grade.State.WRONG

        return Hit(
            doc_id=result.doc_id,
            score=result.score,
            correct=Grade(state))


@attr.s
class TopicStats:
    """
    Construct this class from a collection of search results
    """

    # based on the ground truth
    positives: int = attr.ib()  # total amount of correct documents

    # based on the search results
    hits: Tuple[Hit] = attr.ib()

    # optional arguments
    timedelta = attr.ib(default=None)

    # precompute data linearly per k <= len(self.hits)
    # (otherwise avgp etc. will turn this to n**2)

    def __attrs_post_init__(self):
        tp_sparse = np.zeros(len(self.hits) + 1)
        fp_sparse = np.zeros(len(self.hits) + 1)

        for i, hit in enumerate(self.hits, 1):
            arr = tp_sparse if hit else fp_sparse
            arr[i] = 1

        # primitives

        TP = self._arr_true_positives = np.cumsum(tp_sparse).astype(np.int)
        FP = self._arr_false_positives = np.cumsum(fp_sparse).astype(np.int)
        FN = self._arr_false_negatives = self.positives - TP

        # complex

        def div(a, b):
            # due to numerical errors, sometimes very small negative
            # values around zero lead to problems downstream where it
            # is assumed that they are always positive (e.g. integer
            # conversion) - thus they are both rounded and made
            # absolute here
            raw = np.divide(a, b, where=b != 0)
            return np.around(np.abs(raw), decimals=10)

        # precision
        P = self._arr_precision = div(TP, TP + FP)

        # recall
        R = self._arr_recall = div(TP, TP + FN)

        # F1
        self._arr_f1 = 2 * div(P * R, P + R)

        # avg precision
        self._arr_avg_precision = np.zeros(len(self.hits) + 1)
        self._arr_avg_precision[1:] = np.cumsum(np.abs(np.diff(R)) * P[1:])

    # --- primitive data

    @property
    def k(self):
        return len(self.hits)

    @property
    def true_positives(self) -> int:
        return self.arr_true_positives[self.k]

    @property
    def arr_true_positives(self) -> np.array:
        return self._arr_true_positives

    @property
    def false_positives(self) -> int:
        return self.arr_false_positives[self.k]

    @property
    def arr_false_positives(self) -> np.array:
        return self._arr_false_positives

    @property
    def false_negatives(self) -> int:
        return self.arr_false_negatives[self.k]

    @property
    def arr_false_negatives(self) -> int:
        return self._arr_false_negatives

    # --- combined data

    @property
    def precision(self) -> float:
        return self.arr_precision[self.k]

    @property
    def arr_precision(self) -> np.array:
        return self._arr_precision

    @property
    def recall(self) -> float:
        return self.arr_recall[self.k]

    @property
    def arr_recall(self) -> np.array:
        return self._arr_recall

    @property
    def f1(self) -> float:
        return self.arr_f1[self.k]

    @property
    def arr_f1(self) -> np.array:
        return self._arr_f1

    @property
    def avg_precision(self) -> float:
        return self.arr_avg_precision[self.k]

    @property
    def arr_avg_precision(self) -> np.array:
        return self._arr_avg_precision

    # ---

    @staticmethod
    def from_search_results(
            results: Tuple[clients.Result],
            task: 'exp.Task',
            delta=None):

        # if len(results) <= task.correct:
        #     fmt = 'WARNING: searched with k={} but there are {} positives'
        #     print(fmt.format(len(results), task.correct))

        hits = [Hit.from_result(result, task) for result in results]
        return TopicStats(positives=task.correct, hits=hits, timedelta=delta)

    # convenience for tabulate

    def tab_data_at(self, k: int):
        headers = ('k', 'POS', 'TP', 'FP', 'FN',
                   'P', 'R', 'AP', )
                   # 'P', 'R', 'F1', 'AP', )

        row = (
            len(self.hits),
            self.positives,
            self.arr_true_positives[k],
            self.arr_false_positives[k],
            self.arr_false_negatives[k],
            int(self.arr_precision[k] * 100),
            int(self.arr_recall[k] * 100),
            # int(self.arr_f1[k] * 100),
            int(self.arr_avg_precision[k] * 100), )

        assert len(set(headers)) == len(headers)
        assert len(headers) == len(row)
        return headers, row

    @property
    def tab_data(self):
        return self.tab_data_at(self.k)


@attr.s
class ExperimentStats:
    """
    Construct this class from a mapping of top_id's -> TopicStats
    """

    # mapping top_id -> TopicStats
    stats: Dict[str, TopicStats] = attr.ib()

    def __len__(self):
        return self._arr_mean_recall.shape[0]

    def __attrs_post_init__(self):
        topics = tuple(self.stats.values())
        assert all(topic.k == topics[0].k for topic in topics)

        def mean(a):  # (n, k) -> (k, )
            return a.sum(axis=0) / a.shape[0]

        self._arr_mean_precision = mean(
            np.array([top.arr_precision for top in topics]))

        self._arr_mean_recall = mean(
            np.array([top.arr_recall for top in topics]))

        self._arr_mean_f1 = mean(
            np.array([top.arr_f1 for top in topics]))

        self._arr_mean_avg_precision = mean(
            np.array([top.arr_avg_precision for top in topics]))

    @property
    def arr_mean_precision(self):
        return self._arr_mean_precision

    @property
    def mean_precision(self):
        return self.arr_mean_precision[-1]

    @property
    def arr_mean_recall(self):
        return self._arr_mean_recall

    @property
    def mean_recall(self):
        return self.arr_mean_recall[-1]

    @property
    def arr_mean_f1(self):
        return self._arr_mean_f1

    @property
    def mean_f1(self):
        return self.arr_mean_f1[-1]

    @property
    def arr_mean_avg_precision(self):
        return self._arr_mean_avg_precision

    @property
    def mean_avg_precision(self):
        return self.arr_mean_avg_precision[-1]

    @property
    def arr_timedelta(self):
        return np.array([t.timedelta for t in self.stats.values()])

    @property
    def mean_timedelta(self):
        return np.mean(self.arr_timedelta)

    def __str__(self):
        sbuf = ['', 'vngol experiment'.upper(), '']
        hline = ['', '-' * 60, '']

        # print overview

        ovw_data = (
            ('mean precision',
             '{:.2f}'.format(self.mean_precision * 100), ),
            ('mean recall',
             '{:.2f}'.format(self.mean_recall * 100), ),
            # ('mean f1',
            #  '{:.2f}'.format(self.mean_f1 * 100), ),
            ('mean average precision',
             '{:.2f}'.format(self.mean_avg_precision * 100), ), )

        sbuf.append(tabulate(ovw_data))
        sbuf += hline

        # print table with data for tasks

        top_ids, tab_data = [], []

        for top_id, top_stats in self.stats.items():
            tab_headers, tab_row = top_stats.tab_data
            tab_data.append(tab_row)
            top_ids.append(top_id)

        # insert document names
        tab_cols = list(zip(*tab_data))
        tab_cols.insert(0, top_ids)
        assert len(tab_cols) == len(tab_headers) + 1

        tab_rows = list(zip(*tab_cols))
        tab_headers = ('topic', ) + tab_headers
        assert len(tab_rows) == len(tab_data)

        tab_rows.sort(key=lambda t: t[-1], reverse=True)
        sbuf.append(tabulate(tab_rows, headers=tab_headers))

        return '\n'.join(sbuf)

    # convenience for tabulate

    def tab_data_at(self, k: int):
        headers = 'μP', 'μR', 'μAP'  # 'μF1', 'μAP'

        row = (
            int(self.arr_mean_precision[k] * 100),
            int(self.arr_mean_recall[k] * 100),
            # int(self.arr_mean_f1[k] * 100),
            int(self.arr_mean_avg_precision[k] * 100), )

        assert len(set(headers)) == len(headers)
        assert len(headers) == len(row)
        return headers, row

    @property
    def tab_data(self):
        return self.tab_data_at(-1)
