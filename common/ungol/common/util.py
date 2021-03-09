# -*- coding: utf-8 -*-

from ungol.common import logger

import time
import datetime
import functools
import importlib

import numpy as np
from tabulate import tabulate
from tqdm import tqdm as _tqdm

from typing import List
from typing import Callable


tqdm = functools.partial(_tqdm, ncols=80)
log = logger.get('common.util')

# ---


def load_module(f_mod: str):
    mod = None
    spec = None

    try:
        spec = importlib.util.find_spec(f_mod)
    except ModuleNotFoundError:
        # raised "if package is in fact not a package
        # (i.e. lacks a __path__ attribute)."
        pass

    if spec is not None:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        log.error(f'Module "{f_mod}" not imported.')

    return mod


# ---


def ts(l, msg: str) -> Callable[[None], None]:
    tstart = datetime.datetime.now()

    def _done(*args, **kwargs):
        tdelta = (datetime.datetime.now() - tstart).total_seconds()
        l(msg.format(*args, delta=tdelta, **kwargs))

    return _done


class Timer:

    def get_mean(self, key, resolution='ms') -> float:
        factor = self._get_factor(resolution)
        deltas = self.get_deltas_microseconds(key) / factor
        return deltas.mean()

    def get_deltas(self, key) -> List[datetime.timedelta]:
        return self._tdeltas[key]

    def get_deltas_microseconds(self, key) -> np.array:
        return np.array([d.microseconds for d in self.get_deltas(key)])

    def _get_factor(self, resolution):
        return {'us': 1, 'ms': 1e3, 's': 1e6}[resolution]

    def __init__(self):
        self._tdeltas = {}

    def __getitem__(self, key):
        tstart = datetime.datetime.now()

        if key not in self._tdeltas:
            self._tdeltas[key] = []

        def _proxy():
            tdelta = datetime.datetime.now() - tstart
            self._tdeltas[key].append(tdelta)

        return _proxy

    def __str__(self) -> str:
        return self.as_table()

    def as_table(self, resolution: str = 'ms') -> str:
        factor = self._get_factor(resolution)

        data = ((key, self.get_deltas_microseconds(key) / factor)
                for key in self._tdeltas)

        cols = tuple((
            key,
            len(a),
            a.mean(),
            a.std()
        ) for key, a in data)

        headers = ('name', 'N', 'μ', 'σ', )
        s_buf = []

        s_buf.append(f'Measurement Result (resolution={resolution})')
        s_buf.append(tabulate(cols, headers=headers))

        return '\n' + '\n\n'.join(s_buf) + '\n'


def timer_example():
    timer = Timer()

    for _ in tqdm(range(100)):

        done = timer['longest']
        time.sleep(1e-2)
        done()

        done = timer['middle']
        time.sleep(1e-3)
        done()

        done = timer['shortest']
        time.sleep(1e-4)
        done()

    print(str(timer))


if __name__ == '__main__':
    print('running timer example')
    timer_example()
