
from ungol.common import logger

from tqdm import tqdm as _tqdm

import os
import pickle
import pathlib
import functools
import itertools
import multiprocessing as mp


log = logger.get('sentemb.common')
tqdm = functools.partial(_tqdm, ncols=80)


SENT_MIN_LEN = 2
SENT_MAX_LEN = 40

F_ARRS = 'sentences.arrs'
F_TOKS = 'tokens.txt'
F_VOCAB = 'vocab.pickle'
F_COUNTS = 'counts.pickle'


# ---


def get_vocabs(f_in: str, prefix: str = None):
    log.info(f'loading vocabulary files from "{f_in}" (prefix={prefix})')

    if prefix is None:
        prefix = ''
    else:
        prefix = prefix + '.'

    pth = pathlib.Path(f_in)
    with (pth / (prefix + F_VOCAB)).open(mode='rb') as fd:
        vocab = pickle.load(fd)

    with (pth / (prefix + F_COUNTS)).open(mode='rb') as fd:
        counts = pickle.load(fd)

    assert len(vocab) == len(counts)
    return vocab, counts


def write_arr(fd, arr):
    fd.write(repr(arr)[1:-1].strip())
    fd.write('\n')


def read_arr_file(fd, n: int = None):
    count = 0
    for line in fd:

        if n is not None and n <= count:
            break

        yield eval('[' + line + ']')
        count += 1


def read_arr(f_in: str, prefix: str = None, n: int = None):
    pth = pathlib.Path(f_in)
    if prefix is not None:
        prefix = prefix + '.'
    else:
        prefix = ''

    fname = (pth / (prefix + F_ARRS))
    log.info(f'loading "{str(fname)}"')

    with fname.open(mode='r') as fd:
        yield from read_arr_file(fd, n=n)


def read_toks_file(fd, n: int = None):
    count = 0
    for line in fd:

        if n is not None and n <= count:
            break

        yield tuple(line.strip().split(' '))


def read_toks(f_in: str, prefix: str = None, n: int = None):
    pth = pathlib.Path(f_in)
    if prefix is not None:
        prefix = prefix + '.'
    else:
        prefix = ''

    fname = (pth / (prefix + F_TOKS))
    log.info(f'loading "{str(fname)}"')

    with fname.open(mode='r') as fd:
        yield from read_toks_file(fd, n=n)


def _batcher(gen, n):
    while True:
        batch = tuple(itertools.islice(gen, n))
        if not len(batch):
            return

        yield batch


def read_toks_batches(*args, batch_size: int = 128, **kwargs):
    yield from _batcher(read_toks(*args, **kwargs), batch_size)


# ---


class Actor:

    def before(self):
        pass

    def after(self):
        pass

    def update(self, msg):
        pass


def _run_reader(reader, rq, processes: int):
    log.info(f'[{os.getpid()}] [reader] spawning')

    reader(rq=rq)
    for _ in range(processes):
        rq.put(None)

    log.info(f'[{os.getpid()}] [reader] dying')


def _run_worker(worker, wid, rq, wq, batch_size):
    log.info(f'[{os.getpid()}] [{wid}] spawning')
    worker.before()

    pbar = tqdm(desc=f'worker {wid}', position=wid+3)
    while True:
        msg = rq.get()
        if msg is None:
            break

        wq.put(worker.update(msg))
        pbar.update(batch_size)

    log.info(f'[{os.getpid()}] [{wid}] dying')
    wq.put(None)
    pbar.close()

    worker.after()


def _run_writer(actor, processes: int, wq, batch_size):
    actor.before()

    pbar = tqdm(desc='writer', position=processes+4)
    dead = 0

    while True:
        msg = wq.get()
        if msg is None:
            dead += 1
            if dead == processes:
                log.info('all worker died')
                break
            continue

        actor.update(msg)
        pbar.update(batch_size)

    pbar.close()
    actor.after()


def multiprocess(actors, processes: int, batch_size: int):
    print()
    reader, worker, writer = actors

    mgr = mp.Manager()
    rq, wq = mgr.Queue(), mgr.Queue()

    for i in range(processes):
        # no need to join on these as per protocol (death pill)
        args_worker = worker, i, rq, wq, batch_size
        p = mp.Process(target=_run_worker, args=args_worker)
        p.start()

    args_reader = reader, rq, processes
    p_reader = mp.Process(target=_run_reader, args=args_reader)
    p_reader.start()

    _run_writer(writer, processes, wq, batch_size)
    print('\n' * (processes + 4))
