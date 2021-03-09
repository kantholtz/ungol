# -*- coding: utf-8 -*-

"""
This module abstracts writing all files aggregating
data throughout training and evalution. There is
a specific writer implementation per to-be-gathered
statistic.

The StoG object determines which stats to gather based
on the provided configuration. Per statistic type a
StatWriter implementation is bound. Each writer is
dispatched in its own process.

Data is pulled from the model in the intervals specified
by the configuration.

"""

from ungol.common import logger
from ungol.common.util import ts
from ungol.common import embed as uce
from ungol.models import models as umm
from ungol.models import training as umt

import attr
import h5py
import torch
import numpy as np

import os
import math
import enum
import pathlib
import multiprocessing as mp

from typing import Any
from typing import List
from typing import Tuple
from typing import Union
from typing import Callable


# ---

log = logger.get('models.stats')

DEV_CPU = torch.device('cpu')
DEV_GPU = torch.device('cuda')

# ---


class Kind(enum.Enum):
    train = enum.auto()
    valid = enum.auto()
    flush = enum.auto()


@attr.s(frozen=True)
class Update:
    kind: Kind = attr.ib()
    step:  int = attr.ib()  # in [0, epochs // stog.*[ - internal addressing
    epoch: int = attr.ib()  # in [0, epochs] (+ phony run)
    batch: int = attr.ib()  # in [0, V // batch_size[
    last: bool = attr.ib()  # true for last batch of epoch


@attr.s
class StatsToGather:
    folder:  str = attr.ib()

    losses: Union[List[int], int] = attr.ib(default=0)
    codes:  Union[List[int], int] = attr.ib(default=0)

    grad_enc: int = attr.ib(default=0)  # currently unused
    grad_dec: int = attr.ib(default=0)  # currently unused

    model: Union[List[int]] = attr.ib(default=0)

    @staticmethod
    def from_conf(conf, dimensions, name):
        folder = conf['folder']
        folder = str(pathlib.Path(folder, name))

        mapping = {
            'losses': 'losses',
            'encoder gradients': 'grad_enc',
            'decoder gradients': 'grad_dec',
            'codes': 'codes',
            'model': 'model',
        }.items()

        def read_conf(k):
            try:
                ret = conf.as_int(k)
                log.info(' - stats: write %s every %s epoch(s)', k, ret)

            except TypeError:
                ret = [int(d) for d in conf.as_list(k)]
                log.info(' - stats: write %s in epochs: %s', k, str(ret))

            return ret

        return StatsToGather(
            folder,
            **{v: read_conf(k) for k, v in mapping})


class StatWriter():

    def __init__(self, q: mp.Queue):
        self.q = q

    def listen(self):
        epochs = {}

        while True:
            msg = self.q.get()
            if msg is None:
                log.info('[%d] received cyanide, dying', os.getpid())
                break

            up, data = msg

            if up.kind not in epochs:
                epochs[up.kind] = [up.epoch]
            elif epochs[up.kind][-1] != up.epoch:
                epochs[up.kind].append(up.epoch)

            self.receive(up, data)

        return epochs

    # --- utility

    def write_mapping(self, fd, mapping):
        for kind in mapping.keys():
            fd.attrs[kind.name] = mapping[kind]

    # --- interface (invoked inside own process)

    def init(self, folder: pathlib.Path, amount: int, data: Any) -> None:
        """
        Invoked once after instantiation.

          -- folder: pathlib.Path - place to put all files
          -- amount: int - amount of epochs the stog constraint holds
          -- data: Any - returned by prepare()

        """
        raise NotImplementedError()

    def receive(self, up: Update, data: Any) -> None:
        """
        Invoked everytime a batch was processed if
        the relevant StatsToGather constraint evaluates.

          -- up: Update - see Update object
          -- data: Any  - data provided by update()

        """
        raise NotImplementedError()

    def close(self, mapping):
        """
        Invoked after all write operations finished.

          -- mapping: maps steps to epochs, see write_mapping()

        """
        raise NotImplementedError()

    # --- interface (invoked from main process)

    @staticmethod
    def prepare(training: umt.Training) -> Tuple['ctx', Tuple[Any]]:
        """
        Invoked once before starting the corresponding process.
        Used to provide "Any data" to init() and offers with 'ctx'
        a persistent context object saved in the main process
        (e.g. for working directly with GPU memory).

        """
        raise NotImplementedError()

    @staticmethod
    def update(
            ctx: Any,
            up: Update,
            training: umt.Training,
            send: Callable[[Any], None]) -> None:
        """
        Used to read from the training instance. Use the provided
        send Callable to send data to the processes receive() method.
        In contrast to receive() - which is invoked only based on
        the stog constraint - this method is invoked for every batch
        in every epoch. This opens the possibility to interact with
        memory epoch-wise if necessary.

          -- ctx: Any context provided by prepare()
          -- up: see Update object
          -- training: umt.Training instance
          -- send: send(Any) data to receive()

        """
        raise NotImplementedError()


class LossWriter(StatWriter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # --- invoked in own process

    def init(
            self,
            folder: pathlib.Path,
            amount: int,
            batch_count: int):

        log.info(f'[{os.getpid()}] initializing LossWriter')

        self.fd = h5py.File(str(folder/'losses.h5'), 'w')

        self.ds_train = self.fd.create_dataset(
            'train', (amount, batch_count), dtype='float')

        self.ds_valid = self.fd.create_dataset(
            'valid', (amount, batch_count), dtype='float')

    def receive(self, up: Update, data: Any):
        a = self.ds_train if up.kind == Kind.train else self.ds_valid
        a[up.step, up.batch] = data

        if up.last:
            log.info('[%d] average loss %s %s [ %3.5f ] <<<',
                     os.getpid(), ' ' * 20, up.kind.name, a[up.step][:].mean())

    def close(self, mapping):
        self.write_mapping(self.fd, mapping)
        self.fd.close()
        log.info('[%d] closed loss writer', os.getpid())

    # --- invoked from main process

    @staticmethod
    def prepare(training) -> Tuple['ctx', int]:
        return None, training.loader.batch_count

    @staticmethod
    def update(
            ctx: Any,
            up: Update,
            training: umt.Training,
            send: Callable[[Any], None]) -> None:

        if up.kind == Kind.flush:
            return

        send(training.loss)


class CodeWriter(StatWriter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # --- invoked in own process

    def init(self, folder: pathlib.Path, amount: int, data: Any):
        log.info('[%d] initializing CodeWriter', os.getpid())

        self.N, self.M, self.K = data

        self.fd = h5py.File(str(folder/'codes.h5'), 'w')
        self.grp_train = self.fd.create_group('train')
        self.grp_valid = self.fd.create_group('valid')

    def _entropy(self, data):
        entropies = []

        for row in data:
            entropy = sum([
                0 if x == 0 else -x * math.log(x, 2)
                for x in row / row.sum()])

            entropies.append(entropy / math.log(data.shape[-1], 2))

        return np.array(entropies)

    # ---

    def update_train(self, grp, up, data):
        # grp.create_dataset('raw', data=data)

        # histogram
        bins = int(1e2)
        hist, bin_edges = np.histogram(data.flat, bins=bins, range=(0, 1))
        ds_hist = grp.create_dataset('histogram', data=hist)
        ds_hist.attrs['bin_edges'] = bin_edges

        # entropy
        entropy = self._entropy(ds_hist[:].reshape(-1, bins))
        ds_entropy = grp.create_dataset('entropy', data=entropy)
        ds_entropy.attrs['mean'] = entropy.mean()

    def update_valid(self, grp, up, data):
        n, M, K = data.shape

        codes = data.nonzero()[2]
        codes = codes.reshape(-1, M)
        codes = codes.astype(np.uint)

        assert codes.shape[0] == data.shape[0]
        assert codes.shape[1] == data.shape[1]
        assert codes.max() < data.shape[2], codes.max()

        # grp.create_dataset('raw', data=codes)

        # histogram per codebook
        ds_counts = grp.create_dataset('counts', shape=(M, K))
        for i in range(M):
            sel = codes[:, i]
            a = np.histogram(sel, bins=K, range=(0, K-1))[0]
            ds_counts[i] = a

        assert ds_counts.shape[0] == M
        assert ds_counts.shape[1] == K

        # entropy
        entropy = self._entropy(ds_counts[:])
        ds_entropy = grp.create_dataset('entropy', data=entropy)
        ds_entropy.attrs['mean'] = entropy.mean()

    def receive(self, up: Update, buf: np.array):
        log.info('[%d] writing %s codes', os.getpid(), up.kind.name)
        done = ts(log.info, '[{pid}] writing codes took {delta}s')

        # the sender works with fixed buffer sizes, thus
        # the data must be cropped at some point:
        data = buf.reshape((-1, self.M, self.K))[:self.N]

        if up.kind == Kind.train:
            grp = self.grp_train.create_group(str(up.epoch))
            self.update_train(grp, up, data)

        if up.kind == Kind.valid:
            grp = self.grp_valid.create_group(str(up.epoch))
            self.update_valid(grp, up, data)

        done(pid=os.getpid())

    # ---

    def close(self, mapping):
        self.write_mapping(self.fd, mapping)
        self.fd.close()
        log.info('[%d] closed code writer', os.getpid())

    # --- invoked from main process

    @staticmethod
    def prepare(training) -> Tuple['ctx', None]:
        dims = training.t.model.dimensions
        device = training.t.device
        data = training.data

        ctx = [

            # training codes buffer
            torch.zeros(
                data.batch_count_train,
                data.batch_size,
                dims.components,
                dims.codelength).to(device=device),

            # validation codes buffer
            torch.zeros(
                data.batch_count_valid,
                data.batch_size,
                dims.components,
                dims.codelength).to(device=device),
        ]

        return ctx, (len(data.embed), dims.components, dims.codelength)

    @staticmethod
    def update(
            ctx: Any,
            up: Update,
            training: umt.Training,
            send: Callable[[Any], None]):

        if up.kind == Kind.flush:
            return

        if up.kind == Kind.train:
            buf = ctx[0]

        if up.kind == Kind.valid:
            buf = ctx[1]

        # codes.shape: (n, M, K)
        codes = training.t.model.encoder_codes

        # buf.shape: (batch, n, M, K)
        buf[up.batch][:codes.shape[0]] = codes

        # flush buffer at the very last batch
        if up.last:
            # log.info('sending code batches')
            send(buf.cpu().numpy())


class ModelWriter(StatWriter):

    def init(self, folder, amount, _):
        self.path = pathlib.Path(folder / 'compressor')
        self.path.mkdir(exist_ok=True)

    def receive(self, up: Update, data: Tuple['state_dict', bytes, bytes]):
        f_model = self.path / 'model-{}.torch'.format(up.epoch)

        state_dict, json_dims, json_stog = data
        torch.save(state_dict, f_model)

        f_dims_json = 'dimensions.json'
        with (self.path / f_dims_json).open('w') as f:
            f.write(json_dims)

        f_dims_stog = 'stog.json'
        with (self.path / f_dims_stog).open('w') as f:
            f.write(json_stog)

        log.info('[%d] persisted model to %s',
                 os.getpid(),
                 str('/'.join(f_model.parts[-3:])))

    def close(self, mapping):
        pass

    # ---

    @staticmethod
    def prepare(training) -> Tuple['ctx', None]:
        return None, None

    @staticmethod
    def update(
            ctx: Any,
            up: Update,
            training: umt.Training,
            send: Callable[[Any], None]) -> None:

        if up.kind == Kind.flush or (up.step is not None and up.last):
            ice = training.t.model.freeze()
            send(ice)

# ---


class Multiplexer():
    """
    Class handling all interprocess communication.
    Every StatWriter is offered access to the
    necessary queues. Here all is wired together.
    """

    @attr.s
    class Writer:
        stog:            int = attr.ib()
        klass:    StatWriter = attr.ib()

        q:   mp.Queue = attr.ib()
        p: mp.Process = attr.ib()

        ctx:     Any = attr.ib()

        counter: int = attr.ib(default=0)

    # ---

    def _stog_to_list(self,
                      arg: Union[List[int], int],
                      epochs: int) -> List[int]:

        if type(arg) == list:
            return arg

        if arg == 0:
            return []

        def cond(epoch) -> bool:
            return any((
                epoch == 0,
                epoch == epochs,
                epoch % arg == 0))

        return list(filter(cond, range(epochs + 1)))

    def __init__(self, stog: StatsToGather, training, epochs):
        log.info('initializing statistics multiplexer')

        self._stog = stog
        pathlib.Path(stog.folder).mkdir(exist_ok=True, parents=True)

        # registered writer classes

        _writer = [
            dict(stog=stog.losses, klass=LossWriter, ),
            dict(stog=stog.codes, klass=CodeWriter, ),
            dict(stog=stog.model, klass=ModelWriter, ),
        ]

        self._writer_args = list(filter(lambda v: v['stog'] != 0, _writer))
        for arg in self._writer_args:
            arg['stog'] = self._stog_to_list(arg['stog'], epochs)

        self._dispatch(training, epochs)

    def _dispatch(self, training, epochs):
        log.info('dispatching Multiplexer')

        def proxy(Klass, q, raw_args):
            """
              -- Klass: Statwriter - to be instantiated
              -- q: mp.Queue - for communication with the instance
              -- raw_args - (path, epochs, ...)
            """
            # prepare
            path = pathlib.Path(raw_args[0])
            args = (path, ) + raw_args[1:]

            # initialize
            instance = Klass(q)
            instance.init(*args)

            # dispatch
            mapping = instance.listen()
            instance.close(mapping)

        # ---

        self._writer = []
        for writer_args in self._writer_args:
            q = mp.Queue()
            amount = len(writer_args['stog'])

            # aggregate arg vectors
            ctx, prepped_args = writer_args['klass'].prepare(training)
            klass_args = self._stog.folder, amount, prepped_args
            proxy_args = (writer_args['klass'], q, klass_args, )

            # start initialization in seperate process
            p = mp.Process(target=proxy, args=proxy_args)
            p.start()

            self._writer.append(Multiplexer.Writer(
                ctx=ctx, q=q, p=p, **writer_args))

    # ---

    def _send_to_writer(self, writer, up, training):
        def send(data):
            if up.kind == Kind.flush or up.step is not None:
                writer.q.put((up, data))
                writer.counter += 1

        writer.klass.update(writer.ctx, up, training, send)

    def flush(self, training, epoch=-1):
        log.info('flushing stats')

        # this might be solved better in the future
        # - maybe attach epoch/batch etc. info to the exception?
        up = Update(
            kind=Kind.flush,
            epoch=epoch,
            batch=-1,
            step=None,
            last=-1)

        for writer in self._writer:
            self._send_to_writer(writer, up, training)

    def close(self):
        log.info('sending cyanide to stat writers')
        for writer in self._writer:
            writer.q.put(None)
            writer.p.join()

    def ping(self, kind, training, epoch, batch):
        """

        invoked batch-wise!

        """

        assert epoch >= 0, epoch
        assert batch >= 0, batch

        batch_count = training.loader.batch_count

        if batch >= batch_count:
            log.error(f'ping() exceeded bounds! batch={batch} ({batch_count})')
            return

        for writer in self._writer:

            try:
                step = writer.stog.index(epoch)
            except ValueError:
                step = None

            up = Update(
                kind=kind,
                epoch=epoch,
                batch=batch,
                step=step,
                last=batch == batch_count - 1, )

            self._send_to_writer(writer, up, training)


# def _handle_sigint(proc: mp.Process, statq: mp.Queue):
#     def _handler(_s, _f):
#         log.info('process was interrupted by SIGINT, waiting for writer')
#         StatWriter.close(proc, statq)

#         log.info('exiting by interrupt')
#         sys.exit(0)

#     log.info('registering SIGINT handler')
#     signal.signal(signal.SIGINT, _handler)


class EmberMock():

    def get_vocab(self):
        return range(25)


class ModelMock():

    @property
    def encoder_codes(self):  # -> (n, M, K)
        n = self.batch_size
        M = self.dimensions.components
        K = self.dimensions.codelength

        if self.train:
            arr = torch.arange(0, 1, 1 / (n * M * K))
            return arr.view((n, M, K))
        else:
            arr = torch.zeros(n * M * K).reshape((n, M, K))

            # split to gain a bit of entropy
            arr[:n // 3, :, 0] = 1
            arr[n // 3:, :, 1] = 1

            return arr

    def __init__(self, dimensions, batch_size, epochs):
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.epochs = epochs


def _test_setup(epochs, dimensions, batch_size):
    # FIXME replace Ember

    loader = uce.Loader(
        batch_size=batch_size,
        ember=EmberMock(),
        device=DEV_CPU)

    params = umt.Parameters(
        learningrate=0, momentum=0,
        gumbel_scale=[1, None, None],
        gumbel_temperature=[1, None, None])

    t = umt.Training.Torch(
        model=ModelMock(dimensions, batch_size, epochs),
        optimizer=None,
        criterion=None,
        device=None)

    training = umt.Training(
        loader=loader, t=t, params=params)

    print('initializing')
    return training


def _test_training(epochs, samples, training, stats):
    batch_count = samples // training.loader.batch_size

    # training
    for epoch in range(epochs + 1):  # +1 because of phony run
        for batch in range(batch_count):
            training.t.model.train = True
            training._loss = epoch + batch / batch_count
            stats.ping(Kind.train, training, epoch, batch)

    # validation
    for epoch in range(epochs + 1):  # +1 because of phony run
        for batch in range(batch_count):
            training.t.model.train = False
            training._loss = epoch + batch / batch_count
            stats.ping(Kind.valid, training, epoch, batch)


def _assert_losses(stog: StatsToGather):
    with h5py.File(stog.folder + '/losses.h5', 'r') as fd:
        # every 2 epochs + phony epoch and last epoch
        exp_dpoints = 1 + math.ceil(epochs / stog.losses)

        print('train shape:', fd['train'].shape)
        print('mapping train', fd.attrs['train'])

        assert fd['train'].shape[0] == exp_dpoints
        assert len(fd.attrs['train']) == exp_dpoints

        print('valid shape:', fd['valid'].shape)
        print('mapping valid', fd.attrs['valid'])

        assert fd['valid'].shape[0] == exp_dpoints
        assert len(fd.attrs['valid']) == exp_dpoints

        # samples

        # epoch + batch / batch_count
        batch_count = samples // training.loader.batch_size
        print(fd['train'][2][3], 4 + 3 / batch_count)
        assert fd['train'][2][3] == 4 + 3 / batch_count
        assert fd['valid'][2][3] == 4 + 3 / batch_count

    print('\n>>> assert losses - no errors')


def _assert_codes(stog: StatsToGather):
    # FIXME write assertions

    with h5py.File(stog.folder + '/codes.h5', 'r') as fd:
        print('groups')
        for kind in fd:
            print(' ' * 2, kind)
            for epoch in fd[kind]:
                print(' ' * 4, epoch, fd[kind][epoch]['raw'].shape)

        print('attrs')
        for k in fd.attrs:
            print(' ' * 2, k)

    print('\n>>> assert codes - no errors')


if __name__ == '__main__':
    log.info('-' * 40)

    epochs = 10
    samples = 25
    batch_size = 5

    dimensions = umm.Dimensions(
        embedding=300,  # E
        components=3,   # M (components)
        codelength=2,   # K (codomain)
        fcl1=-1, )

    stog = StatsToGather(
        folder='opt/integration/stats',
        losses=2,
        codes=5, )

    training = _test_setup(epochs, dimensions, batch_size)
    stats = Multiplexer(stog, training, epochs)

    _test_training(epochs, samples, training, stats)
    stats.close()

    # run tests
    _assert_losses(stog)
    _assert_codes(stog)
