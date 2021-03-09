# -*- coding: utf-8 -*-

from ungol.common.util import ts

from ungol.common import logger
from ungol.common import embed as uce
from ungol.models import stats as ums
from ungol.models import models as umm

import attr
import torch
import configobj
import numpy as np
from tqdm import tqdm as _tqdm

import math
import pathlib
import functools

from typing import Any
from typing import Dict
from typing import List
from typing import Generator


# ---

log = logger.get('models.training')
tqdm = functools.partial(_tqdm, ncols=80, disable=False)

# ---

DEV_CPU = torch.device('cpu')
DEV_GPU = torch.device('cuda')

# ---


#
#   TRAINING INFRASTRUCTURE
#


class EarlyStopping(Exception):

    @property
    def epoch(self) -> int:
        return self._epoch

    def __init__(self, msg: str, epoch: int):
        super().__init__(msg)
        self._epoch = epoch


@attr.s
class Training:
    """

    Encapsulates training preparation, batching, optimization and
    the training routine itself

    """

    @property
    def loss(self) -> float:
        return self._loss

    @property
    def train_losses(self) -> List[float]:
        return self._train_losses

    @property
    def valid_losses(self) -> List[float]:
        return self._valid_losses

    @attr.s
    class Torch:
        model:           torch.nn.Module = attr.ib()
        criterion:       torch.nn.Module = attr.ib()
        optimizer: torch.optim.Optimizer = attr.ib()
        device:             torch.device = attr.ib()

    # ---

    t:              Torch = attr.ib()
    loader:    uce.Loader = attr.ib()
    params:  'Parameters' = attr.ib()

    # ---

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.t.model(x)
        assert y.shape == x.shape, str(y.shape) + ' =/= ' + str(x.shape)

        # --------------------
        # _print_cgraph(self.t.model, y.grad_fn)
        # import sys; sys.exit()
        # --------------------

        loss = self.t.criterion(y, x)
        return loss

    def train(self, epoch: int, x: torch.Tensor) -> float:
        """
        Train the network by providing one batch of embeddings.

        """
        self.t.optimizer.zero_grad()
        loss = self._forward(x)
        loss.backward()

        self.t.optimizer.step()
        return loss.item()

    def validate(self, epoch: int, x: torch.Tensor) -> float:
        return self._forward(x).item()

    def _epoch_train(self, epoch: int) -> float:
        self.t.model.train()

        batches = self.loader.batch_count
        done = ts(log.info, 'training epoch took {delta}s')

        # ---

        it = enumerate(self.loader.gen())
        gen = tqdm(it, desc='training', unit=' batches', total=batches)

        losses = []
        for i, x in gen:
            if i >= batches:
                log.error(f'skipping unexpected batch {i}')
                break

            self._loss = self.train(epoch, x)
            self._stats.ping(ums.Kind.train, self, epoch, i)
            losses.append(self._loss)

        # ---

        self._train_losses.append(np.array(losses).mean())
        done()

    def _epoch_validate(self, epoch: int) -> float:
        self.t.model.eval()

        batches = self.loader.batch_count
        done = ts(log.info, 'validation took {delta}s')

        # ---

        it = enumerate(self.loader.gen())
        gen = tqdm(it, desc='validation', unit=' batches', total=batches)

        losses = []
        for i, x in gen:
            if i >= batches:
                log.error(f'skipping unexpected batch {i}')
                break

            self._loss = self.validate(epoch, x)
            self._stats.ping(ums.Kind.valid, self, epoch, i)
            losses.append(self._loss)

        # ---

        self._valid_losses.append(np.array(losses).mean())
        done()

    def epoch(self, epoch: int):
        """
        Iterate over all embeddings and feed batches to train().
        Calculates training metrics on the way.

        """
        log.info('-' * 20)
        log.info('starting training epoch >> [%d] <<', epoch)

        # --------------------
        # TODO annealing refactoring
        # dynamic parameter adjustment; think about having the
        # model re-configuring itself: knowledge about this does not
        # have to be exposed on this level. Problem: model has to be
        # informed about which epoch is running and also has to save
        # the Model.Parameters
        #
        gumbel = self.t.model.encoder.gumbel

        params = (
            (self.params.gumbel_scale,
             gumbel.set_sampler,
             'scale', ),
            (self.params.gumbel_temperature,
             gumbel.set_temperature,
             'tau', ), )

        for value, update, kwarg in params:
            curr, step, fn = value
            if step is None or epoch % step != 0:
                continue

            log.info('updating parameters (formerly %f)', curr)
            value[0] = fn(curr, epoch)
            update(**{kwarg: value[0]})

        # ODOT
        # --------------------

        self._epoch_train(epoch)
        self._epoch_validate(epoch)

    # ---

    def _phony_run(self):
        log.info('commencing a phony run without training')

        self.t.model.train()
        gen = enumerate(self.loader.gen())
        for i, x in tqdm(gen, total=self.loader.batch_count):
            self._loss = self._forward(x).item()
            self._stats.ping(ums.Kind.train, self, 0, i)

        self.t.model.eval()
        gen = enumerate(self.loader.gen())
        for i, x in tqdm(gen, total=self.loader.batch_count):
            self._loss = self._forward(x).item()
            self._stats.ping(ums.Kind.valid, self, 0, i)

    def _early_stopping(self, epoch):
        n, threshold = self.params.early_stopping
        losses = self.valid_losses

        if len(losses) < n:
            return

        a = np.array(losses)[-n:]
        delta = a.mean() - a[-n//2:].mean()

        log.info(f'current loss development: Δ={delta:.5f}')

        if delta < threshold:
            raise EarlyStopping(
                f'Reached early stopping criterion ({delta:.5f})',
                epoch)

    def _run_training(self, epochs: int, stats: 'ums.Multiplexer'):
        """
        Start the training over the provided amount of epochs.

        """
        log.info('running training with %d epochs', epochs)
        self._stats = stats

        # do a phony run for statistics
        self._phony_run()

        exc_counter = 0

        # commence training
        log.info('running %d epochs', epochs)
        for no in range(1, epochs + 1):
            if not tqdm.keywords['disable']:
                print('\nepoch {}'.format(no))

            try:
                self.epoch(no)
                self.t.model.encoder.gumbel.do_resample()

            except umm.ModelException as exc:
                exc_counter += 1
                log.info(f'received model exception {exc_counter}/10')
                if exc_counter >= 10:
                    raise exc

            self._early_stopping(no)

    def run_training(self, epochs: int, stats: 'ums.Multiplexer'):
        self._train_losses, self._valid_losses = [], []

        try:
            self._run_training(epochs, stats)

        # exception proxy to gracefully shut down processes
        except Exception as exc:
            log.error(f'catched exception "{str(exc)}"')

            try:
                stats.flush(self, epoch=exc.epoch)
            except AttributeError:
                stats.flush(self)

            stats.close()
            raise exc


@attr.s
class Parameters:
    learningrate:       float = attr.ib()
    momentum:           float = attr.ib()

    # List[float, None, None] or
    # List[float, float, lambda x: ...]
    gumbel_scale:       List[Any] = attr.ib()
    gumbel_temperature: List[Any] = attr.ib()

    encoder_init = attr.ib()
    early_stopping = attr.ib()

    decoder_init_factor = attr.ib(default=None)

    @staticmethod
    def from_conf(conf: configobj.Section):

        def _read_param(conf, key: str) -> List[Any]:
            try:
                return [conf.as_float(key), None, None]

            except TypeError:
                mapping = zip([float, float, eval], conf.as_list(key))
                return [f(x) for f, x in mapping]

        # ---

        early_stopping = None
        if 'early stopping' in conf:
            early_stopping = (
                int(conf['early stopping'][0]),
                float(conf['early stopping'][1]), )

        # ---

        encoder_init = (None, )
        if 'encoder init' in conf:
            method, *args = conf['encoder init']
            if method == 'uniform':
                args = float(args[0]), float(args[1])

            encoder_init = (method, ) + tuple(args)

        decoder_init_factor = None
        if 'decoder init factor' in conf:
            decoder_init_factor = float(conf['decoder init factor'])

        # ---

        return Parameters(
            learningrate=_read_param(conf, 'learningrate'),
            momentum=conf.as_float('momentum'),
            gumbel_scale=_read_param(conf, 'gumbel scale'),
            gumbel_temperature=_read_param(conf, 'gumbel temperature'),
            early_stopping=early_stopping,
            encoder_init=encoder_init,
            decoder_init_factor=decoder_init_factor)


@attr.s
class Config:

    epochs:         int = attr.ib()
    batch_size:     int = attr.ib()
    ram_chunk_size: int = attr.ib()
    gpu_chunk_size: int = attr.ib()
    noise_buffer:   int = attr.ib()

    embed:  Dict[str, Any] = attr.ib()
    device: torch.device = attr.ib()

    stog:     'ums.StatsToGather' = attr.ib()
    parameters:        Parameters = attr.ib()
    dimensions:  'umm.Dimensions' = attr.ib()

    @staticmethod
    def create(
            fname: str,
            selection: List[str] = None,
            device: torch.device = DEV_CPU) -> Generator[
                configobj.ConfigObj, None, None]:

        parser = configobj.ConfigObj(fname)

        # check

        assert 'defaults' in parser, 'You need to supply a [defaults] section'
        c_defaults = parser['defaults']

        assert_fmt = 'You need to supply a [{}] section in [defaults]'
        for section in 'general', 'statistics', 'dimensions', 'parameters':
            assert section in c_defaults, assert_fmt.format(section)

        # read

        for experiment in parser:
            if experiment == 'defaults':
                continue

            if selection and experiment not in selection:
                continue

            log.info('-' * 40)
            log.info('reading experiment "{}"'.format(experiment))

            # select

            c_merged = parser['defaults']
            c_merged.merge(parser[experiment])

            c_general = c_merged['general']
            c_stats = c_merged['statistics']

            # initialize

            dimensions = umm.Dimensions.from_conf(c_merged['dimensions'])
            stog = ums.StatsToGather.from_conf(c_stats, dimensions, experiment)
            parameters = Parameters.from_conf(c_merged['parameters'])

            conf = Config(
                epochs=c_general.as_int('epochs'),
                batch_size=c_general.as_int('batch size'),
                ram_chunk_size=c_general.as_int('ram chunk size'),
                gpu_chunk_size=c_general.as_int('gpu chunk size'),
                noise_buffer=c_general.as_int('noise buffer'),
                embed=c_merged['embedding'],
                device=device,
                stog=stog,
                parameters=parameters,
                dimensions=dimensions, )

            conf._parser = parser
            yield experiment, conf

    def write(self):
        path = pathlib.Path(self.stog.folder, 'embcompr.conf')
        with path.open('wb+') as f:
            log.info('writing configuration to "{}"'.format(str(path)))
            self._parser.write(f)


def train(conf: Config, init):
    """
      E -- embedding dimensions
      M -- codebook count
      K -- code components
    """
    log.info('train() compressor')

    # initialization

    embed = uce.create(uce.Config.from_conf(conf.embed))
    conf.dimensions.embedding = embed.dimensions
    model, optimizer, criterion = init(conf, embed)

    # configuration

    loader = uce.Loader(
        batch_size=conf.batch_size,
        ram_chunk_size=conf.ram_chunk_size,
        gpu_chunk_size=conf.gpu_chunk_size,
        embed=embed,
        device=conf.device, )

    torch = Training.Torch(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=conf.device)

    training = Training(
        loader=loader, t=torch,
        params=conf.parameters)

    stats = ums.Multiplexer(conf.stog, training, conf.epochs)
    conf.write()

    training.run_training(conf.epochs, stats)

    # finishing
    stats.close()
    training.loader.close()
    return training
