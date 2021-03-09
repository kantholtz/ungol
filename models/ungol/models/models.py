# -*- coding: utf-8 -*-


from ungol.common import logger
from ungol.common import embed as uce
from ungol.models import stats as ums
from ungol.common.util import ts

import attr
import torch
import configobj
import numpy as np

import os
import json
import queue
import pathlib
import functools
import collections
import multiprocessing as mp
from datetime import datetime

from typing import Tuple

# ---

log = logger.get('models.models')

# ---

DEV_CPU = torch.device('cpu')
DEV_GPU = torch.device('cuda')

log.warn('enabling cudnn benchmark')
torch.backends.cudnn.benchmark = True

# ---


class ModelException(Exception):
    """

    Interrupts training

    """
    def __init__(self, msg):
        super().__init__(msg)


#
#   COMMON BUILDING BLOCKS
#


def detect_device(no_gpu: bool):
    if not torch.cuda.is_available() or no_gpu:
        if no_gpu:
            log.info('forcing to train on CPU')
        else:
            log.info('no CUDA support detected')

        return DEV_CPU

    log.info('detected CUDA support')
    return DEV_GPU


@attr.s
class Dimensions():

    components:  int = attr.ib()  # the amount of components
    codelength:  int = attr.ib()  # code component codomain
    fcl1:        int = attr.ib()  # neurons of the first hidden layer

    # this is optional to be initialized later after uce.Embed is loaded
    embedding:   int = attr.ib(default=None)  # number of embedding dimensions

    @staticmethod
    def from_conf(conf: configobj.Section):

        if 'embedding' in conf:
            log.warning('config: "embedding" is ignored - please remove!')

        M, K = conf.as_int('components'), conf.as_int('codelength')
        fcl1 = (M * K // 2) if 'fcl1' not in conf else conf.as_int('fcl1')

        return Dimensions(components=M, codelength=K, fcl1=fcl1)


class DistanceLoss(torch.nn.Module):

    def __init__(self, p=2):
        super().__init__()
        # self._dist = torch.nn.PairwiseDistance(p=p)

    def forward(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # loss = self._dist(y, x).mean()
        # assert not np.isnan(loss.item()), 'y={}, x={}'.format(y, x)

        losses = 0.5 * torch.sum((y - x) ** 2, dim=1)
        loss = torch.mean(losses)

        return loss


# ---

#
#   BASE
#


class Model(torch.nn.Module):

    @property
    def dimensions(self) -> Dimensions:
        return self._dimensions

    @property
    def stog(self):
        return self._stog

    @property
    def device(self):
        return self._device

    def __init__(self,
                 dimensions: Dimensions,
                 stog,
                 device):

        super().__init__()

        self._dimensions = dimensions
        self._stog = stog
        self._device = device

        # check if methods are implemented
        self.name

    # ---

    def freeze(self) -> Tuple['state_dict', bytes, bytes]:
        state_dict = {k: v.cpu() for k, v in self.state_dict().items()}
        json_dims = json.dumps(attr.asdict(self.dimensions), indent=2)
        json_stog = json.dumps(attr.asdict(self.stog), indent=2)

        return state_dict, json_dims, json_stog

    def save(self, folder: str, suffix: str = ''):
        path = pathlib.Path(folder)
        f_model = path / '{}-{}.torch'.format(self.name, suffix)

        state_dict, json_dims, json_stog = self.freeze()
        torch.save(state_dict, f_model)

        f_dims_json = '{}-dimensions.json'.format(self.name)
        with (path / f_dims_json).open('w') as f:
            f.write(json_dims)

        f_dims_stog = '{}-stog.json'.format(self.name)
        with (path / f_dims_stog).open('w') as f:
            f.write(json_stog)

        log.info('persisted model to {}'.format(
                str('/'.join(f_model.parts[-3:]))))

    @staticmethod
    def _load(
            Klass: 'Model',
            folder: str,
            fname: str,
            device: torch.device = None) -> 'Model':

        if device is None:
            device = DEV_CPU

        path = pathlib.Path(folder)

        with (path / 'dimensions.json').open('r') as f:
            dimensions = Dimensions(**json.loads(f.read()))
            log.info('loading dimensions: %s', str(dimensions))

        with (path / 'stog.json').open('r') as f:
            stog = ums.StatsToGather(**json.loads(f.read()))

        model = Klass(dimensions=dimensions, stog=stog, device=device)
        model.load_state_dict(torch.load(str(path / fname)))

        return model

    @staticmethod
    def load(folder: str, fname: str, device: torch.device = None) -> 'Model':
        raise NotImplementedError()


#
#   GUMBEL SOFTMAX TRICK LAYER
#


class AsyncSampler():

    # to accelerate startup time, use a formerly created noise file
    INIT = 'opt/.noise.{}'

    @staticmethod
    def _write_buf(f_name: str, arr: np.array):
        with open(f_name, mode='wb') as fd:
            log.info(f'[{os.getpid()}] writing "{f_name}" for later use')
            done = ts(log.info, '[{pid}] writing took {delta}s')

            arr.tofile(fd)

            done(pid=os.getpid())

    @staticmethod
    def _read_buf(f_name: str, buf: mp.Array):
        with open(f_name, mode='rb') as fd:
            log.info(f'[{os.getpid()}] loading cached "{f_name}"')
            done = ts(log.info, '[{pid}] loading took {delta}s')

            arr = np.fromfile(fd, dtype=np.float32)
            AsyncSampler._atobuf(buf, arr)

            done(pid=os.getpid())

    @staticmethod
    def _atobuf(buf: mp.Array, arr: np.array, pos: int):
        # copy raw values to the shared memory chunk
        a_buf = np.frombuffer(buf.get_obj(), dtype=np.float32)
        # np.copyto(a_buf, arr)
        a_buf[pos:pos + len(arr)] = arr

    @staticmethod
    def _resample(
            q: mp.Queue,
            pos: int,
            size: int,
            buf: mp.Array,
            loc: float,
            scale: float) -> np.array:

        _ts = datetime.now()

        # dtype=torch.float32, re-seeding b/c of forking
        torch.manual_seed(os.getpid())
        samples = torch.distributions.gumbel.Gumbel(
            torch.Tensor([loc]),
            torch.Tensor([scale])).sample(sample_shape=(size, ))

        arr = samples.squeeze().numpy()
        AsyncSampler._atobuf(buf, arr, pos)

        _delta = datetime.now() - _ts
        q.put(_delta.total_seconds())

        # ---
        # save array for next runs
        # pth = pathlib.Path(AsyncSampler.INIT.format(bufsize))
        # if not pth.exists():
        #     AsyncSampler._write_buf(str(pth), arr)

    def __init__(self, bufsize: int, loc: float = 0.0, scale: float = 1.0):
        log.info(f'create async sampler (loc={loc:.3f}, scale={scale:.3f})')

        self.buf = mp.Array('f', bufsize)
        self.bufsize = bufsize
        self._processes = 16  # FIXME make config option
        self._q = mp.Queue()

        assert self.bufsize % self._processes == 0
        self._chunksize = self.bufsize // self._processes
        self._sample_args = self.buf, loc, scale

        # try:
        #     f_name = AsyncSampler.INIT.format(bufsize)
        #     AsyncSampler._read_buf(f_name, self.buf)

        # except FileNotFoundError:
        #     self._create_spare()
        self._create_spare()

    def _create_spare(self):
        self._p = []
        for pos in range(0, self.bufsize, self._chunksize):
            args = (self._q, pos, self._chunksize) + self._sample_args

            p = mp.Process(
                target=AsyncSampler._resample,
                args=args)

            p.start()
            self._p.append(p)

        log.info(f'dispatched {len(self._p)} sampler processes')

    def _retrieve_from_queue(self):
        times = []

        try:
            timeout = 60
            n = len(self._p)

            log.info(f'retrieving {n} stat messages'
                     f'({timeout}s timeout)')

            times = [self._q.get(timeout=timeout) for _ in self._p]

        except queue.Empty:
            log.error(f'received only {len(times)}/{n}'
                      'messages from sampler processes!')

        avg_time = np.array(times).mean()
        log.info(f'sampling took {avg_time:.3f}s on average')

    def _join_processes(self):
        n = len(self._p)
        # log.info(f'joining on {n} sampler processes')
        # the timeout returns from .join() regardless whether
        # the process finished or...
        # [p.join(timeout=60) for p in self._p]
        log.info(f'sending SIGTERM to {n} processes')
        [p.terminate() for p in self._p]

    @property
    def samples(self) -> np.array:
        done = ts(log.info, 'joining on sampler processes took {delta}s')

        try:
            self._retrieve_from_queue()
            self._join_processes()

        except AttributeError:
            raise Exception('no _create_spare() invoked')

        done()

        samples = np.frombuffer(self.buf.get_obj(), dtype=np.float32)[:]
        self._create_spare()
        return samples


class Gumbel(Model):
    """
    You must initialize the sampler and temperature seperately
    with `set_sampler` and `set_temperature`.
    """

    name = 'gumbel'
    NOISE_BUFFER = int(2**30)

    @attr.s
    class Buffer:
        size:     int = attr.ib()

        device = attr.ib()
        data: torch.Tensor = attr.ib(default=None)

        position: int = attr.ib(default=0)
        reused:   int = attr.ib(default=0)

        def __attrs_post_init__(self):
            log.info('allocating space for the noise buffer')
            self.data = torch.zeros((self.size, )).to(device=self.device)

    @property
    def buf(self):
        return self._buf

    def set_temperature(self, tau: float = 1.0):
        log.info('set softmax temperature to %f', tau)
        self._temperature = tau

    # ---  sampling

    def set_sampler(self, loc: float = 0.0, scale: float = 1.0):
        log.info('set gumbel sampler to loc = %f, scale = %f', loc, scale)
        self._sampler = AsyncSampler(self.buf.size, loc, scale)
        self._exchange_buffer()

    def _exchange_buffer(self):
        done = ts(log.info, 'moving noise samples to GPU took {delta}s')
        self.buf.data[:] = torch.from_numpy(self._sampler.samples)
        done()

        self.buf.position = 0
        self.buf.reused = 0

    def _next_samples(self, size: torch.Size):
        amount = np.prod(size)
        lower = self.buf.position
        upper = lower + amount

        if upper > self.buf.size:
            self.buf.reused += 1

            if self._force_resample or self._do_resample:
                self._exchange_buffer()
                self._do_resample = False
                log.info('exchanged sample buffer, resuming operations')

            lower, upper = 0, amount
            if self.buf.reused > 0:
                log.warn(f're-using noise buffer {self.buf.reused} times now')

        self.buf.position = upper
        return self.buf.data[lower:upper].view(size)

    # ---

    def __init__(self, noise_buffer: int = None, **kwargs):
        super().__init__(**kwargs)

        self._buf = Gumbel.Buffer(
            size=noise_buffer or Gumbel.NOISE_BUFFER,
            device=self.device)

        self._do_resample = False
        self._force_resample = True  # FIXME configuration option

    def _forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """
        Use the gumbel distribution to force code vectors
        towards a one-hot representation

          x -- tensor of shape (n, M * K)
        """

        # --- gumbel ON
        noise = self._next_samples(x.shape)
        assert not noise.requires_grad
        intermed = (x + noise) / self._temperature

        # --- gumbel OFF
        # intermed = x

        y = torch.nn.functional.softmax(intermed, dim=-1)

        # checks (slow)
        # --------------------
        # def _between(v, val=1, eps=1e-5):
        #     return val - eps < v and v < val + eps
        # _flat = y.sum(-1).view(-1)
        # assert all([_between(v) for v in _flat]), str(_flat)
        # --------------------

        return y

    def _forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        """
        Creates a "hard" one hot encoding per codebook.

          x -- tensor of shape (n, M, K)
        """
        buf = torch.zeros(x.size()).to(device=self.device)
        y = buf.scatter(-1, x.argmax(-1).unsqueeze(-1), 1)

        # checks (slow)
        # --------------------
        # _uniq = y.to(device=DEV_CPU).unique()
        # assert len(_uniq) == 2, str(_uniq)
        # assert 0 in _uniq and 1 in _uniq, str(_uniq)
        # assert all([v.item() == 1 for v in y.sum(-1).view(-1)]), str(y)
        # --------------------

        if y.shape[0] > 1 and len((y[0].expand(y.shape) - y).nonzero()) == 0:
            codemap = y.nonzero()[:, 2].to(dtype=torch.uint8)
            codes = codemap.view(-1, 128)
            log.warn('[!] whole batch had equivalent activations!')
            log.warn(f'{[v.item() for v in codes[0][:20]]}...')
            log.warn(f'{[v.item() for v in codes[1][:20]]}...')
            raise ModelException('useless encoder')

        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add samples from gumbel distribution and apply softmax.

          x -- Tensor of shape (n, M, K)

        Returns:
          Tensor of shape (n, M, K) - where there are M categorical
          distributions with K components

        """
        dims = self.dimensions
        M, K = dims.components, dims.codelength

        # reshape tensor such that the last dimension holds
        # each code for applying the softmax
        codes = x.view(-1, M, K)
        assert codes.size()[1] == M
        assert codes.size()[2] == K

        if self.training:
            return self._forward_train(codes)
        else:
            return self._forward_eval(codes)

    def do_resample(self):
        self._do_resample = True


#
#   ENCODER
#


class Encoder(Model):

    name = 'encoder'

    @property
    def gumbel(self):
        return self._gumbel

    @property
    def grads(self) -> torch.Tensor:
        return self._grads

    def __init__(self, noise_buffer: int = None, **kwargs):
        super().__init__(**kwargs)
        dims = self.dimensions
        E, M, K = dims.embedding, dims.components, dims.codelength

        gumbel = Gumbel(noise_buffer=noise_buffer, **kwargs).to(self.device)
        self._gumbel = gumbel

        self._arch = collections.OrderedDict([
            ('fcl1', torch.nn.Linear(E, dims.fcl1)),
            ('fcl1-a', torch.nn.Tanh()),
            ('fcl2', torch.nn.Linear(dims.fcl1, M * K)),
            ('fcl2-a', torch.nn.Softplus()),
        ])

        # necessary for initialization with xavier [normal|uniform]
        self._gain = {
            'fcl1-a': torch.nn.init.calculate_gain('tanh'),
            'fcl2-a': torch.nn.init.calculate_gain('relu'),
        }

        self._layer = torch.nn.Sequential(self._arch)

        # --------------------

        if self.stog.grad_enc:
            log.info('registering backwards hook for encoder gradients')

            def _hook(grad):
                self._grads = grad.detach()
                return grad

            self._arch['fcl2'].weight.register_hook(_hook)

        # --------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self._layer(x)   # logits.shape = (n, M * K)
        y = self._gumbel(logits)  # y.shape = (n, M, K)
        return y

    # ---

    def init(self, method: str, *args):
        log.info(f'initialize model using method="{method}"')

        if method is None:
            log.warn('using standard encoder initialization!')
            method, args = 'uniform', (-1, 1)

        t_fcl1_w = self._arch['fcl1'].weight
        t_fcl2_w = self._arch['fcl2'].weight

        if method == 'uniform':
            lower, upper = args
            torch.nn.init.uniform_(t_fcl1_w, a=lower, b=upper)
            torch.nn.init.uniform_(t_fcl1_w, a=lower, b=upper)

        elif method == 'xavier uniform':
            torch.nn.init.xavier_uniform_(t_fcl1_w, self._gain['fcl1-a'])
            torch.nn.init.xavier_uniform_(t_fcl1_w, self._gain['fcl2-a'])

        elif method == 'xavier normal':
            torch.nn.init.xavier_normal_(t_fcl1_w, self._gain['fcl1-a'])
            torch.nn.init.xavier_normal_(t_fcl1_w, self._gain['fcl2-a'])

        elif method == 'embcompr':
            f_name = args[0]
            log.info(f'loading model state from "{f_name}"')
            state = torch.load(str(f_name))

            # remove leading '_encoder.' and select necessary vals
            selection = {key: state['_encoder.' + key] for key in (
                "_layer.fcl1.weight",
                "_layer.fcl1.bias",
                "_layer.fcl2.weight",
                "_layer.fcl2.bias",
            )}

            self.load_state_dict(selection)

        else:
            assert False, f'unknown initialization method "{method}"'

        # ---

        _fmt = f'μ="{t_fcl1_w.mean():.3f}", σ="{t_fcl1_w.std():.3f}"'
        log.info(f'initialized fc-layer 1 {_fmt}')
        _fmt = f'μ="{t_fcl2_w.mean():.3f}", σ="{t_fcl2_w.std():.3f}"'
        log.info(f'initialized fc-layer 2 {_fmt}')

        assert t_fcl1_w.requires_grad
        assert t_fcl2_w.requires_grad


class BaselineEncoder(Model):

    name = 'encoder_baseline'

    @property
    def sample(self):
        return self._sampler().squeeze()

    def __init__(self, sample_size, **kwargs):
        super().__init__(**kwargs)

        U = torch.distributions.uniform.Uniform
        sampler = U(torch.tensor([0.0]), torch.tensor([1.0]))
        self._sampler = functools.partial(
            sampler.sample, sample_shape=sample_size)

        # mocking
        self.gumbel = Gumbel(**kwargs).to(self.device)

    def _forward_train(self) -> torch.Tensor:
        logits = self.sample.to(device=self.device)
        return torch.nn.functional.softmax(logits, dim=-1)

    def _forward_valid(self) -> torch.Tensor:
        x = self._forward_train()
        y = torch.zeros(x.size()).to(device=self.device)
        return y.scatter(-1, x.argmax(-1).unsqueeze(-1), 1)

    def forward(self, _) -> torch.Tensor:
        if self.training:
            return self._forward_train()
        else:
            return self._forward_valid()

#
#   DECODER
#


class QuantizationDecoder(Model):

    MAX_SAMPLES = int(1e6)  # for initialization / must fit into RAM
    name = 'decoder_quantization'

    @property
    def grads(self) -> torch.Tensor:
        return self._grads

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        dims = self.dimensions
        M, K = dims.components, dims.codelength

        components = torch.empty(M * K, dims.embedding)
        self.components = torch.nn.Parameter(components)

        # --------------------
        if self.stog.grad_dec:
            log.info('registering backwards hook for decoder gradients')

            def _hook(grad):
                self._grads = grad.detach()
                return grad

            self.components.register_hook(_hook)

        # --------------------

    def forward(self, x: torch.Tensor):
        """
        Use the codes provided in x to re-create embedding vectors
        based on the components.

          x -- Tensor of shape (n, M, K)

        Returns:
          Tensor of shape (n, E)

        """
        dims = self.dimensions
        M, K = dims.components, dims.codelength
        y = x.view(-1, M * K).matmul(self.components)
        return y

    def init(self, embed: uce.Embed, factor: float):
        dims = self.dimensions
        M, K = dims.components, dims.codelength

        # low, high = 0, 1024
        upper = min(len(embed), QuantizationDecoder.MAX_SAMPLES)
        log.info(f'initialize codebooks by selecting from {upper} samples')

        x = factor or 1
        log.info(f'scaling samples by {x:.3f}')

        idx = np.random.randint(0, upper, M * K)
        self.components.data[:] = torch.Tensor(embed[:upper][idx] * x)


class HammingDecoder(Model):

    name = 'decoder_hamming'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cossim = torch.nn.CosineSimilarity()

    def _forward_train(self, *args):
        y = self._cossim(*args).sum(dim=-1)
        y /= self.dimensions.components

        # assert all(0 <= y), y
        # assert all(y <= 1), y

        return y

    def _forward_eval(self, *args):
        x1, x2 = (x.to(dtype=torch.uint8) for x in args)
        y = (x1 & x2).sum(dim=-1).sum(dim=-1).to(dtype=torch.float)
        y /= self.dimensions.components

        # assert all(0 <= y), y
        # assert all(y <= 1), y

        return y

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Takes two encoded batches of embeddings and calculates their
        distance.  The codomain of x1, x2 is either in [0, 1] for
        training or in {0, 1} otherwise.

          x1 -- Tensor of shape (n, M, K)
          x2 -- Tensor of shape (n, M, K)

        Returns
          Tensor of shape (n, ) in [0, 1] with distance of each (x1_i, x2_i)

        """
        assert x1.shape == x2.shape

        if self.training:
            return self._forward_train(x1, x2)
        else:
            return self._forward_eval(x1, x2)
