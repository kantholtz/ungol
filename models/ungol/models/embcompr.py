#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   NEURAL COMPRESSOR
#
#   by Shu, R., & Nakayama, H. (2017). Compressing Word Embeddings via
#   Deep Compositional Code Learning. arXiv preprint arXiv:1711.01068.
#
#   Reference Code: https://github.com/zomux/neuralcompressor
#


from ungol.common import logger
from ungol.common import embed as uce
from ungol.models import models as umm
from ungol.models import training as umt

import re
import argparse
import functools

import torch
from tqdm import tqdm as _tqdm


log = logger.get('models.embcompr')
tqdm = functools.partial(_tqdm, ncols=80, disable=False)


# ---

DEV_CPU = torch.device('cpu')
DEV_GPU = torch.device('cuda')

# ---


def _print_examples(compressor, training):
    vocab = list(training.ember.vocab.keys())[1000:1010]
    dist = torch.nn.PairwiseDistance()

    for word in vocab:
        x = training.ember[word]
        x = torch.from_numpy(x).requires_grad_(False)
        x = x.to(device=training.device, dtype=torch.float)
        y = compressor(x)

        print('{}:\t{:5f}'.format(word, dist(x, y).item()))
        print('x', x.norm().item())
        print('y', y.norm().item())
        print('-' * 20)


def _print_cgraph(model, fn):
    """
    Invoke with _print_cgraph(t.grad_fn)
    where t is a torch.Tensor
    """
    print(str(model) + '\n')
    fmt = '{}{}: {}'

    def _r(curr_fn, level=1):
        if not len(curr_fn.next_functions):
            return

        for i, next_fn in enumerate(curr_fn.next_functions):
            print(fmt.format('| ' * level, i, next_fn[0]))
            if next_fn[0] is not None:
                _r(next_fn[0], level=level + 1)

    print(fmt.format('', 0, fn))
    _r(fn)


def _print_grad_norm(self, grad_input, grad_output):
    print('inside ' + self.__class__.__name__ + ' backward')
    print('inside class:' + self.__class__.__name__)
    print('')
    print('  grad_input: ', type(grad_input))
    print('  grad_input[0]: ', type(grad_input[0]))
    print('  grad_output: ', type(grad_output))
    print('  grad_output[0]: ', type(grad_output[0]))
    print('')
    print('  grad_input size:', grad_input[0].size())
    print('  grad_output size:', grad_output[0].size())
    print('  grad_input norm:', grad_input[0].norm())
    print('-' * 80)


#
#  TORCH MODELS
#


class Compressor(umm.Model):
    """

    Neural Embedding Compressor Autoencoder

    Commonly used identifier:
      n -- Batch size
      M -- Number of components
      K -- Codomain of codebook components
      E -- Embedding dimensionality

    """

    # ---

    name = 'compressor'

    @property
    def version(self) -> float:
        return 0.2

    def __str__(self) -> str:
        s = 'Neural Compressor Version {}'.format(self.version)
        return s + '\n' + super().__str__()

    @property
    def encoder(self) -> torch.nn.Module:
        return self._encoder

    @property
    def decoder(self) -> torch.nn.Module:
        return self._decoder

    @property
    def encoder_codes(self) -> torch.Tensor:
        assert self.stog.codes
        return self._encoder_codes

    # see class um.Model for kwargs
    def __init__(self, noise_buffer: int = None, **kwargs):
        super().__init__(**kwargs)
        self._encoder = umm.Encoder(noise_buffer=noise_buffer, **kwargs)
        self._decoder = umm.QuantizationDecoder(**kwargs)

    # Forward pass accepts one tensor containing a batch of
    # embedding vectors which have to be reconstructed by the model
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self._encoder(x)

        if self.stog.codes:
            self._encoder_codes = encoded.detach()

        decoded = self._decoder(encoded)
        return decoded

    @staticmethod
    def load(*args, **kwargs):
        return umm.Model._load(Compressor, *args, **kwargs)


# ---


def _init_model(conf: umt.Config) -> Compressor:

    compressor = Compressor(
        noise_buffer=conf.noise_buffer,
        dimensions=conf.dimensions,
        stog=conf.stog,
        device=conf.device)

    log.info('initialized compressor')
    for param in compressor.parameters():
        log.info('  parameter [%s] with size %s',
                 str(param.requires_grad),
                 str(param.size()))

    log.info('transferring model to device: %s', str(conf.device))
    compressor = compressor.to(device=conf.device)

    params = conf.parameters
    gumbel = compressor.encoder.gumbel

    gumbel.set_sampler(scale=params.gumbel_scale[0])
    gumbel.set_temperature(params.gumbel_temperature[0])

    return compressor


def _init_optimizer(
        model: torch.nn.Module,
        params: umt.Parameters) -> torch.optim.Optimizer:

    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=params.learningrate[0],
    #     momentum=params.momentum)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params.learningrate[0])

    # optimizer = torch.optim.Adagrad(
    #     model.parameters(),
    #     lr=params.learningrate[0])

    log.info('initialized optimizer: {}'.format(
        re.sub(' +', ' ', str(optimizer).replace('\n', ''))))

    return optimizer


def _init_criterion():
    return umm.DistanceLoss()


def _init_embcompr(conf: umt.Config, embed: uce.Embed):
    compressor = _init_model(conf)
    compressor.encoder.init(*conf.parameters.encoder_init)
    compressor.decoder.init(embed, conf.parameters.decoder_init_factor)

    optimizer = _init_optimizer(compressor, conf.parameters)
    criterion = _init_criterion()

    return compressor, optimizer, criterion


def _init_baseline(conf: umt.Config, ember: uce.Embed):
    log.info('initializing the baseline encoder')

    compressor, optimizer, criterion = _init_embcompr(conf, ember)
    compressor._encoder = umm.BaselineEncoder(
        torch.Size([
            conf.batch_size,
            conf.dimensions.components,
            conf.dimensions.codelength]),
        dimensions=conf.dimensions,
        stog=conf.stog,
        device=conf.device)

    compressor._encoder.to(device=conf.device)

    return compressor, optimizer, criterion


# ---


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--no-gpu', action='store_true', default=False,
        help='disable training on GPU')

    parser.add_argument(
        '--baseline', action='store_true',
        help='use the baseline encoder')

    # ---

    parser.add_argument(
        'conf', type=str,
        help='configuration file')

    return parser.parse_args()


def main():
    args = parse_args()

    if not tqdm.keywords['disable']:
        print('\nwelcome to embcompr'.upper())

    init = _init_baseline if args.baseline else _init_embcompr

    device = umm.detect_device(args.no_gpu)
    for name, conf in umt.Config.create(args.conf, device=device, ):
        print(f'\nrunning {name}\n')

        try:
            umt.train(conf, init)

        except umm.ModelException as exc:
            log.error(f'[!] catched model exception: {str(exc)}')
            print(f'\n\n[!] error: "{str(exc)}"')
            with open(f'{conf.stog.folder}/error.txt', mode='w') as fd:
                fd.write(f'\nException thrown: "{str(exc)}"\n')

        except umt.EarlyStopping:
            log.info(f'[!] reached early stopping criterion, aborting loop')


if __name__ == '__main__':
    log.info('-' * 80)
    log.info('embcompr invoked directly')

    try:
        main()
    except KeyboardInterrupt:
        log.info('interrupted by keyboard interrupt')

    log.info('exiting gracefully')
