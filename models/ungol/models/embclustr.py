#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   NEURAL CODE CLUSTERING
#
#     NOTE: this code has never been run in production
#     and is merely some sort of note...
#
#     Will remove this if it is actually never needed.
#


from ungol.models import models as um
from ungol.models import training as ut
from ungol.models.common import logger

import torch
from tqdm import tqdm as _tqdm

import re
import argparse
import functools


# ---


log = logger('embclustr')
tqdm = functools.partial(_tqdm, ncols=80, disable=False)

# ---


class CodeCluster(um.Model):

    name = 'codecluster'

    @property
    def version(self) -> float:
        return 0.1

    def __str__(self) -> str:
        s = 'Neural Compressor Version {}'.format(self.version)
        return s + '\n' + super().__str__()

    @property
    def encoder(self) -> torch.nn.Module:
        return self._encoder

    @property
    def decoder(self) -> torch.nn.Module:
        return self._decoder

    # see class um.Model for kwargs
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._encoder = um.Encoder(**kwargs)
        self._decoder = um.HammingDecoder(**kwargs)

    @staticmethod
    def load(*args, **kwargs):
        return um.Model._load(CodeCluster, *args, **kwargs)

    def _dist(self, x1, x2):
        return (x1 - x2).abs().norm(dim=-1, p=2)

    @property
    def encoder_codes(self) -> torch.Tensor:
        assert self.stog.codes
        return self._encoder_codes

    # Forward pass accepts a tensor batch randomly sampled from
    # the embedding space and then split in half. The distance between
    # the vectors in linear space shall correlate to their codes
    # distance in hamming space.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        assert n % 2 == 0

        x1, x2 = x, x[torch.randperm(n)]
        self._expected = self._dist(x1, x2)

        y1, y2 = self._encoder(x1), self._encoder(x2)
        y = self._decoder(y1, y2)

        if self.stog.codes:
            # FIXME both codes
            # stacked = torch.stack((y1, y2))
            # self._encoder_codes = stacked.view(n * 2, -1).detach()
            self._encoder_codes = y1.detach()

        return y


# ---


def _init_model(conf: ut.Config) -> CodeCluster:
    clustr = CodeCluster(dimensions=conf.dimensions,
                         stog=conf.stog,
                         device=conf.device)

    log.info('initialized compressor')
    for param in clustr.parameters():
        log.info('  parameter [%s] with size %s',
                 str(param.requires_grad),
                 str(param.size()))

    log.info('transferring model to device: %s', str(conf.device))
    clustr = clustr.to(device=conf.device)

    params = conf.parameters
    gumbel = clustr.encoder.gumbel

    gumbel.set_sampler(scale=params.gumbel_scale[0])
    gumbel.set_temperature(params.gumbel_temperature[0])

    return clustr


def _init_optimizer(
        model: torch.nn.Module,
        params: ut.Parameters) -> torch.optim.Optimizer:

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
    return torch.nn.MSELoss()


def _init_embclustr(conf: ut.Config, _):
    clustr = _init_model(conf)
    optimizer = _init_optimizer(clustr, conf.parameters)
    criterion = _init_criterion()

    return clustr, optimizer, criterion


# ---


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--no-gpu', action='store_true', default=False,
        help='disable training on GPU')

    # overwrites provided configuration file options

    parser.add_argument(
        '--epochs', nargs=1, type=int,
        help='how many times training iterates')

    parser.add_argument(
        '--stats-dir', nargs=1, type=str,
        help='')

    # ---

    parser.add_argument(
        'conf', type=str,
        help='configuration file')

    return parser.parse_args()


def main():
    args = parse_args()

    if not tqdm.keywords['disable']:
        print('\nwelcome to embclustr\n'.upper())

    for name, conf in ut.Config.create(
            args.conf,
            device=um.detect_device(args.no_gpu), ):

        ut.train(conf, _init_embclustr)


if __name__ == '__main__':
    log.info('-' * 80)
    log.info('embcompr invoked directly')

    try:
        main()
    except KeyboardInterrupt:
        log.info('interrupted by keyboard interrupt')

    log.info('exiting gracefully')
