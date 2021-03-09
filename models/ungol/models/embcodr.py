#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   NEURAL CODE PROCESSOR
#
#   Work with codes produced by embcompr.
#
#   The layout of the folder to read from must contain the following files:
#   E.g. reading model embcompr01000.torch from /path/to/experiment01/
#
#   ./embcodr.py /path/to/exp/compressor/model-1000.torch <EXP> <CMD>
#
#   Searches for:
#     /path/to/exp/compressor/model-1000.torch
#     /path/to/exp/compressor/dimensions.json
#     /path/to/exp/compressor/stog.json
#     /path/to/exp/embcompr.conf
#
#   and uses <EXP> section of the found embcompr.conf
#

from ungol.common import logger
from ungol.common import embed as uce
from ungol.models import models as umm
from ungol.models import embcompr as ume
from ungol.models import training as umt

import attr
import h5py
import torch
import numpy as np
from tqdm import tqdm as _tqdm

import math
import pathlib
import argparse
import functools

from typing import Any
from typing import Dict
from typing import Tuple


log = logger.get('models.embcodr')
tqdm = functools.partial(_tqdm, ncols=80, disable=False)


# --- external interface


def create_codes(
        compr: ume.Compressor,
        batch: torch.Tensor,
        components: int) -> np.array:

    onehot = compr.encoder(batch)

    codemap = onehot.nonzero()[:, 2].to(dtype=torch.uint8)
    codes = codemap.view(-1, components).numpy()

    assert codes.shape[1] == components
    return codes


# create range of potencies to calculate byte values
# ([128, 64, 32, ...])
# (8, ) bits
_potency_byte = np.flip(2 ** np.arange(8, dtype=np.uint))


def create_hashes(data: np.array):

    c_words, c_bits = data.shape
    # assert c_bits % 8 == 0, c_bits % 8

    # introduce padding
    if c_bits % 8 != 0:
        c_bits = c_bits + (8 - (c_bits % 8))
        _data = np.zeros((c_words, c_bits))
        _data[:, :data.shape[1]] = data
        data = _data

    # WORDS x BYTES x BITS (n, 256) -> (n, 32, 8)
    c_bytes = c_bits // 8
    raw = data.reshape(c_words, c_bytes, -1)

    # repeat the potency byte until they match the whole chunk
    # (n * c_bits, )
    _potency_flat = np.tile(_potency_byte, c_bytes * c_words)

    # (n, c_bits, 8)
    potency = _potency_flat.reshape(*raw.shape)

    # convert raw array to int representation
    # of bytes (n, c_bytes)
    byte_arr = (raw * potency[:c_words]).sum(axis=2)

    return byte_arr.astype(np.uint8)


# The dict contains the following meta information:
#   - 'bits': int -- number of bits per code
#   - 'words': int -- size of the vocabulary
#   - 'knn': List[int] -- the distances saved for the k-th nn
#
# The dists array is of size (n, len(knn))
# The codes array is of size (n, bits)
Codes = Tuple[Dict[str, Any], 'dists (np.array)', 'codes (np.array)']


# provide a *.bin file produced with embcodr --binary
def load_codes_bin(fname: str) -> Codes:
    with open(fname, 'rb') as fd:
        meta_buf = fd.read(Writer.BIN_META)
        data_buf = fd.read()

    meta = eval(meta_buf.decode('ascii').partition('\0')[0])

    assert 'bits' in meta
    assert 'words' in meta

    # counts
    c_bytes = meta['bits'] // 8
    c_words = meta['words']

    raw = np.frombuffer(data_buf, dtype=np.uint8)
    codes = raw.reshape(c_words, c_bytes)

    assert c_words == codes.shape[0]

    # return codes
    return meta, codes


# ---


@attr.s
class Writer:

    # Byte size of the meta block of binary text files
    BIN_META = 1024

    # ---

    # output files
    path: pathlib.Path = attr.ib()
    fname:         str = attr.ib()

    dims:   umm.Dimensions = attr.ib()
    vocab:  Dict[str, int] = attr.ib()
    lookup: Dict[int, str] = attr.ib()

    binary: bool = attr.ib()
    notext: bool = attr.ib()

    # ---

    def __attrs_post_init__(self):
        if self.binary:
            assert self.dims.codelength == 2, 'not a binary code!'
            print('enabled binary writing codes')

    def __str__(self):
        return 'Writer (binary={}) for {} codes'.format(
            self.binary, len(self.vocab))

    def __enter__(self):
        p_base = self.path / self.fname
        s_base = str(p_base)

        print('opening files "(...)/{}/{}.*"'.format(*p_base.parts[-2:]))
        self._fd_h5 = h5py.File(s_base + '.h5', 'w')

        if not self.notext:
            self._fd_txt = open(s_base + '.txt', 'w')

        if self.binary:
            self._fd_bin_txt = open(s_base + '.bin', 'wb')

        return self

    def __exit__(self, *args):
        self._fd_h5.close()

        if not self.notext:
            self._fd_txt.close()

        if self.binary:
            self._fd_bin_txt.close()

        print('closed open files')

    # ---

    def _write_chunk_txt(self, pos, data):
        chunk, lower, upper = pos
        s_buf = []

        try:
            fmt_word, fmt_no = self._txt_fmt

        except AttributeError:
            word_maxlen = max([len(word) for word in self.vocab])

            fmt_word = '{:%ds}' % word_maxlen
            fmt_no = '{:%dd}' % len(str(self.dims.codelength))

            self._txt_fmt = fmt_word, fmt_no

        words = [self.lookup[i] for i in range(lower, upper)]

        for word, arr in zip(words, data):
            l_buf = [fmt_word.format(word)]
            l_buf += [fmt_no.format(n) for n in arr]
            s_buf.append(' '.join(l_buf))

        self._fd_txt.write('\n'.join(s_buf))

    def _write_chunk_h5(self, pos, data):
        ds_key = 'codes'

        if ds_key not in self._fd_h5:
            shape = len(self.vocab), self.dims.components
            self._h5_ds = self._fd_h5.create_dataset(
                ds_key, shape=shape, dtype='u8')

        chunk, lower, upper = pos
        self._h5_ds[lower:upper] = data

    # ---

    def _write_bin_txt_meta(self, meta: Dict['str', Any]):
        buf_meta = bytearray(Writer.BIN_META)
        buf_dict = bytes(repr(meta), encoding='ascii')

        assert len(buf_dict) < len(buf_meta)

        buf_meta[:len(buf_dict)] = buf_dict
        self._fd_bin_txt.write(buf_meta)

    def _write_chunk_bin_txt(self, pos, data):
        chunk, lower, upper = pos
        c_bits = data.shape[1]
        byte_arr = create_hashes(data)

        if chunk == 0:
            # prepare meta information
            meta = dict(bits=c_bits, words=len(self.vocab), )
            self._write_bin_txt_meta(meta)

        for code in byte_arr:
            self._fd_bin_txt.write(bytes(a for a in code))

    # ---

    def write_chunk(
            self,
            pos: Tuple[int, int, int],
            data: np.array):

        args = pos, data
        self._write_chunk_h5(*args)

        if not self.notext:
            self._write_chunk_txt(*args)

        if self.binary:
            self._write_chunk_bin_txt(*args)


def create(
        model: ume.Compressor,
        conf: umt.Config,
        path: pathlib.Path,
        prefix: str,
        suffix: str,
        binary: bool = False,
        embed: uce.Embed = None,
        notext: bool = False):
    """

    This function takes a model, iterates over the whole vocabulary
    and produces a code per word. The produced codes are then saved
    to disk.

    """
    print('creating codes from vocabulary')

    if embed is None:
        print('creating embedding provider from configuration')
        embed = uce.create(uce.Config.from_conf(conf.embed))

    print('will write {} codes'.format(len(embed)))

    writer_args = dict(
        path=path,
        fname=f'{prefix}.{suffix}',
        dims=conf.dimensions,
        vocab=embed.vocab,
        lookup={v: k for k, v in embed.vocab.items()},
        binary=binary,
        notext=notext, )

    with Writer(**writer_args) as writer:
        model.encoder.eval()

        print('')
        total = math.ceil(len(embed) // uce.Embed.CHUNK_SIZE)
        for chunk, data in tqdm(enumerate(embed.chunks()), total=total):

            batch = torch.from_numpy(data).to(dtype=torch.float)
            codes = create_codes(model, batch, conf.dimensions.components)

            lower = chunk * data.shape[0]
            upper = lower + data.shape[0]

            pos = chunk, lower, upper
            writer.write_chunk(pos, codes)

    print('\n', 'done')


# ---


def _load_conf(folder: pathlib.Path):
    f_conf = 'embcompr.conf'
    conf_file = folder.parent / f_conf
    if not conf_file.is_file():
        raise Exception(f'cannot find "{f_conf}"')

    exp, conf = next(umt.Config.create(
        str(conf_file),
        selection=[args.selection]))

    assert exp == args.selection
    print(f'got configuration for ({args.selection})')

    return exp, conf


def main(args):
    print('\n', 'welcome to embcodr'.upper(), '\n')

    log.info('-' * 80)
    log.info('embcodr invoked')

    path = pathlib.Path(args.model)
    folder = path.parent

    # load model
    model = ume.Compressor.load(folder, path.name)
    print('successfully loaded model')

    # (optionally) load input embedding
    embed_conf = uce.Config.from_args(args, required=False)
    embed = uce.create(embed_conf) if embed_conf else None

    # load configuration
    exp, conf = _load_conf(folder)

    create(model, conf,
           folder.parent if not args.out else pathlib.Path(args.out),
           prefix=args.out_prefix if args.out_prefix else 'codemap',
           suffix=path.resolve().stem,
           binary=args.binary,
           embed=embed,
           notext=args.no_text, )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='torch file containing saved embcompr model', )

    parser.add_argument(
        'selection', type=str,
        help='experiment name to be selected from configuration')

    parser.add_argument(
        '--out', type=str,
        help='(optional) ouput folder')

    parser.add_argument(
        '--out-prefix', type=str,
        help='prefix for the output file names')

    parser.add_argument(
        '--binary', action='store_true',
        help='for codes with K=2, write special files')

    parser.add_argument(
        '--no-text', action='store_true',
        help='do not write text files (they might be too big)')

    uce.add_parser_args(parser)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
