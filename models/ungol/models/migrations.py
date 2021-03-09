#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

Wild collection of migration operations to transform legacy data to
new file formats. Imports should be placed inside the respective
functions such that they are self-contained. Also code used from ungol
should be copied such that no changes on master break the migrations.

Naming conventions:

[do_]migrate_<YYMMDD>_<NAME>(...)

All do_* routines do not accept parameters.
Usual execution looks like this:

cd to/destination
ipython
> from ungol.models import migrations as um
> um.do_migrate_180828_codes()

Overview:

* do_migrate_180828_codes
  delete 'raw' entries from codes.h5

* do_migrate_180826_models
  simply renaming 'codebooks'->'components' inside model*.torch files

* do_migrate_180803_codes
  migrates legacy raw activation files to codes.h5 hdf5 files

* do_migrate_180803_gradients
  migrates legacy raw/grad_(end|dec)-*.npz files to hdf5

* do_migrate_180803_losses
  migrates legacy raw/losses-(valid|train)-*.npz files to hdf5

* do_migrate_180801_hamming_distances
  migrates legacy npz files produced by analyze.py nn-* to hdf5

"""


def migrate_180828_codes(f_name: str) -> int:

    import os
    import h5py
    import subprocess

    count = 0

    print('migrating', f_name)
    with h5py.File(f_name, 'r+') as fd:
        print('migrating {}'.format(fd.filename))

        for kind in 'train', 'valid':
            print('-' * 30, '\n', kind)

            if kind not in fd:
                print('skipping {}: not in file'.format(kind))
                continue

            for epoch in fd[kind]:
                ds = fd[kind][epoch]

                if 'raw' not in ds:
                    print('  {} - nothing to do here'.format(epoch))
                    continue

                print('  {} - deleting "raw" data'.format(epoch))
                del ds['raw']
                count += 1

    if count > 0:
        print('hdf5 file was modified, repacking to reclaim space')
        f_tmp = f_name + '.tmp'

        command = ["ptrepack", "-o", "--chunkshape=auto", "--propindexes"]
        files = [f_name, f_tmp]

        # stderr to /dev/null because of verbose warnings...
        rc = subprocess.call(command + files, stderr=subprocess.DEVNULL)
        if rc != 0:
            print('an error occured!')
            return

        print('overwriting old file')
        os.rename(f_tmp, f_name)

    print('done')
    return count


def do_migrate_180828_codes():
    import pathlib

    count = 0

    for glob in pathlib.Path('.').glob('**/codes.h5'):
        glob_count = migrate_180828_codes(str(glob))

        print('deleted', glob_count)
        count += glob_count

    print('deleted a total of ', count)

#
# ------------------------------------------------------------
#


def migrate_180826_models(f_folder: str, f_glob: str) -> int:

    """
    inside <EXPERIMENT>/compressor/
    provide f_glob -> '*.torch'

    Converts the old *.torch files containing
    'codebooks' to 'components'
    """

    import torch
    import pathlib

    p_glob = pathlib.Path(f_folder).glob(f_glob)

    old_key = '_decoder.codebooks'
    new_key = '_decoder.components'

    count = 0
    for fname in [str(p) for p in p_glob]:
        state_dict = torch.load(fname)

        if old_key in state_dict:
            state_dict[new_key] = state_dict[old_key]
            del state_dict[old_key]
            torch.save(state_dict, fname)
            count += 1

        assert new_key in state_dict

    return count


def do_migrate_180826_models():
    # inside experiments/<EXPERIMENT>

    import pathlib

    folders = pathlib.Path('.').glob('*/compressor/')
    for folder in folders:
        if not folder.is_dir():
            continue

        print(folder.parts[0])
        converted = migrate_180826_models(str(folder), '*.torch')
        print('  converted: {}'.format(converted))


#
# ------------------------------------------------------------
#


def migrate_180803_codes(
        f_glob: str,
        out_file: str, ):
    """
    Migrates the old raw activation files
    to the shiny new codes.h5 containing
    both histograms and entropy values of
    the underlying data.

    """

    import h5py
    import numpy as np
    from tqdm import tqdm

    import math
    import pathlib
    from typing import List

    # ---

    def _partition(fn, ls):
        buckets = [], []
        for x in ls:
            idx = 0 if fn(x) else 1
            buckets[idx].append(x)
        return buckets

    def _discriminate(glob):
        fname = glob.parts[-1]
        if fname.startswith('codes-train'):
            return True
        elif fname.startswith('codes-valid'):
            return False
        else:
            assert False, fname

    def _grp_from_path(fd: h5py.File, path: pathlib.Path):
        parts = path.parts[-1].split('.')[-2].split('-')

        epoch = int(parts[-1])
        kind = parts[-2]

        assert kind == 'train' or kind == 'valid', kind

        grp = fd.create_group('{}/{}'.format(kind, epoch))
        return grp

    def _arr_from_path(path: pathlib.Path):
        nf = np.load(str(path))
        arr = nf['data']
        nf.close()

        assert len(arr.shape) == 3
        return arr

    def _entropy(arr):
        entropies = []

        for i in range(arr.shape[1]):
            a = arr[:, i, :].sum(axis=0)
            entropy = sum([
                0 if x == 0 else -x * math.log(x, 2)
                for x in a / a.sum()])

            entropies.append(entropy / math.log(arr.shape[2], 2))

        return np.array(entropies)

    # ---

    def _migrate_train(fd: h5py.File, paths: List[pathlib.Path]):
        bins = int(1e2)

        for path in tqdm(paths, total=len(paths), ncols=80):
            grp = _grp_from_path(fd, path)
            arr = _arr_from_path(path)

            N, M, K = arr.shape
            grp.create_dataset('raw', data=arr)

            # histogram
            hist, bin_edges = np.histogram(arr.flat, bins=bins, range=(0, 1))
            ds_hist = grp.create_dataset('histogram', data=hist)
            ds_hist.attrs['bin_edges'] = bin_edges

            # entropy
            entropy = _entropy(arr)
            ds_entropy = grp.create_dataset('entropy', data=entropy)
            ds_entropy.attrs['mean'] = entropy.mean()

    # ---

    def _migrate_valid(fd: h5py.File, paths: List[pathlib.Path]):
        for path in tqdm(paths, total=len(paths), ncols=80):
            grp = _grp_from_path(fd, path)
            arr = _arr_from_path(path)

            N, M, K = arr.shape
            codes = arr.nonzero()[2].reshape(-1, M).astype(np.uint)

            assert codes.shape[0] == arr.shape[0]
            assert codes.shape[1] == arr.shape[1]
            assert codes.max() < arr.shape[2], codes.max()

            grp.create_dataset('raw', data=codes)

            # histogram per codebook
            ds_counts = grp.create_dataset('counts', shape=(M, K))
            for i in range(M):
                sel = codes[:, i]
                a = np.histogram(sel, bins=K, range=(0, K-1))[0]
                ds_counts[i] = a

            assert ds_counts.shape[0] == M
            assert ds_counts.shape[1] == K

    # ---

    globs = sorted(pathlib.Path('.').glob(f_glob))
    train, valid = _partition(_discriminate, globs)

    assert len(globs) > 0

    fd = h5py.File(out_file, 'w')

    try:

        print('\n', 'migrate training data', '\n')
        _migrate_train(fd, train)

        print('\n', 'migrate validation data', '\n')
        _migrate_valid(fd, valid)

        print()

    finally:
        fd.close()


def do_migrate_180803_codes():
    # inside experiments/<EXPERIMENT>

    import pathlib

    folders = pathlib.Path('./').glob('*')
    for folder in folders:
        if not folder.is_dir():
            continue

        f_glob = str(folder/'raw'/'codes-*.npz')
        out_file = str(folder/'codes.h5')

        print(folder.parts[0])
        migrate_180803_codes(f_glob, out_file)


#
# ------------------------------------------------------------
#


def migrate_180803_gradients(
        f_glob: str,
        out_file: str, ):
    """
    Migrate old raw/grad_(end|dec)-*.npz files.
    """

    import h5py
    import numpy as np

    import pathlib
    from tqdm import tqdm

    gen = sorted(pathlib.Path('.').glob(f_glob))
    total = len(gen)  # enc + dec

    assert total > 0

    fd = None
    idx = {'decoder': 0, 'encoder': 0}

    for glob in tqdm(gen, total=total, ncols=80):
        nf = np.load(glob)
        arr = nf['data']

        if fd is None:
            shape = (total // len(idx), arr.shape[0])
            fd = h5py.File(out_file, 'w')
            for key in idx:
                fd.create_dataset(key, shape)

        kind_raw = glob.parts[-1].split('-')[0]
        if kind_raw == 'grad_enc':
            kind = 'encoder'
        elif kind_raw == 'grad_dec':
            kind = 'decoder'

        fd[kind][idx[kind]] = arr
        idx[kind] += 1
        nf.close()

    fd.close()


def do_migrate_180803_gradients():
    # inside experiments/<EXPERIMENT>

    import pathlib

    folders = pathlib.Path('./').glob('*')
    for folder in folders:
        if not folder.is_dir():
            continue

        f_glob = str(folder/'raw'/'grad_*.npz')
        out_file = str(folder/'gradients.h5')

        print(folder.parts[0])
        migrate_180803_gradients(f_glob, out_file)


#
# ------------------------------------------------------------
#


def migrate_180803_losses(
        f_glob: str,
        out_file: str, ):
    """
    Migrate old raw/losses-(valid|train)-*.npz files.
    """
    import h5py
    import numpy as np

    import pathlib
    from tqdm import tqdm

    gen = sorted(pathlib.Path('.').glob(f_glob))
    total = len(gen)  # train + valid

    assert total > 0
    print('processing {} files'.format(len(gen)))

    fd = None
    idx = {'train': 0, 'valid': 0}

    for glob in tqdm(gen, total=total * 2, ncols=80):
        nf = np.load(glob)
        arr = nf['data']

        if fd is None:
            shape = (total // 2, arr.shape[0])
            fd = h5py.File(out_file, 'w')
            for key in idx:
                fd.create_dataset(key, shape)

        kind = glob.parts[-1].split('-')[-2]
        assert kind == 'train' or kind == 'valid', kind

        fd[kind][idx[kind]] = arr
        idx[kind] += 1
        nf.close()

    fd.close()


def do_migrate_180803_losses():
    # inside experiments/<EXPERIMENT>

    import pathlib

    folders = pathlib.Path('./').glob('*')
    for folder in folders:
        if not folder.is_dir():
            continue

        f_glob = str(folder/'raw'/'losses-*.npz')
        out_file = str(folder/'losses.h5')

        print(folder.parts[0])
        migrate_180803_losses(f_glob, out_file)


#
# ------------------------------------------------------------
#


def migrate_180801_hamming_distances(
        f_glob: str,
        out_file: str,
        components: int, ):
    """
    This function migrates files produced by analyze.py nn-* pre-hdf5.
    The old way was storing 100 chunks of npz files. The new format
    contains the mapping, the distances and a histogram over the
    distance distribution in one .h5 file.

    For distances in linear space, re-produce files using analyze.py.
    This function is for hamming codes only. Meta-Information
    (max, min, histogram) cannot be reproduced here.

      -- f_glob: str e.g. 'foo/bar/hamming.??.npz'
      -- out_file: file to write data to

    """

    import h5py
    import numpy as np
    from tqdm import tqdm

    import pathlib

    print('opening', f_glob)

    gen = sorted(pathlib.Path('.').glob(f_glob))

    print('processing {} chunks'.format(len(gen)))

    # to be initialized upon first chunk
    fd = None
    ds_dists = None
    ds_mapping = None
    chunk_shape = None

    minimum = None
    maximum = None

    histogram = None
    bin_edges = None

    print()
    for i, glob in tqdm(enumerate(gen), total=len(gen), ncols=80):
        np_file = np.load(str(glob))

        raw_dists = np_file['dists']
        raw_mapping = np_file['mapping']

        # initialization

        if fd is None:
            chunk_count = len(gen)
            chunk_shape = raw_dists.shape

            shape = (chunk_count * chunk_shape[0], chunk_shape[1])
            histogram = np.zeros(components + 1, dtype=np.int)

            # create .h5 file

            fd = h5py.File(out_file, 'w')
            ds_dists = fd.create_dataset('dists', shape)
            ds_mapping = fd.create_dataset('mapping', shape)

        # ---

        a = i * chunk_shape[0]
        b = a + chunk_shape[0]

        ds_dists[a:b] = raw_dists
        ds_mapping[a:b] = raw_mapping

        _maximum = raw_dists.max()
        if maximum is None or maximum < _maximum:
            maximum = _maximum

        _minimum = raw_dists.min()
        if minimum is None or _minimum < minimum:
            minimum = _minimum

        _hist, bin_edges = np.histogram(
            raw_dists, bins=components + 1, range=(0, components))

        histogram += _hist
        np_file.close()

    print()
    print('set meta information')

    ds_dists.attrs['minimum'] = minimum
    ds_dists.attrs['maximum'] = maximum

    # minimum distance = 0, maximum distance = components
    assert histogram.shape[0] == components + 1

    ds_hist = fd.create_dataset('histogram', data=histogram)
    ds_hist.attrs['bin_edges'] = bin_edges

    fd.close()
    print('done')


def do_migrate_180801_hamming_distances():
    # inside experiments/<EXPERIMENT>

    in_suff = 'gen/hamming/hamming.??.npz'
    out_suff = 'gen/hamming-dists.h5'

    sel = ['1024x2', '128x2', '16x2', '256x2', '32x2', '512x2', '64x2']
    it = list(zip(sel, [int(s.split('x')[0]) + 1 for s in sel]))

    for name, components in it:
        migrate_180801_hamming_distances(
            name + '/' + in_suff, name + '/' + out_suff, components)
