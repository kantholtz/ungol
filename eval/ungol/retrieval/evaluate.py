#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ungol.common import logger
from ungol.retrieval import common
from ungol.retrieval import clients
from ungol.retrieval import experiment

from ungol.similarity import sim as uss
from ungol.similarity import rhwmd as usr

import random
import pickle
import pathlib
import argparse
import functools
import collections

from tabulate import tabulate

from typing import List
from typing import Tuple
from typing import Union
from typing import Generator


log = logger.get('retrieval.evaluate')
Stat = collections.namedtuple('Stat', ('name', 'f_dataset', 'f_name', 'stat'))

#  ---


UNGOL_STRATS = {
    'min': usr.Strategy.MIN,
    'max': usr.Strategy.MAX,
    'adaptive-small': usr.Strategy.ADAPTIVE_SMALL,
    'adaptive-big': usr.Strategy.ADAPTIVE_BIG,
    'sum': usr.Strategy.SUM,
}

UNGOL_SCORERS = {
    'rhwmd': uss.rhwmd,
    'bm25': uss.bm25,
    # 'rhwmd25': uss.rhwmd25,
    'tfidf': uss.tfidf,
}


def _ungol_scorers(
        args: argparse.Namespace) -> Generator[
            Tuple[str, str], None, None]:

    if not len(args.ungol_scorers):
        raise Exception('provide --ungol-scorer')

    if args.ungol_scorers[0] == 'all':
        scorers = list(UNGOL_SCORERS.keys())
    else:
        scorers = args.ungol_scorers

    if args.ungol_strategy[0] == 'all':
        strats = list(UNGOL_STRATS.keys())
    else:
        strats = args.ungol_strategy

    if not all(k in UNGOL_SCORERS for k in scorers):
        raise Exception('unknown, check --ungol-scorer')

    for scorer in scorers:
        if scorer.startswith('rhwmd'):
            for strat in strats:
                yield scorer, strat
        else:
            yield scorer, None


# utility


def _write_experiment(folder: pathlib.Path, stat: Stat):

    folder.mkdir(exist_ok=True, parents=True)
    log.info(f'writing {folder / stat.f_name}')

    with open(folder / (stat.f_name + '.pickle'), 'wb') as fd:
        pickle.dump(stat.stat, fd)

    with open(folder / (stat.f_name + '.txt'), 'w') as fd:
        fd.write(str(stat.stat))


def _load_preselection_dataset(f_dataset: str):
    log.info('loading pre-selection dataset')
    with open(args.dataset, 'rb') as fd:
        dataset = pickle.load(fd)

    return dataset


# factories from args


def _es_client(args: argparse.Namespace) -> clients.Elasticsearch:
    host, port = (args.elastic_endpoint or common.ES_ENDPOINT).split(':')
    es_conf = {'host': host, 'port': int(port)}

    # url = 'http://{host}:{port}'.format(**es_conf)
    # experiment._es_set_bm25_params(url, k1=1.56, b=0.45)

    return clients.Elasticsearch(es_conf)


def _un_client(
        args: argparse.Namespace, scorer,
        strat=None, index=None) -> clients.Ungol:

    assert args.ungol_index, 'provide --ungol-index'

    if scorer.startswith('rhwmd'):
        fn = functools.partial(
            UNGOL_SCORERS[scorer],
            strategy=UNGOL_STRATS[strat],
            verbose=args.ungol_verbose)

        name = f'{scorer}.{strat}'

    else:
        fn = functools.partial(
            UNGOL_SCORERS[scorer],
            verbose=args.ungol_verbose)

        name = scorer

    return clients.Ungol(
        name=name, fn=fn,
        index_file=args.ungol_index,
        multiprocessing=args.ungol_mp,
        verbose=args.ungol_verbose,
        index=index)


def _wmd_client(args: argparse.Namespace) -> clients.WMD:
    assert args.wmd_embeddings, 'provide --wmd-embeddings'
    assert args.ungol_index, 'provide --ungol-index'

    return clients.WMD(args.wmd_embeddings, args.ungol_index)


def _wmd_relaxed_client(args: argparse.Namespace) -> clients.WMDRelax:
    assert args.wmd_embeddings, 'provide --wmd-embeddings'
    assert args.ungol_index, 'provide --ungol-index'

    return clients.WMDRelax(args.wmd_embeddings, args.ungol_index)

# endpoints


def elastic(args, truth, folder) -> Union[Stat, None]:
    stat = None

    client = _es_client(args)
    log.info('evaluating elasticsearch')
    f_dataset = pathlib.Path(args.dataset).name,

    def _write(content: str, f_name: str):
        path = folder / f_name
        log.info('writing', path)
        with open(path, 'w') as fd:
            fd.write(res)

    if args.elastic_find_k:
        res = experiment.es_find_k(client, truth)
        stat = Stat(
            name='elastic.bm25.find-k',
            f_name='elastic_find_k',
            f_dataset=f_dataset,
            stat=res)

    elif args.elastic_gridsearch:
        res = experiment.es_gridsearch(client, truth)
        log.info('\n' + res + '\n')
        _write(res, 'elastic_gridsearch')

    else:
        assert args.k, 'provide -k'
        res = experiment.run(client, truth, k=args.k)
        stat = Stat(
            name='elastic.bm25',
            f_name='elastic',
            dataset=f_dataset,
            stat=res)

    log.info(f'search took {client.time:.5f}ms on average')
    return stat


def ungol(args, truth, client) -> Stat:
    assert args.dataset
    log.info('evaluating ungol')

    dataset = _load_preselection_dataset(args.dataset)

    def hook(top_id):
        only = dataset[top_id]
        random.shuffle(only)
        return client.retrieve(top_id, k=args.k, only=only)

    log.info(f'running {client.name}')
    res = experiment.run(client, truth, hook=hook)

    stem = pathlib.Path(args.dataset).stem
    suffix = stem
    suffix += '_' + client.name.lower()

    stat = Stat(
        name=client.name,
        f_name=f'ungol_{suffix}',
        f_dataset=pathlib.Path(args.dataset).name,
        stat=res)

    log.info(f'search took {client.time:.5f}ms on average')

    client.close()
    return stat


def ungol_reranking(args, truth, folder: pathlib.Path, un) -> Stat:
    log.info('evaluating ungol with reranking')

    assert args.preselect, 'provide --preselect'
    assert args.k, 'provide -k'

    es = _es_client(args)

    client = clients.UngolReranking(un, es, preselect=args.preselect)
    res = experiment.run(client, truth, k=args.k)

    suffix = 'mp' if args.ungol_mp else 'sp'
    suffix += '_' + client.un.name.lower()

    stat = Stat(
        name=client.name,
        f_name=f'ungol_reranking_{suffix}',
        f_dataset=pathlib.Path(args.dataset).name,
        stat=res)

    log.info(f'es search took {client.es.time:.5f}ms on average')
    log.info(f'un search took {client.un.time:.5f}ms on average')

    client.un.close()
    return stat


def wmd(args, truth) -> Stat:
    assert args.k, 'provide -k'
    assert args.dataset, 'provide --dataset'
    log.info('evaluating wmd')

    dataset = _load_preselection_dataset(args.dataset)
    client = _wmd_client(args)

    def hook(top_id):
        only = dataset[top_id]
        random.shuffle(only)
        return client.retrieve(top_id, k=args.k, only=only)

    res = experiment.run(client, truth, hook=hook)

    stem = pathlib.Path(args.dataset).stem
    f_name = f'wmd_{stem}'
    stat = Stat(
        name='wmd',
        f_name=f_name,
        f_dataset=pathlib.Path(args.dataset).name,
        stat=res)

    log.info(f'search took {client.time:.5f}ms on average')
    return stat


def wmd_reranking(args, truth) -> Stat:
    log.info('evaluating wmd with reranking')
    assert args.preselect, 'provide --preselect'
    assert args.k, 'provide -k'

    wmd = _wmd_client(args)
    client = clients.WMDReranking(wmd)
    res = experiment.run(client, truth)

    return Stat(
        name='wmd.reranking',
        f_name='wmd_reranking',
        f_dataset=pathlib.Path(args.dataset).name,
        stat=res)


def wmd_relaxed(args, truth) -> Stat:
    assert args.k, 'provide -k'
    assert args.dataset, 'provide --dataset'
    log.info('evaluating relaxed wmd')

    dataset = _load_preselection_dataset(args.dataset)
    client = _wmd_relaxed_client(args)

    def hook(top_id):
        only = dataset[top_id]
        random.shuffle(only)
        return client.retrieve(top_id, k=args.k, only=only)

    res = experiment.run(client, truth, hook=hook)

    stem = pathlib.Path(args.dataset).stem
    f_name = f'wmd-relaxed_{stem}'
    stat = Stat(
        name='wmd-relaxed',
        f_name=f_name,
        f_dataset=pathlib.Path(args.dataset).name,
        stat=res)

    return stat


def _print_stagg(stagg):

    headers = ('name', 'dataset', 'μAP', 'Δt')
    rows = [
        (s.name, s.f_dataset, s.stat.mean_avg_precision, s.stat.mean_timedelta)
        for s in stagg]

    print(tabulate(rows, headers=headers))


def main(args):
    log.info('running evaluation')
    print('')

    # positional args
    truth = experiment.Truth.from_file(args.truth)
    folder = pathlib.Path(args.out)

    stagg: List[Stat] = []

    def add_to_stagg(stat):
        log.info('{}: μAP {:.3f}; ~ {}'.format(
            stat.name,
            stat.stat.mean_avg_precision * 100,
            stat.stat.mean_timedelta))

        # _write_experiment(folder, stat)
        stagg.append(stat)

    # elasticsearch using whole corpus
    if args.elastic:
        stat = elastic(args, truth, folder)
        if stat is not None:
            add_to_stagg(stat)

    # rhwmd
    if args.ungol:
        index = None  # cache

        for scorer, strat in _ungol_scorers(args):
            client = _un_client(args, scorer, strat=strat, index=index)
            index = client.index

            stat = ungol(args, truth, client)
            add_to_stagg(stat)

    # ungol with pre-selection from elasticsearch
    if args.ungol_reranking:
        index = None  # cache

        for scorer, strat in _ungol_scorers(args):
            client = _un_client(args, scorer, strat=strat, index=index)
            index = client.index

            stat = ungol_reranking(args, truth, folder)
            add_to_stagg(stat)

    # wmd (gensim implementation)
    if args.wmd:
        stat = wmd(args, truth)
        add_to_stagg(stat)

    # wmd (vmarkovtsev implementation)
    if args.wmd_relaxed:
        stat = wmd_relaxed(args, truth)
        add_to_stagg(stat)

    _print_stagg(stagg)

    log.info('writing stat aggregation')
    f_dataset = pathlib.Path(args.dataset).name
    with (pathlib.Path(args.out) / f'{f_dataset}').open('wb') as fd:
        pickle.dump(stagg, fd)

    log.info('done.')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'truth', type=str,
        help='ground truth file', )

    parser.add_argument(
        'out', type=str,
        help='folder to write to', )

    parser.add_argument(
        '--elastic', action='store_true', default=False, )

    parser.add_argument(
        '--wmd', action='store_true', default=False, )

    parser.add_argument(
        '--wmd-relaxed', action='store_true', default=False, )

    parser.add_argument(
        '--ungol', action='store_true', default=False, )

    parser.add_argument(
        '--ungol-reranking', action='store_true', default=False, )

    # additional general options

    parser.add_argument(
        '-k', type=int,
        help='how many results are retrieved')

    parser.add_argument(
        '--pooled', action='store_true', default=False,
        help='use only the pooled documents as preselection')

    parser.add_argument(
        '--preselect', type=int,
        help='number of documents preselected by elasticsearch')

    parser.add_argument(
        '--dataset', type=str,
        help='dataset with preselection')

    # additional options for --elastic

    parser.add_argument(
        '--elastic-endpoint', type=str,
        help='overwrite endpoint in form of <HOST>:<PORT>')

    parser.add_argument(
        '--elastic-find-k', action='store_true', default=False,
        help='find the optimal k value to maximise recall')

    parser.add_argument(
        '--elastic-gridsearch', action='store_true', default=False,
        help='do a grid search over k/b args of BM25')

    # additional options for --ungol*

    parser.add_argument(
        '--ungol-index', type=str,
        help='index produced by index.setup', )

    _opts = '|'.join(['all'] + list(UNGOL_SCORERS.keys()))
    parser.add_argument(
        '--ungol-scorers', type=str, nargs='+', default=['all'],
        help=f'similarity measure: {_opts}')

    parser.add_argument(
        '--ungol-mp', action='store_true', default=False,
        help='use multiple processes')

    _opts = '|'.join(['all'] + list(UNGOL_STRATS.keys()))
    parser.add_argument(
        '--ungol-strategy', type=str, nargs='+', default=['all'],
        help=f'one of {_opts}')

    parser.add_argument(
        '--ungol-verbose', action='store_true', default=False,
        help='write extensive reports')

    # additional options for --wmd*

    parser.add_argument(
        '--wmd-embeddings', type=str,
        help='embeddings to be loaded by gensim (.bin)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
