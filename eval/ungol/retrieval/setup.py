#!/usr/bin/env python
# -*- coding: utf-8 -*-


import dumpr.common as dumpr

from ungol.index import index as uii
from ungol.index import setup as uis

from ungol.common import logger
from ungol.retrieval import common

import elasticsearch as es
from tqdm import tqdm as _tqdm
from bs4 import BeautifulSoup as bs

import os
import sys
import pathlib
import argparse
import functools
import multiprocessing as mp

from typing import Dict
from typing import Tuple
from typing import Generator
from typing import Collection


# ---

sys.path.append('lib/CharSplit')
import char_split  # noqa

# ---


log = logger.get('retrieval.setup')
tqdm = functools.partial(_tqdm, ncols=80)


# ---

def _parse_topics(files: Collection[str]) -> Generator[
        common.Topic, None, None]:

    for xml in files:
        with open(xml, mode='r', encoding='utf-8') as f:
            topics_raw = f.read()

        soup = bs(topics_raw, 'xml')

        print('')
        for topic_node in tqdm(soup.find_all('topic')):
            yield common.Topic.from_soup(topic_node)


# --- setup


def _es_setup_articles_async_fill(argv) -> int:
    w_id, xml = argv

    ctx = common.CTX_ARTICLES
    client = es.Elasticsearch([common.ES_CONF])
    if not client.ping():
        raise Exception('could not reach elasticsearch')

    with (pathlib.Path(xml).parents[0] / 'count.txt').open(mode='r') as fd:
        total = int(fd.read())

    with dumpr.BatchReader(xml) as reader:
        for count, doc in tqdm(enumerate(reader), position=w_id, total=total):
            doc_id, rep = common.Article.from_document(doc)

            res = client.index(id=doc_id, body=rep, **ctx)
            if not res['result'] == 'created':
                raise Exception('could not create {}: {}'.format(
                    str(doc_id, res)))

    return count


def _es_setup_idx(
        client: es.Elasticsearch,
        ctx: Dict,
        mapping: Dict,
        force: bool = False) -> bool:

    if client.indices.exists(index=ctx['index']):
        if force:
            res = client.indices.delete(index=ctx['index'])
            print('delete index:', res)

        else:
            print('index {index} already exists, skipping'.format(**ctx))
            return False

    res = client.indices.create(index=ctx['index'], body=mapping)
    print('re-create index:', res)
    return True


def do_elastic_setup_articles(args: argparse.Namespace) -> int:
    files = args.files

    ctx = common.CTX_ARTICLES
    client = common.get_client()
    mapping = common.Article.mapping

    created = _es_setup_idx(client, ctx, mapping, force=args.force)
    if not created:
        return 0

    # parallel
    with mp.Pool(max(4, len(files))) as pool:
        args = tuple(zip(range(len(files)), files))
        counts = pool.map(_es_setup_articles_async_fill, args)

    # consecutively / good for testing.
    # counts = []
    # for argv in enumerate(files):
    #     counts.append(_setup_es_async_fill(argv))

    print('processed', sum(counts), 'elements')
    return 0


def do_elastic_setup_topics(args: argparse.Namespace) -> int:
    ctx = common.CTX_TOPICS
    client = common.get_client()

    created = _es_setup_idx(
        client, ctx, common.Topic.mapping, force=args.force)

    if not created:
        return 0

    count = 0
    for topic in _parse_topics(args.files):
        res = client.index(id=topic['top_id'], body=topic, **ctx)
        if not res['result'] == 'created':
            raise Exception('could not create {}: {}'.format(
                str(topic['top_id'], res)))

        count += 1

    print('  imported {} topics'.format(count))
    return 0


def do_ungol_setup_articles(args: argparse.Namespace) -> int:

    assert args.out, 'no --out specified'
    assert args.files, 'no --files for articles provided'
    assert args.vocabulary, 'no --vocabulary provided'
    assert args.ungol_codemap, 'no --ungol-codemap provided'

    pid = os.getpid()

    log.info(f'[{pid}] initializing new index')
    ref_args = args.ungol_codemap, args.vocabulary, args.stopwords
    ref = uii.References.from_files(*ref_args)
    idx = uii.Index(ref=ref)

    def _gen(xml) -> Generator[Tuple[str, str], None, int]:
        pid = os.getpid()
        log.info(f'[{pid}] opening {xml}')

        count = 0
        with dumpr.BatchReader(xml) as reader:
            for doc in reader:
                doc_id, article = common.Article.from_document(doc)
                text = '\n\n'.join((article['title'], article['content']))

                yield doc_id, text
                count += 1

        return count

    with uis.Indexer(index=idx, processes=args.processes) as insert:
        for xml in args.files:
            gen = functools.partial(_gen, xml)
            insert(gen)

    log.info(f'[{pid}] finished creating new index')

    log.info(f'[{pid}] writing "{args.out}"')
    idx.to_file(args.out)
    return 0


def do_ungol_setup_topics(args: argparse.Namespace) -> int:
    assert args.out, 'no --out specified'
    assert args.files, 'no --files for topics provided'

    idx = uii.Index.from_file(args.out)

    def _gen() -> Generator[Tuple[str, str], None, int]:
        pid = os.getpid()
        log.info(f'[{pid}] opening {args.files}')

        count = 0
        for topic in _parse_topics(args.files):
            top_id = topic['top_id']
            content = topic['description']
            yield top_id, content
            count += 1

        return count

    with uis.Indexer(index=idx, processes=args.processes) as insert:
        count = insert(_gen)

    print('saving new version of the database')
    idx.to_file(args.out)

    print(f'imported {count} topics'.format(count))
    return 0


# ---


DESC = '''
  V N G O L - E S

  Baseline experiment infrastructure for database setup. The following
  commands are supported:

  Commands:
    elastic-setup-articles --files XML_FILES [--force]
      Processes the list of xml files, transforms them
      to objects fitting to Article.mapping and puts
      them into elasticsearch. If --force is set, index
      is deleted if it exists and re-created.

    elastic-setup-topics --files XML_FILES [--force]
      Processes the list of xml files, transforms them
      to objects fitting to Topic.mapping and puts
      them into elasticsearch. If --force is set, index
      is deleted if it exists and re-created.

    ungol-setup-articles --files XML_FILES
      Using dumpr, the list of xml files is processed
      for use by the Ungol client. Additional options:

        --out FILE specifies the target to write the
          database to

        --vocabulary FILE pickled dictionary mapping
          words to indexes

        [--stopwords FILE...] optional enumeration of
          text files containing words to be ignored

    ungol-setup-topics --files XML_FILES
      Adds the provided topics to the databse

        --out FILE the database file produced by
          ungol-setup-articles
'''


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=DESC)

    parser.add_argument(
        'cmd', help='command to execute')

    parser.add_argument(
        '--files', nargs='*', type=str,
        help='list of files')

    parser.add_argument(
        '-f', '--force', action='store_true',
        help='force cmd')

    parser.add_argument(
        '-o', '--out', type=str,
        help='path to target if a file is written', )

    parser.add_argument(
        '--processes', type=int, default=4,
        help='amount of processes used for pre-processing (default 4)')

    # setup-ungol

    parser.add_argument(
        '--vocabulary', type=str,
        help='for ungol-*: vocabulary pickle', )

    parser.add_argument(
        '--stopwords', type=str, nargs='+',
        help='for ungol-*: stopword files', )

    parser.add_argument(
        '--ungol-codemap', type=str,
        help='for ungol-*; codemap produced by embcodr')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    print('\nwelcome to vngol retrieval\n'.upper())
    rc = globals()['do_' + args.cmd.replace('-', '_')](args)
    sys.exit(rc)
