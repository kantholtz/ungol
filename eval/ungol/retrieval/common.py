# -*- coding: utf-8 -*-

import dumpr.common as dumpr

import attr
import elasticsearch as es

from datetime import datetime

from typing import Any
from typing import Dict
from typing import Callable


# ---


# defaults
ES_ENDPOINT = '172.26.90.209:9200'  # for (jf)
CTX_ARTICLES = {'index': 'articles', 'doc_type': 'doc'}
CTX_TOPICS = {'index': 'topics', 'doc_type': 'doc'}


# --- utility


def get_client(conf: Dict[str, Any]) -> es.Elasticsearch:
    assert 'host' in conf
    assert 'port' in conf

    client = es.Elasticsearch([conf])
    if not client.ping():
        raise Exception('could not reach elasticsearch')

    print('[common] connected to elasticsearch')
    return client


def timer() -> Callable[[], float]:  # in miliseconds
    stamp = datetime.now()

    def done() -> float:
        delta = datetime.now() - stamp
        return delta.total_seconds() * 1000

    return done


# --- data model (for orm)


@attr.s
class Article():

    mapping = {
        'mappings': {
            'doc': {
                'properties': {
                    'content': {'type': 'text', 'analyzer': 'german', },
                    'title': {'type': 'text', 'analyzer': 'german', },
                    'fname': {'type': 'keyword', },
                    'keywords': {'type': 'keyword', }
                }
            }
        }
    }

    content:  str = attr.ib()
    title:    str = attr.ib()
    fname:    str = attr.ib()
    keywords: str = attr.ib()

    def __str__(self) -> str:
        threshold = 2048
        suffix = '...' if len(self.content) > threshold else ''
        text = self.content[:threshold] + suffix

        sbuf = [f'{self.title}\n']
        sbuf.append(f'Keywords: {self.keywords}')
        sbuf.append(f'Filename: {self.fname}\n')
        sbuf.append(f'Content Excerpt:\n\n{text}')

        return '\n'.join(sbuf)

    @staticmethod
    def from_document(doc: dumpr.Document):
        """
        Create from dumpr Documents

        """
        # id: spiegel, frankfurter rs
        # docid/docno: sda94, sda95
        doc_id = doc.meta['id'] if 'id' in doc.meta else doc.meta['docid']

        # title-X: spiegel, frankfurter rs
        # tite: sda94, sda95
        if 'title' in doc.meta:
            titles = [doc.meta['title']]
        else:
            titles = [doc.meta[key] for key in sorted(doc.meta.keys())
                      if key.startswith('title-')]

        rep = {
            'title': ' '.join(titles),
            'content': doc.content,
            'fname': doc.meta['file'] if 'file' in doc.meta else '',
            'keywords': doc.meta['kw'] if 'kw' in doc.meta else ''
        }

        # you can do article = Article(**rep)
        return doc_id, rep

    @staticmethod
    def from_es(es_client: es.Elasticsearch, doc_id: str):
        ctx = CTX_ARTICLES
        res = es_client.get(id=doc_id, **ctx)

        if not res['found']:
            raise Exception('Could not find document "{}"'.format(doc_id))

        return Article(**res['_source'])


@attr.s
class Topic():

    mapping = {
        'mappings': {
            'doc': {
                'properties': {
                    'top_id': {'type': 'keyword', },
                    'title': {'type': 'text', 'analyzer': 'german', },
                    'description': {'type': 'text', 'analyzer': 'german', },
                    'narrative': {'type': 'text', 'analyzer': 'german', },
                }
            }
        }
    }

    top_id:      str = attr.ib()
    title:       str = attr.ib()
    description: str = attr.ib()
    narrative:   str = attr.ib()

    def __str__(self):
        return "[{}] {}\n- Description: {}\n- Narrative: {}\n".format(
            self.top_id, self.title, self.description, self.narrative)

    @staticmethod
    def from_es(es_client: es.Elasticsearch, top_id: str) -> 'Topic':
        ctx = CTX_TOPICS
        res = es_client.get(id=top_id, **ctx)

        if not res['found']:
            raise Exception('Could not find topic "{}"'.format(top_id))

        return Topic(**res['_source'])

    @staticmethod
    def from_soup(soup) -> Dict[str, str]:
        # The double call to str(<soup>.string) is because
        # <soup>.string returns a bs4.element.NavigableString object
        # which is not picklable. This seems to be the
        # downside of duck typing.

        args = 'title', 'description', 'narrative'
        dic = {s: str(soup.find(s).string) for s in args}
        dic['top_id'] = str(soup.find('identifier').string)

        return dic
