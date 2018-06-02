# -*- coding: utf-8 -*-

import hashlib
import logging
import os

GCUBE_TOKEN = os.environ.get('GCUBE_TOKEN')

logger = logging.getLogger(__name__)


class Mention(object):
    __slots__ = ('_title', '_text', '_span')

    def __init__(self, title, text, span):
        self._title = title
        self._text = text
        self._span = span

    @property
    def title(self):
        return self._title

    @property
    def text(self):
        return self._text

    @property
    def span(self):
        return self._span

    def __repr__(self):
        return '<Mention %s -> %s>' % (self._text.encode('utf-8'), self._title.encode('utf-8'))

    def __reduce__(self):
        return (self.__class__, (self._title, self._text, self._span))


class TagmeEntityLinker(object):
    def __init__(self, entity_db, min_link_prob, min_disambi_score, cache_dict={}):
        self._entity_db = entity_db
        self._cache_dict = cache_dict

        self._min_link_prob = min_link_prob
        self._min_disambi_score = min_disambi_score

    @property
    def min_link_prob(self):
        return self._min_link_prob

    @min_link_prob.setter
    def min_link_prob(self, value):
        self._min_link_prob = value

    @property
    def min_disambi_score(self):
        return self._min_disambi_score

    @min_disambi_score.setter
    def min_disambi_score(self, value):
        self._min_disambi_score = value

    def detect_mentions(self, text, max_mention_len=100):
        cache_key = hashlib.md5(text.encode('utf-8')).hexdigest()
        if cache_key in self._cache_dict:
            responses = self._cache_dict[cache_key]

        else:
            try:
                responses = self.get_tagme_results(text)
            except KeyboardInterrupt:
                raise
            except:
                logger.error('Unknown error: %s', text)
                return []

        ret = []

        for item in responses:
            offset = int(item['start'])
            text = item['spot']
            if item['link_probability'] < self._min_link_prob:
                continue

            if item['rho'] < self._min_disambi_score:
                continue

            span = (offset, offset + len(text))
            if 'title' not in item:
                continue

            title = self._entity_db.resolve_redirect(item['title'])
            ret.append(Mention(title, text, span))

        return ret

    @staticmethod
    def get_tagme_results(text):
        import requests
        return requests.post('https://tagme.d4science.org/tagme/tag',
                             data={'lang': 'en', 'gcube-token': GCUBE_TOKEN, 'text': text},
                             headers=dict(Accept='application/json')).json()['annotations']
