# -*- coding: utf-8 -*-

import joblib
import logging
from collections import Counter
from marisa_trie import Trie

from tokenizer import RegexpTokenizer

logger = logging.getLogger(__name__)


class Vocab(object):
    def __init__(self, dic, start_index=0):
        if isinstance(dic, Trie):
            self._dic = dic
        else:
            self._dic = Trie(dic)

        self._start_index = start_index

    @property
    def size(self):
        return len(self)

    def __len__(self):
        return len(self._dic)

    def __iter__(self):
        return iter(self._dic)

    def __contains__(self, key):
        return key in self._dic

    def get_index(self, key, default=None):
        try:
            return self._dic.key_id(key) + self._start_index
        except KeyError:
            return default

    def get_key_by_index(self, index):
        return self._dic.restore_key(index - self._start_index)

    def save(self, out_file):
        joblib.dump(self.serialize(), out_file)

    def serialize(self):
        return dict(dic=self._dic.tobytes(), start_index=self._start_index)


class WordVocab(Vocab):
    def __init__(self, dic, lowercase, start_index=0):
        super(WordVocab, self).__init__(dic, start_index)
        self._lowercase = lowercase

    def __contains__(self, word):
        if self._lowercase:
            word = word.lower()

        return word in self._dic

    def get_index(self, word, default=None):
        if self._lowercase:
            word = word.lower()

        return super(WordVocab, self).get_index(word, default)

    @staticmethod
    def build(description_db, start_index, min_count, lowercase, target_vocab=None):
        counter = Counter()
        tokenizer = RegexpTokenizer()

        for (title, text, _) in description_db.iterator():
            if target_vocab is not None and title not in target_vocab:
                continue

            if lowercase:
                counter.update([t.text.lower() for t in tokenizer.tokenize(text)])
            else:
                counter.update([t.text for t in tokenizer.tokenize(text)])

        dic = Trie([w for (w, c) in counter.iteritems() if c >= min_count])

        return WordVocab(dic, lowercase, start_index)

    def save(self, out_file):
        joblib.dump(self.serialize(), out_file)

    def serialize(self):
        return dict(dic=self._dic.tobytes(), lowercase=self._lowercase,
                    start_index=self._start_index)

    @staticmethod
    def load(input):
        if isinstance(input, dict):
            obj = input
        else:
            obj = joblib.load(input)

        dic = Trie()
        dic.frombytes(obj['dic'])
        return WordVocab(dic, obj['lowercase'], obj.get('start_index', 0))


class EntityVocab(Vocab):
    @staticmethod
    def build(description_db, entity_db, white_list, start_index, min_inlink_count,
              target_vocab=None):
        counter = Counter()
        db_titles = set()

        for (title, _, titles) in description_db.iterator():
            if target_vocab is not None and title not in target_vocab:
                continue

            counter.update(titles)
            db_titles.add(title)

        title_list = [t for (t, c) in counter.iteritems() if c >= min_inlink_count]

        white_list = [entity_db.resolve_redirect(t) for t in white_list]
        white_list = [t for t in white_list if t in db_titles]

        title_list = set(title_list + white_list)

        return EntityVocab(Trie(title_list), start_index)

    @staticmethod
    def load(input):
        if isinstance(input, dict):
            obj = input
        else:
            obj = joblib.load(input)

        dic = Trie()
        dic.frombytes(obj['dic'])
        return EntityVocab(dic, obj.get('start_index', 0))
