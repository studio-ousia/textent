# -*- coding: utf-8 -*-

import joblib
import numpy as np

from marisa_trie import Trie
from vocab import WordVocab, EntityVocab


class EmbeddingReader(object):
    def __init__(self, word_embedding, entity_embedding, word_vocab, entity_vocab):
        self._word_embedding = word_embedding
        self._entity_embedding = entity_embedding
        self._word_vocab = word_vocab
        self._entity_vocab = entity_vocab

    @property
    def word_embedding(self):
        return self._word_embedding

    @property
    def entity_embedding(self):
        return self._entity_embedding

    @property
    def word_vocab(self):
        return self._word_vocab

    @property
    def entity_vocab(self):
        return self._entity_vocab

    def words(self):
        if self._word_vocab is None:
            return []
        else:
            return list(self._word_vocab)

    def entities(self):
        if self._entity_vocab is None:
            return []
        else:
            return list(self._entity_vocab)

    def get_word_vector(self, word, default=None):
        if self._word_vocab is None:
            return default

        index = self._word_vocab.get_index(word)
        if index is None:
            return default
        else:
            return self._word_embedding[index]

    def get_entity_vector(self, title, default=None):
        if self._entity_vocab is None:
            return default

        index = self._entity_vocab.get_index(title)
        if index is None:
            return default
        else:
            return self._entity_embedding[index]

    def get_word_index(self, word, default=None):
        if self._word_vocab is None:
            return default

        return self._word_vocab.get_index(word, default)

    def get_entity_index(self, title, default=None):
        if self._entity_vocab is None:
            return default

        return self._entity_vocab.get_index(title, default)

    def save(self, out_file):
        obj = dict(word_embedding=self._word_embedding, entity_embedding=self._entity_embedding)

        if self._word_vocab is not None:
            obj['word_vocab'] = self._word_vocab.serialize()
        if self._entity_vocab is not None:
            obj['entity_vocab'] = self._entity_vocab.serialize()

        joblib.dump(obj, out_file)

    @staticmethod
    def load(in_file, mmap='r'):
        obj = joblib.load(in_file, mmap_mode=mmap)

        word_vocab = None
        if 'word_vocab' in obj:
            word_vocab = WordVocab.load(obj['word_vocab'])
        entity_vocab = None
        if 'entity_vocab' in obj:
            entity_vocab = EntityVocab.load(obj['entity_vocab'])

        return EmbeddingReader(obj['word_embedding'], obj['entity_embedding'],
                               word_vocab, entity_vocab)

    @staticmethod
    def load_figment(in_file, mapping_file):
        mapping = {}
        for line in mapping_file:
            (mid, title) = line.rstrip().decode('utf-8').split('\t')
            mapping[mid] = title

        vectors = {}
        for line in in_file:
            values = line.split()
            mid = values[0]
            if mid not in mapping:
                continue
            vectors[mapping[mid]] = [float(v) for v in values[1:]]

        entity_vocab = EntityVocab(Trie(vectors.keys()))
        entity_embedding = np.zeros((len(entity_vocab), len(vectors.values()[0])))

        for title in entity_vocab:
            entity_embedding[entity_vocab.get_index(title)] = vectors[title]

        return EmbeddingReader(None, entity_embedding, None, entity_vocab)

    @staticmethod
    def load_wikipedia2vec(model_file):
        word_vectors = {}
        entity_vectors = {}

        for (n, line) in enumerate(model_file):
            (item_str, vec_str) = line.rstrip().decode('utf-8').split('\t')
            vector = [float(v) for v in vec_str.split(' ')]
            if item_str.startswith('ENTITY/'):
                title = item_str[7:].replace('_', ' ')
                entity_vectors[title] = vector
            else:
                word_vectors[item_str] = vector

        word_vocab = WordVocab(Trie(word_vectors.keys()), True)
        word_embedding = np.zeros((len(word_vocab), len(word_vectors.values()[0])))
        entity_vocab = EntityVocab(Trie(entity_vectors.keys()))
        entity_embedding = np.zeros((len(entity_vocab), len(entity_vectors.values()[0])))

        for word in word_vocab:
            word_embedding[word_vocab.get_index(word)] = word_vectors[word]
        for title in entity_vocab:
            entity_embedding[entity_vocab.get_index(title)] = entity_vectors[title]

        return EmbeddingReader(word_embedding, entity_embedding, word_vocab, entity_vocab)
