# -*- coding: utf-8 -*-

import logging
import multiprocessing
import numpy as np
import time
import Queue
import random
import torch
from torch.autograd import Variable

from utils.tokenizer import RegexpTokenizer

logger = logging.getLogger(__name__)

_description_db = None
_word_vocab = None
_entity_vocab = None
_target_entity_vocab = None


class DBIterator(multiprocessing.Process):
    def __init__(self, key_queue, iter_done):
        multiprocessing.Process.__init__(self)

        self._key_queue = key_queue
        self._iter_done = iter_done

    def run(self):
        keys = [k for k in _description_db.keys() if k in _target_entity_vocab]
        random.shuffle(keys)

        for key in keys:
            self._key_queue.put(key)

        self._iter_done.set()


class BatchGenerator(multiprocessing.Process):
    def __init__(self, key_queue, batch_queue, iter_done, batch_done, entity_negatives,
                 batch_size, negative, max_text_len, max_entity_len):
        multiprocessing.Process.__init__(self)

        self._key_queue = key_queue
        self._batch_queue = batch_queue
        self._iter_done = iter_done
        self._batch_done = batch_done

        self._entity_negatives = entity_negatives
        self._batch_size = batch_size
        self._negative = negative
        self._max_text_len = max_text_len
        self._max_entity_len = max_entity_len

    def run(self):
        self._tokenizer = RegexpTokenizer()

        buf = []
        while True:
            try:
                key = self._key_queue.get(False, 1)
            except Queue.Empty:
                if self._iter_done.is_set():
                    if buf:
                        for batch in self._generate_batch(buf):
                            self._batch_queue.put(batch)
                    break
                else:
                    time.sleep(0.1)
                    continue

            item = self._process(key)
            if item is None:
                continue
            buf.append(item)

            if len(buf) >= self._batch_size * 100:
                for batch in self._generate_batch(buf):
                    self._batch_queue.put(batch)

                buf = []

        if buf:
            for batch in self._generate_batch(buf):
                self._batch_queue.put(batch)

        self._batch_done.set()

    def _process(self, title):
        (text, link_titles) = _description_db[title]

        word_indices = []
        for token in self._tokenizer.tokenize(text):
            word_index = _word_vocab.get_index(token.text)
            if word_index is not None:
                word_indices.append(word_index)

        if not word_indices:
            return None

        entity_indices = []
        for link_title in link_titles:
            entity_index = _entity_vocab.get_index(link_title)
            if entity_index is not None:
                entity_indices.append(entity_index)

        positive_id = _target_entity_vocab.get_index(title)
        if positive_id is None:
            return None

        target_entity_indices = np.empty(self._negative + 1, dtype=np.int)
        target_entity_indices[0] = positive_id
        negative_ids = set()
        while True:
            negative_id = np.random.choice(self._entity_negatives)
            if negative_id != positive_id and negative_id not in negative_ids:
                negative_ids.add(negative_id)
                if len(negative_ids) == self._negative:
                    break

        target_entity_indices[1:] = list(negative_ids)

        return (word_indices, entity_indices, target_entity_indices)

    def _generate_batch(self, buf):
        buf = sorted(buf, key=lambda o: len(o[0]), reverse=True)

        for i in range(0, len(buf), self._batch_size):
            items = buf[i:i + self._batch_size]
            max_text_len = max(min(len(items[0][0]), self._max_text_len), 1)
            word_batch = np.zeros((len(items), max_text_len), dtype=np.int)

            max_entity_len = max(min(max([len(o[1]) for o in items]), self._max_entity_len), 1)
            entity_batch = np.zeros((len(items), max_entity_len), dtype=np.int)

            target_entity_batch = []

            for (j, (word_indices, entity_indices, target_entity_indices)) in enumerate(items):
                word_indices = word_indices[:max_text_len]
                word_batch[j][:len(word_indices)] = word_indices

                entity_indices = entity_indices[:max_entity_len]
                entity_batch[j][:len(entity_indices)] = entity_indices

                target_entity_batch.append(target_entity_indices)

            yield ((
                Variable(torch.LongTensor(word_batch)),
                Variable(torch.LongTensor(entity_batch)),
                Variable(torch.LongTensor(target_entity_batch)),
            ), Variable(torch.LongTensor(np.zeros(len(items), dtype=np.int))))


def generate_data(description_db, word_vocab, entity_vocab, target_entity_vocab,
                  entity_negatives, batch_size, negative, max_text_len, max_entity_len,
                  pool_size, key_queue_size=1000, batch_queue_size=10000):
    global _description_db, _word_vocab, _entity_vocab, _target_entity_vocab
    _description_db = description_db
    _word_vocab = word_vocab
    _entity_vocab = entity_vocab
    _target_entity_vocab = target_entity_vocab

    key_queue = multiprocessing.Queue(key_queue_size)
    batch_queue = multiprocessing.Queue(batch_queue_size)
    iter_done = multiprocessing.Event()

    task_generator = DBIterator(key_queue, iter_done)
    task_generator.daemon = True
    task_generator.start()

    batch_generators = []
    batch_done_events = []
    for n in range(pool_size):
        batch_done = multiprocessing.Event()
        batch_generator = BatchGenerator(
            key_queue, batch_queue, iter_done, batch_done, entity_negatives,
            batch_size, negative, max_text_len, max_entity_len
        )
        batch_generator.daemon = True
        batch_generator.start()
        batch_done_events.append(batch_done)
        batch_generators.append(batch_generator)

    try:
        while True:
            try:
                batch = batch_queue.get(True, 1)
                yield batch

            except Queue.Empty:
                if all([d.is_set() for d in batch_done_events]):
                    break
                logging.debug('The batch queue is empty (task queue: %d result queue: %d)' %
                              (key_queue.qsize(), batch_queue.qsize()))

    finally:
        task_generator.terminate()
        for batch_generator in batch_generators:
            batch_generator.terminate()
