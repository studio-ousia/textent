# -*- coding: utf-8 -*-

import logging
import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
from joblib import Memory
from marisa_trie import Trie
from tempfile import NamedTemporaryFile
from torch.autograd import Variable

from utils.vocab import EntityVocab

logger = logging.getLogger(__name__)
memory = Memory('.')


class EntityTypeClassifier(nn.Module):
    def __init__(self, entity_embedding, num_classes, hidden_units):
        super(EntityTypeClassifier, self).__init__()

        self._num_classes = num_classes

        self._entity_embedding = nn.Embedding(entity_embedding.shape[0], entity_embedding.shape[1])
        self._entity_embedding.weight = nn.Parameter(torch.FloatTensor(entity_embedding).half().cuda())
        self._entity_embedding.weight.requires_grad = False

        self._hidden_layer = nn.Linear(entity_embedding.shape[1], hidden_units, bias=False)
        self._output_layer = nn.Linear(hidden_units, num_classes, bias=False)

    def forward(self, entity_indices):
        entity_emb = self._entity_embedding(entity_indices).float()

        hidden_vector = self._hidden_layer(entity_emb)
        hidden_vector = F.tanh(hidden_vector)

        return F.sigmoid(self._output_layer(hidden_vector))


def evaluate(entity_embedding, entity_vocab, entity_db, dataset_dir='.', dev_probs_file=None,
             test_probs_file=None, batch_size=32, epoch=100, patience=5, hidden_units=200,
             exclude_oov=False, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    dataset_obj = _read_dataset(dataset_dir)

    target_entities = [t for k in dataset_obj.keys() for (_, t, _, _) in dataset_obj[k]
                       if t in entity_vocab]
    target_entity_vocab = EntityVocab(Trie(target_entities), start_index=1)

    with open(os.path.join(dataset_dir, 'types')) as types_file:
        target_labels = [l.decode('utf-8').split('\t')[0][1:] for l in types_file]
    label_index = {l: n for (n, l) in enumerate(target_labels)}

    target_emb = np.empty((len(target_entity_vocab), entity_embedding.shape[1]))
    target_emb = np.vstack([np.zeros(target_emb.shape[1]), target_emb])

    for title in target_entity_vocab:
        target_emb[target_entity_vocab.get_index(title)] = entity_embedding[entity_vocab.get_index(title)]

    del entity_embedding

    type_classifier = EntityTypeClassifier(target_emb, len(target_labels), hidden_units)
    type_classifier = type_classifier.cuda()

    type_classifier.train()

    parameters = [p for p in type_classifier.parameters() if p.requires_grad]
    optimizer_ins = optim.Adam(parameters)

    def generate_batch(fold):
        index_batch = []
        target_batch = []

        for (n, (_, title, labels, count)) in enumerate(dataset_obj[fold]):
            title = entity_db.resolve_redirect(title)
            if title is None or title not in target_entity_vocab:
                if exclude_oov:
                    continue
                index_batch.append(0)
            else:
                index_batch.append(target_entity_vocab.get_index(title))

            target = np.zeros(len(target_labels), dtype=np.float)
            for label in labels:
                target[label_index[label]] = 1
            target_batch.append(target)

            if len(target_batch) == batch_size or n == len(dataset_obj[fold]) - 1:
                yield (index_batch, target_batch)

                index_batch = []
                target_batch = []

    p1_scores = [0.0]
    best_dev_probs = None

    with NamedTemporaryFile(dir='/dev/shm') as f:
        for n_epoch in range(epoch):
            for (n, (arg, target)) in enumerate(generate_batch('train')):
                arg = Variable(torch.LongTensor(arg)).cuda()
                target = Variable(torch.FloatTensor(target)).cuda()

                optimizer_ins.zero_grad()
                output = type_classifier(arg)
                loss = F.binary_cross_entropy(torch.cat(output), torch.cat(target))
                loss.backward()
                optimizer_ins.step()

            type_classifier.eval()

            correct = 0
            total = 0
            dev_probs = []
            for (arg, target) in generate_batch('dev'):
                arg = Variable(torch.LongTensor(arg)).cuda()
                output = type_classifier(arg).cpu().data.numpy()
                dev_probs.append(output)

                for (n, ind) in enumerate(np.argmax(output, axis=1)):
                    total += 1
                    if target[n][ind] == 1:
                        correct += 1

            dev_probs = np.vstack(dev_probs)

            type_classifier.train()

            p1_score = float(correct) / total
            logger.debug('P@1 (dev): %.3f (%d/%d, Epoch: %d)', p1_score, correct, total, n_epoch)

            if p1_score > max(p1_scores):
                state_dict = type_classifier.state_dict()
                torch.save(state_dict, f.name)
                best_dev_probs = dev_probs

            p1_scores.append(p1_score)

            if patience is not None:
                if len(p1_scores) - p1_scores.index(max(p1_scores)) > patience:
                    break

        state_dict = torch.load(f.name)

    if dev_probs_file:
        for ((fb_id, _, _, _), probs) in zip(dataset_obj['dev'], best_dev_probs):
            probs_str = ' '.join([str(p) for p in probs])
            dev_probs_file.write('%s\t%s\n' % (fb_id.encode('utf-8'), probs_str))

    type_classifier.load_state_dict(state_dict)
    type_classifier.eval()

    output_arr = []
    target_arr = []
    for (arg, target) in generate_batch('test'):
        arg = Variable(torch.LongTensor(arg)).cuda()
        output = type_classifier(arg).cpu().data.numpy()
        output_arr.append(output)
        target_arr.append(np.vstack(target))

    output_arr = np.vstack(output_arr)
    target_arr = np.vstack(target_arr)

    correct = 0
    for (n, (output, target)) in enumerate(zip(output_arr, target_arr)):
        if target[np.argmax(output)] == 1:
            correct += 1

    test_score = float(correct) / output_arr.shape[0]
    logger.debug('P@1 (test): %.3f (%d/%d)', test_score, correct, output_arr.shape[0])

    if test_probs_file:
        for ((fb_id, _, _, _), probs) in zip(dataset_obj['test'], output_arr):
            probs_str = ' '.join([str(p) for p in probs])
            test_probs_file.write('%s\t%s\n' % (fb_id.encode('utf-8'), probs_str))

    logger.info('P@1 (dev): %.3f', max(p1_scores))
    logger.info('P@1 (test): %.3f', test_score)

    return (max(p1_scores), test_score)


@memory.cache
def _read_dataset(dataset_dir):
    name_mapping = {'train': 'Etrain', 'dev': 'Edev', 'test': 'Etest'}

    fb_wiki_mapping = {}
    with open(os.path.join(dataset_dir, 'fb_wiki_mapping.tsv')) as mapping_file:
        for line in mapping_file:
            (fb_id, wiki_title) = line.rstrip().decode('utf-8').split('\t')
            fb_wiki_mapping[fb_id] = wiki_title

    data = defaultdict(list)

    for dataset_type in name_mapping.keys():
        with open(os.path.join(dataset_dir, name_mapping[dataset_type])) as f:
            for line in f:
                obj = [s.strip() for s in line.rstrip().decode('utf-8').split()]
                fb_id = obj[0]
                title = fb_wiki_mapping.get(fb_id)
                labels = []
                count = None
                if dataset_type == 'test':
                    count = int(obj.pop())

                for s in obj[1:]:
                    if s.startswith('-'):
                        labels.append(s.rstrip()[1:])

                data[dataset_type].append((fb_id, title, labels, count))

    return data
