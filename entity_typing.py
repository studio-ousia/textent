# -*- coding: utf-8 -*-

import click
import logging
import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
from tempfile import NamedTemporaryFile
from torch.autograd import Variable

from utils.vocab import EntityVocab

logger = logging.getLogger(__name__)


class EntityTypeClassifier(nn.Module):
    def __init__(self, entity_embedding, num_classes, num_hidden_units):
        super(EntityTypeClassifier, self).__init__()

        self._num_classes = num_classes

        self._entity_embedding = nn.Embedding(entity_embedding.shape[0], entity_embedding.shape[1])
        self._entity_embedding.weight = nn.Parameter(torch.FloatTensor(entity_embedding).cuda())
        self._entity_embedding.weight.requires_grad = False

        self._hidden_layer = nn.Linear(entity_embedding.shape[1], num_hidden_units, bias=False)
        self._output_layer = nn.Linear(num_hidden_units, num_classes, bias=False)

    def forward(self, entity_indices):
        entity_emb = self._entity_embedding(entity_indices)
        hidden_vector = F.tanh(self._hidden_layer(entity_emb))

        return F.sigmoid(self._output_layer(hidden_vector))


def evaluate(entity_embedding, entity_vocab, entity_db, dataset_dir, batch_size, epoch, patience,
             num_hidden_units, exclude_oov, temp_dir, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    dataset_obj = _read_dataset(dataset_dir)

    target_entities = [title for fold in dataset_obj.keys() for (_, title, _) in dataset_obj[fold]
                       if title in entity_vocab]
    target_entity_vocab = EntityVocab(target_entities, start_index=1)

    with open(os.path.join(dataset_dir, 'types.txt')) as types_file:
        target_labels = [l.rstrip().decode('utf-8') for l in types_file]

    label_index = {l: n for (n, l) in enumerate(target_labels)}

    target_emb = np.empty((len(target_entity_vocab), entity_embedding.shape[1]))
    target_emb = np.vstack([np.zeros(target_emb.shape[1]), target_emb])

    for title in target_entity_vocab:
        target_emb[target_entity_vocab.get_index(title)] = entity_embedding[entity_vocab.get_index(title)]

    type_classifier = EntityTypeClassifier(target_emb, len(target_labels), num_hidden_units)
    type_classifier = type_classifier.cuda()

    type_classifier.train()

    parameters = [p for p in type_classifier.parameters() if p.requires_grad]
    optimizer_ins = optim.Adam(parameters)

    def generate_batches(fold):
        index_batch = []
        target_batch = []

        for (n, (_, title, labels)) in enumerate(dataset_obj[fold]):
            title = entity_db.resolve_redirect(title)
            if title is None or title not in target_entity_vocab:
                if exclude_oov:
                    continue
                index_batch.append(0)
            else:
                index_batch.append(target_entity_vocab.get_index(title))

            target = np.zeros(len(target_labels), dtype=np.float)
            target[[label_index[l] for l in labels]] = 1
            target_batch.append(target)

            if len(target_batch) == batch_size or n == len(dataset_obj[fold]) - 1:
                yield (np.array(index_batch, dtype=np.int), np.vstack(target_batch))

                index_batch = []
                target_batch = []

    dev_p1_scores = [0.0]
    best_dev_data = None

    with NamedTemporaryFile(dir=temp_dir) as f:
        for n_epoch in range(epoch):
            for (n, (arg, target)) in enumerate(generate_batches('train')):
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
            dev_targets = []
            for (arg, target) in generate_batches('dev'):
                arg = Variable(torch.LongTensor(arg)).cuda()
                output = type_classifier(arg).cpu().data.numpy()
                dev_probs.append(output)
                dev_targets.append(target)

                correct += np.sum(target[np.arange(target.shape[0]), np.argmax(output, axis=1)])
                total += target.shape[0]

            dev_probs = np.vstack(dev_probs)
            dev_targets = np.vstack(dev_targets)

            type_classifier.train()

            p1_score = float(correct) / total
            logger.info('P@1 (dev): %.3f (%d/%d, Epoch: %d)', p1_score, correct, total, n_epoch)

            if p1_score > max(dev_p1_scores):
                torch.save(type_classifier.state_dict(), f.name)
                best_dev_data = (dev_probs, dev_targets)

            dev_p1_scores.append(p1_score)

            if patience is not None:
                if len(dev_p1_scores) - dev_p1_scores.index(max(dev_p1_scores)) > patience:
                    break

        state_dict = torch.load(f.name)

    thresholds = np.array([_compute_bep(best_dev_data[0][:, n], best_dev_data[1][:, n])[1]
                          for n in range(len(target_labels))])

    type_classifier.load_state_dict(state_dict)
    type_classifier.eval()

    output_arr = []
    target_arr = []
    for (arg, target) in generate_batches('test'):
        arg = Variable(torch.LongTensor(arg)).cuda()
        output = type_classifier(arg).cpu().data.numpy()
        output_arr.append(output)
        target_arr.append(target)

    output_arr = np.vstack(output_arr)
    target_arr = np.vstack(target_arr)

    p1_correct = 0.0
    acc_correct = 0.0
    bep_scores = []
    f1_scores = []
    binary_predictions = []
    for n in range(output_arr.shape[0]):
        if target_arr[n, np.argmax(output_arr[n])] == 1:
            p1_correct += 1
        bep_scores.append(_compute_bep(output_arr[n], target_arr[n])[0])

        binary_pred = np.greater_equal(output_arr[n], thresholds).astype(np.float)
        if np.array_equal(target_arr[n], binary_pred):
            acc_correct += 1

        binary_predictions.append(binary_pred)
        f1_scores.append(_compute_f1(target_arr[n], binary_pred))

    binary_predictions = np.vstack(binary_predictions)

    test_p1_score = p1_correct / output_arr.shape[0]

    click.echo('Best P@1 (dev): %.3f' % max(dev_p1_scores))

    click.echo('P@1: %.3f' % test_p1_score)
    click.echo('BEP: %.3f' % np.mean(bep_scores))
    click.echo('Accuracy: %.3f' % (acc_correct / output_arr.shape[0]))
    click.echo('Macro F1: %.3f' % np.mean(f1_scores))
    click.echo('Micro F1: %.3f' % _compute_f1(target_arr.flatten(), binary_predictions.flatten()))

    return (max(dev_p1_scores), test_p1_score)


def _read_dataset(dataset_dir):
    name_mapping = {'train': 'train.tsv', 'dev': 'dev.tsv', 'test': 'test.tsv'}

    data = defaultdict(list)

    for dataset_type in name_mapping.keys():
        with open(os.path.join(dataset_dir, name_mapping[dataset_type])) as f:
            for (n, line) in enumerate(f):
                if n == 0:
                    continue
                (fb_id, title, labels) = line.rstrip().decode('utf-8').split('\t')
                labels = labels.split(',')

                data[dataset_type].append((fb_id, title, labels))

    return data


def _compute_f1(labels, predictions):
    correct = float(np.sum(np.logical_and(labels, predictions)))
    if correct == 0:
        return 0.0

    total_pred = np.sum(predictions)
    if total_pred == 0:
        return 0.0
    prec = correct / total_pred
    recall = correct / np.sum(labels)
    return 2.0 * prec * recall / (prec + recall)


# Based on https://github.com/yyaghoobzadeh/figment/blob/master/src/myutils.py#L536
def _compute_bep(probs, labels):
    arr = np.vstack([probs, labels]).T
    arr = arr[arr[:, 0].argsort()[::-1]]

    correct = 0.0
    max_f1 = 0.0
    thresh = 0.0
    total = np.sum(arr[:, 1])

    for n in range(arr.shape[0]):
        if arr[n, 1] == 1.0:
            correct += 1.0
            prec = correct / (n + 1)
            recall = correct / total
            f1 = 2.0 * prec * recall / (prec + recall)
            if f1 > max_f1:
                max_f1 = f1
                thresh = arr[n, 0]

    return (max_f1, thresh)
