# -*- coding: utf-8 -*-

import click
import gensim
import logging
import os
import random
import torch
import unicodedata
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict, Counter
from marisa_trie import Trie
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from tempfile import NamedTemporaryFile
from torch.autograd import Variable

from model import Encoder
from train import load_model
from utils.tokenizer import RegexpTokenizer
from utils.vocab import WordVocab, EntityVocab

logger = logging.getLogger(__name__)


class TextClassifier(Encoder):
    def init(self, word_embedding, entity_embedding, num_classes, dropout_prob):
        self._dropout_prob = dropout_prob

        self._word_embedding = nn.Embedding(word_embedding.shape[0], word_embedding.shape[1],
                                            padding_idx=0)
        self._word_embedding.weight = nn.Parameter(torch.FloatTensor(word_embedding))

        self._entity_embedding = nn.Embedding(entity_embedding.shape[0], entity_embedding.shape[1],
                                              padding_idx=0)
        self._entity_embedding.weight = nn.Parameter(torch.FloatTensor(entity_embedding))

        self._out_layer = nn.Linear(self.dim_size, num_classes, bias=True)

    def forward(self, arg):
        (word_indices, entity_indices) = arg

        if self._dropout_prob != 0.0 and self.training:
            mask = word_indices.clone().float()
            mask.data.fill_(1.0 - self._dropout_prob)
            word_indices = torch.mul(word_indices, torch.bernoulli(mask).long())

        if self._dropout_prob != 0.0 and self.training:
            mask = entity_indices.clone().float()
            mask.data.fill_(1.0 - self._dropout_prob)
            entity_indices = torch.mul(entity_indices, torch.bernoulli(mask).long())

        feature_vector = self.compute_feature_vector(word_indices, entity_indices)
        if not self._word_only and not self._entity_only:
            feature_vector = self._output_layer(feature_vector)

        return self._out_layer(feature_vector)


def evaluate(model_file, entity_linker, target_dataset, dataset_path, batch_size,
             epoch, patience, optimizer, learning_rate, dev_size, min_word_count,
             min_entity_count, max_text_len, max_entity_len, seed, temp_dir, **model_kwargs):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if target_dataset == '20ng':
        dataset = _read_20ng_dataset(dev_size)
    else:
        dataset = _read_r8_dataset(dataset_path, dev_size)

    model = load_model(model_file, model_cls=TextClassifier)

    word_counter = Counter([w.lower() for f in ('train', 'dev', 'test')
                            for (_, wl, _) in dataset[f] for w in wl])
    word_vocab = WordVocab(
        Trie([w for (w, c) in word_counter.iteritems() if c >= min_word_count]),
        lowercase=True, start_index=1
    )

    word_emb = np.random.uniform(low=-0.05, high=0.05, size=(word_vocab.size, model.dim_size))
    word_emb = np.vstack([np.zeros(model.dim_size), word_emb]).astype('float32')
    model_word_emb = model.word_embedding
    for word in word_vocab:
        ind = model.word_vocab.get_index(word)
        if ind is not None:
            word_emb[word_vocab.get_index(word)] = model_word_emb[ind]

    entity_counter = Counter([m.title for f in ('train', 'dev', 'test')
                              for (t, _, _) in dataset[f] for m in entity_linker.detect_mentions(t)])
    entity_vocab = EntityVocab(
        Trie([t for (t, c) in entity_counter.iteritems() if c >= min_entity_count]),
        start_index=1
    )

    entity_emb = np.random.uniform(low=-0.05, high=0.05, size=(entity_vocab.size, model.dim_size))
    entity_emb = np.vstack([np.zeros(model.dim_size), entity_emb]).astype('float32')
    model_entity_emb = model.entity_embedding
    for entity in entity_vocab:
        ind = model.entity_vocab.get_index(entity)
        if ind is not None:
            entity_emb[entity_vocab.get_index(entity)] = model_entity_emb[ind]

    model.init(word_emb, entity_emb, len(dataset['target_names']), **model_kwargs)
    model = model.cuda()

    def generate_batches(fold):
        word_batch = []
        entity_batch = []
        labels = []

        random.shuffle(dataset[fold])
        for (n, (text, words, label)) in enumerate(dataset[fold]):
            word_indices = []
            for word in words:
                word_index = word_vocab.get_index(word)
                if word_index is not None:
                    word_indices.append(word_index)

            word_batch.append(word_indices)

            entity_indices = []
            for mention in entity_linker.detect_mentions(text):
                entity_index = entity_vocab.get_index(mention.title)
                if entity_index is not None:
                    entity_indices.append(entity_index)

            entity_batch.append(entity_indices)

            labels.append(label)

            if len(word_batch) == batch_size or n == len(dataset[fold]) - 1:
                text_len = max(min(max(len(b) for b in word_batch), max_text_len), 1)
                entity_len = max(min(max(len(b) for b in entity_batch), max_entity_len), 1)

                word_arr = np.zeros((len(word_batch), text_len), dtype=np.int)
                entity_arr = np.zeros((len(entity_batch), entity_len), dtype=np.int)

                for (n, (word_indices, entity_indices)) in enumerate(zip(word_batch, entity_batch)):
                    word_indices = word_indices[:text_len]
                    word_arr[n][:len(word_indices)] = word_indices

                    entity_indices = entity_indices[:entity_len]
                    entity_arr[n][:len(entity_indices)] = entity_indices

                yield ((Variable(torch.LongTensor(word_arr)),
                        Variable(torch.LongTensor(entity_arr))),
                       Variable(torch.LongTensor(labels)))

                word_batch = []
                entity_batch = []
                labels = []

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer_args = {}
    if learning_rate is not None:
        optimizer_args['lr'] = learning_rate

    optimizer_ins = getattr(optim, optimizer)(parameters, **optimizer_args)

    n_correct = 0
    n_total = 0
    cur_correct = 0
    cur_total = 0
    cur_loss = 0.0
    dev_scores = [0.0]

    batch_idx = 0

    with NamedTemporaryFile(dir=temp_dir) as f:
        for n_epoch in range(epoch):
            logger.info('Epoch: %d', n_epoch)

            for (batch_idx, (args, target)) in enumerate(generate_batches('train'), batch_idx):
                args = tuple([o.cuda() for o in args])
                target = target.cuda()

                optimizer_ins.zero_grad()
                output = model(args)
                loss = F.cross_entropy(output, target)
                loss.backward()

                optimizer_ins.step()

                cur_correct += (torch.max(output, 1)[1].view(target.size()).data == target.data).sum()
                cur_total += len(target)
                cur_loss += loss.data
                if batch_idx != 0 and batch_idx % 50 == 0:
                    n_correct += cur_correct
                    n_total += cur_total
                    logger.info('Processed %d batches (epoch: %d, loss: %.4f acc: %.4f total acc: %.4f)' % (
                        batch_idx, n_epoch, cur_loss[0] / cur_total, 100. * cur_correct / cur_total, 100. * n_correct / n_total
                    ))
                    cur_correct = 0
                    cur_total = 0
                    cur_loss = 0.0

            model.eval()

            dev_correct = 0
            dev_total = 0
            for (args, target) in generate_batches('dev'):
                args = tuple([o.cuda() for o in args])
                target = target.cuda()

                output = model(args)
                dev_correct += (torch.max(output, 1)[1].view(target.size()).data == target.data).sum()
                dev_total += len(target)

            dev_score = float(dev_correct) / dev_total
            if dev_score > max(dev_scores):
                torch.save(model.state_dict(), f.name)

            dev_scores.append(dev_score)

            logger.info('dev score: %.4f (%d/%d) (max: %.4f)', dev_score, dev_correct, dev_total,
                        max(dev_scores))

            model.train()

            if patience is not None:
                if len(dev_scores) - dev_scores.index(max(dev_scores)) > patience:
                    break

        model.load_state_dict(torch.load(f.name))

    model.eval()

    results = []
    targets = []
    for (args, target) in generate_batches('test'):
        args = tuple([o.cuda() for o in args])
        target = target.cuda()

        output = model(args)
        results.append(torch.max(output, 1)[1].view(target.size()).data.cpu().numpy())
        targets.append(target.data.cpu().numpy())

    results = np.hstack(results)
    targets = np.hstack(targets)

    test_score = accuracy_score(targets, results)

    click.echo('accuracy: %.4f' % test_score)
    click.echo('f-measure: %.4f' % f1_score(targets, results, average='macro'))

    click.echo('')
    click.echo('class-level results:')
    prf_scores = precision_recall_fscore_support(
        targets, results, average=None, labels=range(len(dataset['target_names']))
    )
    for (i, name) in enumerate(dataset['target_names']):
        click.echo('label: %s precision: %.4f recall: %.4f f-measure: %.4f' % (
            name, prf_scores[0][i], prf_scores[1][i], prf_scores[2][i]
        ))

    return (max(dev_scores), test_score)


def _read_20ng_dataset(dev_size=0.1):
    tokenizer = RegexpTokenizer()

    ret = dict(train=[], dev=[], test=[], target_names=fetch_20newsgroups()['target_names'])

    for fold in ('train', 'test'):
        dataset_obj = fetch_20newsgroups(subset=fold, shuffle=False)

        for (text, label) in zip(dataset_obj['data'], dataset_obj['target']):
            text = gensim.utils.deaccent(unicodedata.normalize('NFKD', text))
            words = [t.text for t in tokenizer.tokenize(text)]
            ret[fold].append((text, words, label))

    dev_size = int(len(ret['train']) * dev_size)
    random.shuffle(ret['train'])
    ret['dev'] = ret['train'][-dev_size:]
    ret['train'] = ret['train'][:-dev_size]

    logger.info('train size: %d', len(ret['train']))
    logger.info('dev size: %d', len(ret['dev']))
    logger.info('test size: %d', len(ret['test']))

    return ret


def _read_r8_dataset(dataset_path, dev_size=0.1):
    tokenizer = RegexpTokenizer()

    target_names = set()
    data = defaultdict(list)

    for (fold, fname) in (('train', os.path.join(dataset_path, 'r8-train-all-terms.txt')),
                          ('test', os.path.join(dataset_path, 'r8-test-all-terms.txt'))):
        with open(fname) as f:
            for line in f:
                (target, text) = line.decode('utf-8').rstrip().split('\t')
                target_names.add(target)
                data[fold].append((text, target))

    target_names = list(target_names)
    ret = dict(train=[], dev=[], test=[], target_names=target_names)
    target_index = {t: i for (i, t) in enumerate(target_names)}

    for (fold, items) in data.iteritems():
        for (text, target) in items:
            words = [t.text for t in tokenizer.tokenize(text)]
            ret[fold].append((text, words, target_index[target]))

    dev_size = int(len(ret['train']) * dev_size)
    random.shuffle(ret['train'])
    ret['dev'] = ret['train'][-dev_size:]
    ret['train'] = ret['train'][:-dev_size]

    logger.info('train size: %d', len(ret['train']))
    logger.info('dev size: %d', len(ret['dev']))
    logger.info('test size: %d', len(ret['test']))

    return ret
