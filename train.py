# -*- coding: utf-8 -*-

import joblib
import logging
import numpy as np
import random
import re
import torch
import torch.optim as optim
import torch.nn.functional as F

import entity_typing
from generator import generate_data
from model import Encoder
from utils.vocab import WordVocab, EntityVocab

logger = logging.getLogger(__name__)


def train(description_db, entity_db, word_vocab, entity_vocab, target_entity_vocab,
          out_file, embeddings, dim_size, batch_size, negative, epoch, optimizer, max_text_len,
          max_entity_len, pool_size, seed, save, **model_params):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    word_matrix = np.random.uniform(low=-0.05, high=0.05, size=(word_vocab.size, dim_size))
    word_matrix = np.vstack([np.zeros(dim_size), word_matrix]).astype('float32')

    entity_matrix = np.random.uniform(low=-0.05, high=0.05, size=(entity_vocab.size, dim_size))
    entity_matrix = np.vstack([np.zeros(dim_size), entity_matrix]).astype('float32')

    target_entity_matrix = np.random.uniform(low=-0.05, high=0.05, size=(target_entity_vocab.size, dim_size))
    target_entity_matrix = np.vstack([np.zeros(dim_size), target_entity_matrix]).astype('float32')

    for embedding in embeddings:
        for word in word_vocab:
            vec = embedding.get_word_vector(word)
            if vec is not None:
                word_matrix[word_vocab.get_index(word)] = vec

        for title in entity_vocab:
            vec = embedding.get_entity_vector(title)
            if vec is not None:
                entity_matrix[entity_vocab.get_index(title)] = vec

        for title in target_entity_vocab:
            vec = embedding.get_entity_vector(title)
            if vec is not None:
                target_entity_matrix[target_entity_vocab.get_index(title)] = vec

    entity_negatives = np.arange(1, target_entity_matrix.shape[0])

    model_params.update(dict(dim_size=dim_size))
    model = Encoder(word_embedding=word_matrix, entity_embedding=entity_matrix,
                    target_entity_embedding=target_entity_matrix, word_vocab=word_vocab,
                    entity_vocab=entity_vocab, target_entity_vocab=target_entity_vocab,
                    **model_params)

    del word_matrix
    del entity_matrix
    del target_entity_matrix

    model = model.cuda()

    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer_ins = getattr(optim, optimizer)(parameters)

    n_correct = 0
    n_total = 0
    cur_correct = 0
    cur_total = 0
    cur_loss = 0.0

    batch_idx = 0

    joblib.dump(dict(model_params=model_params,
                     word_vocab=word_vocab.serialize(),
                     entity_vocab=entity_vocab.serialize(),
                     target_entity_vocab=target_entity_vocab.serialize()),
                out_file + '.joblib')

    if not save or 0 in save:
        state_dict = model.state_dict()
        torch.save(state_dict, out_file + '_epoch0.bin')

    for n_epoch in range(1, epoch + 1):
        logger.info('Epoch: %d', n_epoch)

        if n_epoch % 5 == 0:
            entity_typing.evaluate(
                model._target_entity_embedding.weight.data.cpu().float().numpy(),
                model._target_entity_vocab, entity_db
            )

        for (batch_idx, (args, target)) in enumerate(generate_data(
            description_db, word_vocab, entity_vocab, target_entity_vocab, entity_negatives,
            batch_size, negative, max_text_len, max_entity_len, pool_size
        ), batch_idx):
            args = tuple([o.cuda(async=True) for o in args])
            target = target.cuda()

            optimizer_ins.zero_grad()
            output = model(args)
            loss = F.cross_entropy(output, target)
            loss.backward()

            optimizer_ins.step()

            cur_correct += (torch.max(output, 1)[1].view(target.size()).data == target.data).sum()
            cur_total += len(target)
            cur_loss += loss.data
            if batch_idx != 0 and batch_idx % 1000 == 0:
                n_correct += cur_correct
                n_total += cur_total
                logger.info('Processed %d batches (epoch: %d, loss: %.4f acc: %.4f total acc: %.4f)' % (
                    batch_idx, n_epoch, cur_loss[0] / cur_total, 100. * cur_correct / cur_total, 100. * n_correct / n_total
                ))
                cur_correct = 0
                cur_total = 0
                cur_loss = 0.0

        # if n_epoch in save:
        if (not save and n_epoch % 10 == 0) or n_epoch in save:
            state_dict = model.state_dict()
            torch.save(state_dict, out_file + '_epoch%d.bin' % n_epoch)


def load_model(model_file, model_cls=Encoder):
    meta = joblib.load(re.sub(r'_epoch\d+$', '', model_file) + '.joblib')
    model_params = meta['model_params']

    word_vocab = WordVocab.load(meta['word_vocab'])
    entity_vocab = EntityVocab.load(meta['entity_vocab'])
    target_entity_vocab = EntityVocab.load(meta['target_entity_vocab'])

    state_dict = torch.load(model_file + '.bin')
    state_dict = {k[7:] if k.startswith('module.') else k: v for (k, v) in state_dict.items()}

    model = model_cls(word_vocab, entity_vocab, target_entity_vocab, **model_params)
    model.load_state_dict(state_dict)

    return model
