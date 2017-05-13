# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, word_vocab, entity_vocab, target_entity_vocab, dim_size,
                 word_dropout_prob, entity_dropout_prob, word_only, entity_only,
                 entity_normalize, float16=True, word_embedding=None, entity_embedding=None,
                 target_entity_embedding=None, **kwargs):
        super(Encoder, self).__init__()

        self._word_vocab = word_vocab
        self._entity_vocab = entity_vocab
        self._target_entity_vocab = target_entity_vocab

        self._word_dropout_prob = word_dropout_prob
        self._entity_dropout_prob = entity_dropout_prob
        self._word_only = word_only
        self._entity_only = entity_only
        self._dim_size = dim_size

        if not word_only and not entity_only:
            self._feature_size = dim_size * 2
            self._output_layer = nn.Linear(self._feature_size, dim_size, bias=False)
            self._output_layer.weight = nn.Parameter(torch.cat((torch.eye(dim_size),
                                                     torch.eye(dim_size)), 1))

        else:
            self._feature_size = dim_size

        self._word_embedding = nn.Embedding(len(word_vocab) + 1, dim_size, padding_idx=0)
        if word_embedding is not None:
            word_emb = torch.FloatTensor(word_embedding)
            if float16:
                word_emb = word_emb.half()
            self._word_embedding.weight = nn.Parameter(word_emb.cuda())

        self._entity_embedding = nn.Embedding(len(entity_vocab) + 1, dim_size, padding_idx=0)
        if entity_embedding is not None:
            entity_emb = torch.FloatTensor(entity_embedding)
            if float16:
                entity_emb = entity_emb.half()
            self._entity_embedding.weight = nn.Parameter(entity_emb.cuda())

        self._target_entity_embedding = nn.Embedding(len(target_entity_vocab) + 1,
                                                     dim_size, padding_idx=0)
        if target_entity_embedding is not None:
            if entity_normalize:
                norms = np.linalg.norm(target_entity_embedding[1:], 2, 1).repeat(dim_size).reshape(-1, dim_size)
                target_entity_embedding[1:] = target_entity_embedding[1:] / norms

            emb = torch.FloatTensor(target_entity_embedding)
            if float16:
                emb = emb.half()
            self._target_entity_embedding.weight = nn.Parameter(emb.cuda())

    @property
    def dim_size(self):
        return self._dim_size

    @property
    def feature_size(self):
        return self._feature_size

    @property
    def word_vocab(self):
        return self._word_vocab

    @property
    def entity_vocab(self):
        return self._entity_vocab

    @property
    def target_entity_vocab(self):
        return self._target_entity_vocab

    @property
    def word_embedding(self, numpy=True):
        return self._word_embedding.weight.data.cpu().numpy()

    @property
    def entity_embedding(self):
        return self._entity_embedding.weight.data.cpu().numpy()

    @property
    def target_entity_embedding(self):
        return self._target_entity_embedding.weight.data.cpu().numpy()

    def get_word_vector(self, word):
        index = self._word_vocab.get_index(word, default=0)
        return self._word_embedding.weight.data[index].cpu().float().numpy()

    def get_entity_vector(self, title):
        index = self._target_entity_vocab.get_index(title, default=0)
        return self._target_entity_embedding.weight.data[index].cpu().float().numpy()

    def forward(self, (word_indices, entity_indices, target_entity_indices)):
        target_entity_size = target_entity_indices.size()[1]

        if self._word_dropout_prob != 0.0 and self.training:
            mask = word_indices.clone().float()
            mask.data.fill_(1.0 - self._word_dropout_prob)
            word_indices = torch.mul(word_indices, torch.bernoulli(mask).long())

        if self._entity_dropout_prob != 0.0 and self.training:
            mask = entity_indices.clone().float()
            mask.data.fill_(1.0 - self._entity_dropout_prob)
            entity_indices = torch.mul(entity_indices, torch.bernoulli(mask).long())

        feature_vector = self.compute_feature_vector(word_indices, entity_indices)
        if not self._word_only and not self._entity_only:
            feature_vector = self._output_layer(feature_vector)

        feature_matrix = feature_vector.unsqueeze(1).repeat(1, target_entity_size, 1)
        target_entity_emb = self._target_entity_embedding(target_entity_indices).float()

        return torch.sum(torch.mul(feature_matrix, target_entity_emb), 2).squeeze(2)

    def compute_feature_vector(self, word_indices, entity_indices):
        if self._word_only:
            ret = self.compute_word_vector(word_indices)

        elif self._entity_only:
            ret = self.compute_entity_vector(entity_indices)

        else:
            ret = torch.cat((self.compute_word_vector(word_indices),
                             self.compute_entity_vector(entity_indices)), 1)

        return ret

    def compute_word_vector(self, word_indices):
        word_emb = self._word_embedding(word_indices).float()
        return torch.mean(word_emb, 1).squeeze(1)

    def compute_entity_vector(self, entity_indices):
        entity_emb = self._entity_embedding(entity_indices).float()
        return torch.mean(entity_emb, 1).squeeze(1)
