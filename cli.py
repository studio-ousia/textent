# -*- coding: utf-8 -*-

import click
import joblib
import logging

import train
import entity_typing
import text_classification
from utils import word2vec
from utils.description_db import DescriptionDB
from utils.embedding_reader import EmbeddingReader
from utils.entity_db import EntityDB
from utils.entity_linker import TagmeEntityLinker
from utils.vocab import WordVocab, EntityVocab

logger = logging.getLogger(__name__)


@click.group()
def cli():
    LOG_FORMAT = '[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)'
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


@cli.command()
@click.argument('dump_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.Path())
@click.option('--pool-size', default=20)
@click.option('--chunk-size', default=30)
def build_entity_db(dump_file, out_file, **kwargs):
    db = EntityDB.build(dump_file, **kwargs)
    db.save(out_file)


@cli.command()
@click.argument('nif_context_file', type=click.Path(exists=True))
@click.argument('nif_text_links_file', type=click.Path(exists=True))
@click.argument('entity_db_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.Path())
@click.option('--deaccent', is_flag=True)
def build_description_db(nif_context_file, nif_text_links_file, entity_db_file, out_file, **kwargs):
    entity_db = EntityDB.load(entity_db_file)
    DescriptionDB.build(nif_context_file, nif_text_links_file, entity_db, out_file, **kwargs)


@cli.command()
@click.argument('description_db_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.Path())
@click.option('-t', '--target-vocab', type=click.Path(exists=True))
@click.option('--min-count', default=10)
@click.option('--lowercase/--no-lowercase', default=True)
def build_word_vocab(description_db_file, out_file, target_vocab, **kwargs):
    description_db = DescriptionDB(description_db_file)

    if target_vocab:
        target_vocab = EntityVocab.load(target_vocab)

    word_vocab = WordVocab.build(description_db, start_index=1, target_vocab=target_vocab,
                                 **kwargs)
    word_vocab.save(out_file)


@cli.command()
@click.argument('description_db_file', type=click.Path(exists=True))
@click.argument('entity_db_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.Path())
@click.option('-t', '--target-vocab', type=click.Path(exists=True))
@click.option('--white-list', type=click.File(), default=[], multiple=True)
@click.option('--min-inlink-count', default=10)
def build_entity_vocab(description_db_file, entity_db_file, out_file, target_vocab,
                       white_list, **kwargs):
    description_db = DescriptionDB(description_db_file)

    if target_vocab:
        target_vocab = EntityVocab.load(target_vocab)

    entity_db = EntityDB.load(entity_db_file)
    white_titles = []
    for f in white_list:
        white_titles += [l.rstrip().decode('utf-8') for l in f]

    entity_vocab = EntityVocab.build(description_db, entity_db, white_titles,
                                     start_index=1, target_vocab=target_vocab, **kwargs)
    entity_vocab.save(out_file)


@cli.command()
@click.argument('in_file', type=click.File())
@click.argument('mapping_file', type=click.File())
@click.argument('out_file', type=click.Path())
def load_figment_embedding(in_file, mapping_file, out_file):
    embedding = EmbeddingReader.load_figment(in_file, mapping_file)
    embedding.save(out_file)


@cli.command()
@click.argument('model_file', type=click.File())
@click.argument('out_file', type=click.Path())
def load_wikipedia2vec(model_file, out_file):
    embedding = EmbeddingReader.load_wikipedia2vec(model_file)
    embedding.save(out_file)


@cli.command()
@click.argument('dump_file', type=click.Path(exists=True))
@click.argument('entity_db_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.Path())
@click.option('--pool-size', default=20)
@click.option('--chunk-size', default=30)
def generate_word2vec_corpus(dump_file, entity_db_file, out_file, **kwargs):
    entity_db = EntityDB.load(entity_db_file)
    word2vec.generate_corpus(dump_file, entity_db, out_file, **kwargs)


@cli.command()
@click.argument('corpus_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.Path())
@click.option('--mode', type=click.Choice(['sg', 'cbow']), default='sg')
@click.option('--dim-size', default=300)
@click.option('--window', default=10)
@click.option('--min-count', default=5)
@click.option('--negative', default=15)
@click.option('--epoch', default=5)
@click.option('--workers', default=30)
def train_word2vec(corpus_file, out_file, **kwargs):
    embedding = word2vec.train(corpus_file, **kwargs)
    embedding.save(out_file)


@cli.command()
@click.argument('description_db_file', type=click.Path(exists=True))
@click.argument('entity_db_file', type=click.Path(exists=True))
@click.argument('word_vocab_file', type=click.Path(exists=True))
@click.argument('entity_vocab_file', type=click.Path(exists=True))
@click.argument('target_entity_vocab_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.Path())
@click.option('--embedding', type=click.Path(exists=True), multiple=True, default=[])
@click.option('--dim-size', default=300)
@click.option('--batch-size', default=100)
@click.option('--negative', default=100)
@click.option('--epoch', default=300)
@click.option('--optimizer', default='Adadelta')
@click.option('--max-text-len', default=1000)
@click.option('--max-entity-len', default=300)
@click.option('--pool-size', default=3)
@click.option('--seed', default=0)
@click.option('--word-dropout-prob', default=0.5)
@click.option('--entity-dropout-prob', default=0.5)
@click.option('--word-only', is_flag=True)
@click.option('--entity-only', is_flag=True)
@click.option('--entity-normalize/--no-entity-normalize', default=True)
@click.option('--float16', is_flag=True)
@click.option('-s', '--save', type=float, multiple=True, default=[])
def train_model(description_db_file, entity_db_file, word_vocab_file, entity_vocab_file,
                target_entity_vocab_file, out_file, embedding, **kwargs):
    description_db = DescriptionDB(description_db_file)
    entity_db = EntityDB.load(entity_db_file)

    word_vocab = WordVocab.load(word_vocab_file)
    entity_vocab = EntityVocab.load(entity_vocab_file)
    target_entity_vocab = EntityVocab.load(target_entity_vocab_file)

    embeddings = [EmbeddingReader.load(f) for f in embedding]

    train.train(description_db, entity_db, word_vocab, entity_vocab, target_entity_vocab,
                out_file, embeddings, **kwargs)


@cli.command()
@click.argument('model_file', type=click.Path())
@click.argument('entity_db_file', type=click.Path(exists=True))
@click.option('--dataset-dir', type=click.Path(exists=True, file_okay=False), default='dataset/entity_typing')
@click.option('--embedding', is_flag=True)
@click.option('--batch-size', default=32)
@click.option('--epoch', default=500)
@click.option('--patience', default=5)
@click.option('--num-hidden-units', default=200)
@click.option('--exclude-oov', is_flag=True)
@click.option('--temp-dir', type=click.Path(exists=True, file_okay=False), default='/dev/shm')
@click.option('--seed', default=0)
def evaluate_entity_typing(model_file, entity_db_file, dataset_dir, embedding, **kwargs):
    entity_db = EntityDB.load(entity_db_file)

    if embedding:
        model = EmbeddingReader.load(model_file)
        entity_embedding = model.entity_embedding
        entity_vocab = model.entity_vocab
    else:
        model = train.load_model(model_file)
        entity_embedding = model.target_entity_embedding
        entity_vocab = model.target_entity_vocab

    entity_typing.evaluate(entity_embedding, entity_vocab, entity_db, dataset_dir, **kwargs)


@cli.command()
@click.argument('model_file', type=click.Path())
@click.argument('entity_db_file', type=click.Path(exists=True))
@click.option('-t', '--target-dataset', type=click.Choice(['20ng', 'r8']), default='20ng')
@click.option('--dataset-path', type=click.Path(exists=True, file_okay=False), default='dataset/text_classification')
@click.option('--min-link-prob', default=0.00)
@click.option('--min-disambi-score', default=0.05)
@click.option('--tagme-cache', type=click.Path(), default=None)
@click.option('--batch-size', default=32)
@click.option('--epoch', default=100)
@click.option('--patience', default=5)
@click.option('--optimizer', default='Adam', type=click.Choice(['SGD', 'Adagrad', 'RMSprop', 'Adam', 'Adadelta', 'Adamax']))
@click.option('--learning-rate', default=0.001, type=float)
@click.option('--dev-size', default=0.1)
@click.option('--min-word-count', default=5)
@click.option('--min-entity-count', default=5)
@click.option('--max-text-len', default=1000)
@click.option('--max-entity-len', default=500)
@click.option('--temp-dir', type=click.Path(exists=True, file_okay=False), default='/dev/shm')
@click.option('--seed', default=0)
@click.option('--dropout-prob', default=0.5)
def evaluate_text_classification(model_file, entity_db_file, min_link_prob, min_disambi_score,
                                 tagme_cache, **kwargs):
    entity_db = EntityDB.load(entity_db_file)
    if tagme_cache:
        tagme_cache = joblib.load(tagme_cache)

    entity_linker = TagmeEntityLinker(entity_db, min_link_prob, min_disambi_score, tagme_cache)

    text_classification.evaluate(model_file, entity_linker, **kwargs)


if __name__ == '__main__':
    cli()
