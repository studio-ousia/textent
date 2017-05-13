# -*- coding: utf-8 -*-

import bz2
import logging
import numpy as np
from contextlib import closing
from gensim.models.word2vec import Word2Vec, LineSentence
from marisa_trie import Trie
from multiprocessing.pool import Pool

from embedding_reader import EmbeddingReader
from vocab import WordVocab, EntityVocab
from wiki_extractor import WikiExtractor
from wiki_dump_reader import WikiDumpReader

MARKER = u'ENTITY/'

logger = logging.getLogger(__name__)
_extractor = None


def generate_corpus(dump_file, entity_db, out_file, pool_size, chunk_size):
    dump_reader = WikiDumpReader(dump_file)

    global _extractor
    _extractor = WikiExtractor()

    with bz2.BZ2File(out_file, mode='w') as f:
        logger.info('Starting to process Wikipedia dump...')
        with closing(Pool(pool_size)) as pool:
            for paragraphs in pool.imap_unordered(
                _process_page, dump_reader, chunksize=chunk_size
            ):
                for paragraph in paragraphs:
                    para_text = u''
                    cur = 0
                    for link in sorted(paragraph.wiki_links, key=lambda l: l.span[0]):
                        if link.title.startswith('File:'):
                            continue

                        title = entity_db.resolve_redirect(link.title).replace(u' ', u'_')
                        para_text += paragraph.text[cur:link.span[0]].lower()
                        para_text += u' ' + MARKER + title + u' '
                        cur = link.span[1]

                    para_text += paragraph.text[cur:].lower()

                    f.write(para_text.encode('utf-8') + '\n')


def train(corpus_file, mode, dim_size, window, min_count, negative, epoch, workers):
    with bz2.BZ2File(corpus_file) as f:
        sentences = LineSentence(f)
        sg = int(mode == 'sg')

        model = Word2Vec(sentences, size=dim_size, window=window, min_count=min_count,
                         workers=workers, iter=epoch, negative=negative, sg=sg)

    words = []
    entities = []
    for (w, _) in model.vocab.iteritems():
        if w.startswith(MARKER):
            entities.append(w[len(MARKER):].replace(u'_', u' '))
        else:
            words.append(w)

    word_vocab = WordVocab(Trie(words), lowercase=True)
    entity_vocab = EntityVocab(Trie(entities))

    word_embedding = np.zeros((len(words), dim_size), dtype=np.float32)
    entity_embedding = np.zeros((len(entities), dim_size), dtype=np.float32)
    for word in words:
        ind = word_vocab.get_index(word)
        if ind is not None:
            word_embedding[ind] = model[word]

    for entity in entities:
        entity_embedding[entity_vocab.get_index(entity)] = model[MARKER + entity.replace(u' ', u'_')]

    return EmbeddingReader(word_embedding, entity_embedding, word_vocab, entity_vocab)


def _process_page(page):
    return _extractor.extract_paragraphs(page)
