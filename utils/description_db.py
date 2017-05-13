# -*- coding: utf-8 -*-

import bz2
import gensim
import lmdb
import logging
import zlib
import cPickle as pickle
from collections import defaultdict
from contextlib import closing
from rdflib import Graph

logger = logging.getLogger(__name__)

MAP_SIZE = 100000000000  # 100GB


class DescriptionDB(object):
    def __init__(self, db_file):
        self._db = lmdb.open(db_file, readonly=True, subdir=False, lock=False,
                             map_size=MAP_SIZE)

    def __getitem__(self, key):
        return self.get(key)

    def __iter__(self):
        for obj in self.iterator():
            yield obj

    def iterator(self):
        with self._db.begin() as txn:
            cur = txn.cursor()
            for (key, val) in iter(cur):
                (text, links) = pickle.loads(zlib.decompress(val))
                yield (key.decode('utf-8'), text, links)

    def get(self, key):
        with self._db.begin() as txn:
            val = txn.get(key.encode('utf-8'))
            if not val:
                raise KeyError(key)

        return pickle.loads(zlib.decompress(val))

    def keys(self):
        with self._db.begin() as txn:
            cur = txn.cursor()
            return [k.decode('utf-8') for k in cur.iternext(values=False)]

    @staticmethod
    def build(nif_context_file, nif_text_links_file, entity_db, out_file, deaccent, chunk_size=1000):
        buf = []

        def chunked_ttl_reader(f):
            lines = []
            for line in f:
                lines.append(line.decode('utf-8').rstrip())
                if len(lines) == chunk_size:
                    g = Graph()
                    g.parse(data=u'\n'.join(lines), format='n3')
                    for triple in g:
                        yield triple
                    lines = []

            if lines:
                g = Graph()
                g.parse(data=u'\n'.join(lines), format='n3')
                for triple in g:
                    yield triple

        links_arr = []
        link_refs = {}

        with bz2.BZ2File(nif_text_links_file) as f:
            for (n, (s, p, o)) in enumerate(chunked_ttl_reader(f)):
                if n != 0 and n % 100000 == 0:
                    logger.info('Processed %d lines', n)

                s = unicode(s)
                p = unicode(p)
                o = unicode(o)

                if p == 'http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#referenceContext':
                    link_refs[s] = o

                elif p == 'http://www.w3.org/2005/11/its/rdf#taIdentRef':
                    if not o.startswith('http://dbpedia.org/resource/'):
                        continue

                    title = o[len('http://dbpedia.org/resource/'):].replace('_', ' ')
                    title = entity_db.resolve_redirect(title)
                    links_arr.append((s, title))

        links = defaultdict(list)
        for (key, title) in links_arr:
            ref_key = link_refs[key]
            links[ref_key].append(title)

        del link_refs
        del links_arr

        titles = {}
        texts = {}

        with bz2.BZ2File(nif_context_file) as f:
            for (n, (s, p, o)) in enumerate(chunked_ttl_reader(f)):
                if n != 0 and n % 100000 == 0:
                    logger.info('Processed %d lines', n)

                s = unicode(s)
                p = unicode(p)
                o = unicode(o)
                if p == 'http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#isString':
                    if deaccent:
                        texts[s] = gensim.utils.deaccent(o)
                    else:
                        texts[s] = o
                elif p == 'http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#sourceUrl':
                    title = o[len('http://en.wikipedia.org/wiki/'):].replace(u'_', u' ')
                    titles[s] = entity_db.resolve_redirect(title)

        with closing(lmdb.open(out_file, subdir=False, map_async=True, map_size=MAP_SIZE)) as db:
            buf = []
            for (key, title) in titles.iteritems():
                if key not in texts:
                    continue

                buf.append((title.encode('utf-8'),
                            zlib.compress(pickle.dumps((texts[key], links.get(key, []))))))

                if len(buf) == chunk_size:
                    with db.begin(write=True) as txn:
                        cur = txn.cursor()
                        cur.putmulti(buf)
                    buf = []

            if buf:
                with db.begin(write=True) as txn:
                    cur = txn.cursor()
                    cur.putmulti(buf)
