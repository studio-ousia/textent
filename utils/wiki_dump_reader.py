# -*- coding: utf-8 -*-

import bz2
import logging
import re
from gensim.corpora import wikicorpus

logger = logging.getLogger(__name__)

DEFAULT_IGNORED_NS = ('wikipedia:', 'file:', 'portal:', 'template:', 'mediawiki:',
                      'user:', 'help:', 'book:', 'draft:')

REDIRECT_REGEXP = re.compile(
    ur"(?:\#|＃)(?:REDIRECT|転送)[:\s]*(?:\[\[(.*)\]\]|(.*))", re.IGNORECASE
)


class WikiPage(object):
    def __init__(self, title, wiki_text):
        self._title = title
        self._wiki_text = wiki_text

    def __repr__(self):
        return '<WikiPage %s>' % (self.title.encode('utf-8'))

    def __reduce__(self):
        return (self.__class__, (self.title, self.wiki_text))

    @property
    def title(self):
        return self._title

    @property
    def wiki_text(self):
        return self._wiki_text

    @property
    def is_redirect(self):
        return bool(self.redirect)

    @property
    def redirect(self):
        red_match_obj = REDIRECT_REGEXP.match(self.wiki_text)
        if not red_match_obj:
            return None

        if red_match_obj.group(1):
            dest = red_match_obj.group(1)
        else:
            dest = red_match_obj.group(2)

        if not dest:
            return None

        return self._normalize_title(dest)

    def _normalize_title(self, title):
        return (title[0].upper() + title[1:]).replace(u'_', u' ')


class WikiDumpReader(object):
    def __init__(self, dump_file, ignored_ns=DEFAULT_IGNORED_NS):
        self._dump_file = dump_file
        self._ignored_ns = ignored_ns

    @property
    def dump_file(self):
        return self._dump_file

    def __iter__(self):
        with bz2.BZ2File(self._dump_file) as f:
            c = 0
            for (title, wiki_text, wiki_id) in wikicorpus.extract_pages(f):
                if any([title.lower().startswith(ns) for ns in self._ignored_ns]):
                    continue
                c += 1

                yield WikiPage(unicode(title), unicode(wiki_text))

                if c % 10000 == 0:
                    logger.info('Processed: %d', c)
