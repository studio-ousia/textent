# -*- coding: utf-8 -*-

import re
import mwparserfromhell

SPACE_REGEXP = re.compile(ur'^\s*$')


class Paragraph(object):
    __slots__ = ('_text', '_wiki_links', '_abstract')

    def __init__(self, text, wiki_links, abstract):
        self._text = text
        self._wiki_links = wiki_links
        self._abstract = abstract

    @property
    def text(self):
        return self._text

    @property
    def wiki_links(self):
        return self._wiki_links

    @property
    def abstract(self):
        return self._abstract

    def __repr__(self):
        return '<Paragraph %s>' % (self._text[:20].encode('utf-8') + '...')

    def __reduce__(self):
        return (self.__class__, (self._text, self._wiki_links, self._abstract))


class WikiLink(object):
    __slots__ = ('_title', '_text', '_span', '_abstract')

    def __init__(self, title, text, span, abstract):
        self._title = title
        self._text = text
        self._span = span
        self._abstract = abstract

    @property
    def title(self):
        return self._title

    @property
    def text(self):
        return self._text

    @property
    def span(self):
        return self._span

    @property
    def abstract(self):
        return self._abstract

    def __repr__(self):
        return '<WikiLink %s>' % self._title.encode('utf-8')

    def __reduce__(self):
        return (self.__class__, (self._title, self._text, self._span, self._abstract))


class WikiExtractor(object):
    def __init__(self, entity_db=None):
        self._entity_db = entity_db

    def extract_paragraphs(self, page):
        paragraphs = []
        cur_text = u''
        cur_links = []

        if page.is_redirect:
            return []

        abstract = True

        for node in self._parse_page(page).nodes:
            if isinstance(node, mwparserfromhell.nodes.Text):
                for (n, paragraph) in enumerate(unicode(node).split('\n')):
                    if n == 0:
                        cur_text += paragraph
                    else:
                        if not SPACE_REGEXP.match(cur_text):
                            paragraphs.append(Paragraph(cur_text, cur_links, abstract))
                        cur_text = paragraph
                        cur_links = []

            elif isinstance(node, mwparserfromhell.nodes.Wikilink):
                title = node.title.strip_code()
                if not title:
                    continue

                title = self._normalize_title(title)
                if self._entity_db is not None:
                    title = self._entity_db.resolve_redirect(title)

                if node.text:
                    text = node.text.strip_code()
                else:
                    text = node.title.strip_code()

                span = (len(cur_text), len(cur_text) + len(text))
                cur_text += text
                cur_links.append(WikiLink(title, text, span, abstract))

            elif isinstance(node, mwparserfromhell.nodes.Tag):
                if node.tag not in ('b', 'i'):
                    continue
                if not node.contents:
                    continue

                cur_text += node.contents.strip_code()

            elif isinstance(node, mwparserfromhell.nodes.Heading):
                abstract = False

        if not SPACE_REGEXP.match(cur_text):
            paragraphs.append(Paragraph(cur_text, cur_links, abstract))

        return paragraphs

    def _parse_page(self, page):
        try:
            return mwparserfromhell.parse(page.wiki_text)
        except Exception:
            return mwparserfromhell.parse('')

    @staticmethod
    def _normalize_title(title):
        return (title[0].upper() + title[1:]).replace('_', ' ')
