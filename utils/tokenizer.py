# -*- coding: utf-8 -*-

import re


class Token(object):
    __slots__ = ('_text', '_span')

    def __init__(self, text, span):
        self._text = text
        self._span = span

    @property
    def text(self):
        return self._text

    @property
    def span(self):
        return self._span

    def __repr__(self):
        return '<Token %s>' % self.text.encode('utf-8')

    def __reduce__(self):
        return (self.__class__, (self.text, self.span))


class RegexpTokenizer(object):
    __slots__ = ('_rule',)

    def __init__(self, rule=ur'[\w\d]+'):
        self._rule = re.compile(rule, re.UNICODE)

    def tokenize(self, text):
        spans = [o.span() for o in self._rule.finditer(text)]
        return [Token(text[s[0]:s[1]], s) for s in spans]
