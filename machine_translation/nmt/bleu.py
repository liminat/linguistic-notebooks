# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""BLEU."""
import sys
import re
import math
import unicodedata
from collections import Counter
import six
LIST_TYPES = (list, tuple)

__all__ = ['compute_bleu']


def _ngrams(segment, n):
    """Extracts n-grams from an input segment.

    Parameters
    ----------
    segment: list
        Text segment from which n-grams will be extracted.
    n: int
        Order of n-gram.

    Returns
    -------
    ngram_counts: Counter
        Contain all the nth n-grams in segment with a count of how many times each n-gram occurred.
    """
    ngram_counts = Counter()
    for i in range(0, len(segment) - n + 1):
        ngram = tuple(segment[i:i + n])
        ngram_counts[ngram] += 1
    return ngram_counts


def _split_compound_word(segment):
    """Put compounds in ATAT format.
       rich-text format" --> rich ##AT##-##AT## text format.
    """
    return re.sub(r'(\S)-(\S)', '\\1 ##AT##-##AT## \\2', ' '.join(segment)).split()


def _bpe_to_words(sentence, delimiter='@@'):
    """Convert a sequence of bpe words into sentence."""
    words = []
    word = ''
    delimiter_len = len(delimiter)
    for subwords in sentence:
        if len(subwords) >= delimiter_len and subwords[-delimiter_len:] == delimiter:
            word += subwords[:-delimiter_len]
        else:
            word += subwords
            words.append(word)
            word = ''
    return words


def _tokenize_mteval_13a(segment):
    r"""
    Tokenizes a string following the tokenizer in mteval-v13a.pl.
    See https://github.com/moses-smt/mosesdecoder/"
           "blob/master/scripts/generic/mteval-v14.pl#L917-L942
    Parameters
    ----------
    segment: str
        A string to be tokenized

    Returns
    -------
    The tokenized string
    """

    norm = segment.rstrip()

    norm = norm.replace('<skipped>', '')
    norm = norm.replace('-\n', '')
    norm = norm.replace('\n', ' ')
    norm = norm.replace('&quot;', '"')
    norm = norm.replace('&amp;', '&')
    norm = norm.replace('&lt;', '<')
    norm = norm.replace('&gt;', '>')

    norm = u' {} '.format(norm)
    norm = re.sub(r'([\{-\~\[-\` -\&\(-\+\:-\@\/])', ' \\1 ', norm)
    norm = re.sub(r'([^0-9])([\.,])', '\\1 \\2 ', norm)
    norm = re.sub(r'([\.,])([^0-9])', ' \\1 \\2', norm)
    norm = re.sub(r'([0-9])(-)', '\\1 \\2 ', norm)
    norm = re.sub(r'\s+', ' ', norm)
    norm = re.sub(r'^\s+', '', norm)
    norm = re.sub(r'\s+$', '', norm)

    return norm


class UnicodeRegex(object):
    """Ad-hoc hack to recognize all punctuation and symbols.
    """
    def __init__(self):
        punctuation = self._property_chars('P')
        self.nondigit_punct_re = re.compile(r'([^\d])([' + punctuation + r'])')
        self.punct_nondigit_re = re.compile(r'([' + punctuation + r'])([^\d])')
        self.symbol_re = re.compile('([' + self._property_chars('S') + '])')

    def _property_chars(self, prefix):
        return ''.join(six.unichr(x) for x in range(sys.maxunicode)
                       if unicodedata.category(six.unichr(x)).startswith(prefix))


unicodeRegex = UnicodeRegex()


def _tokenize_mteval_v14_intl(segment):
    r"""Tokenize a string following following the international tokenizer in mteval-v14a.pl.
    See https://github.com/moses-smt/mosesdecoder/"
           "blob/master/scripts/generic/mteval-v14.pl#L954-L983

    Parameters
    ----------
    segment: str
        A string to be tokenized

    Returns
    -------
    The tokenized string
    """
    segment = segment.rstrip()
    segment = unicodeRegex.nondigit_punct_re.sub(r'\1 \2 ', segment)
    segment = unicodeRegex.punct_nondigit_re.sub(r' \1 \2', segment)
    segment = unicodeRegex.symbol_re.sub(r' \1 ', segment)
    return segment.strip()


TOKENIZERS = {
    '13a': _tokenize_mteval_13a,
    'intl': _tokenize_mteval_v14_intl,
    None: lambda x: x,
}


def compute_bleu(reference_corpus_list, translation_corpus, tokenized=True,
                 tokenizer='13a', max_n=4, smooth=False, lower_case=False,
                 bpe=False, split_compound_word=False):
    r"""Compute bleu score of translation against references.

    Parameters
    ----------
    reference_corpus_list: list of list(list(str)) or list of list(str)
        list of list(list(str)): tokenized references
        list of list(str): plain text
        List of references for each translation.
    translation_corpus: list(list(str)) or list(str)
        list(list(str)