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
    norm = re.sub(r'([\{-\~\[-\` -\&\(-\+\:-\@\/])', ' \\1 ', nor