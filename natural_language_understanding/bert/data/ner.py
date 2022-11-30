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
"""Data utilities for the named entity recognition task."""

import logging
from collections import namedtuple

import numpy as np
import mxnet as mx
import gluonnlp as nlp

TaggedToken = namedtuple('TaggedToken', ['text', 'tag'])
PredictedToken = namedtuple('PredictedToken', ['text', 'true_tag', 'pred_tag'])

NULL_TAG = 'X'

def bio_bioes(tokens):
    """Convert a list of TaggedTokens in BIO(2) scheme to BIOES scheme.

    Parameters
    ----------
    tokens: List[TaggedToken]
        A list of tokens in BIO(2) scheme

    Returns
    -------
    List[TaggedToken]:
        A list of tokens in BIOES scheme
    """
    ret = []
    for index, token in enumerate(tokens):
        if token.tag == 'O':
            ret.append(token)
        elif token.tag.startswith('B'):
            # if a B-tag is continued by other tokens with the same entity,
            # then it is still a B-tag
            if index + 1 < len(tokens) and tokens[index + 1].tag.startswith('I'):
                ret.append(token)
            else:
                ret.append(TaggedToken(text=token.text, tag='S' + token.tag[1:]))
        elif token.tag.startswith('I'):
            # if an I-tag is continued by other tokens with the same entity,
            # then it is still an I-tag
            if index + 1 < len(tokens) and tokens[index + 1].tag.startswith('I'):
                ret.append(token)
            else:
                ret.append(TaggedToken(text=token.text, tag='E' + token.tag[1:]))
    return ret


def read_bio_as_bio2(data_path):
    """Read CoNLL-formatted text file in BIO scheme in given path as sentences in BIO2 scheme.

    Parameters
    ----------
    data_path: str
        Path of the data file to read

    Returns
    -------
    List[List[TaggedToken]]:
        List of sentences, each of which is a List of TaggedTokens
    """

    with open(data_path, 'r') as ifp:
        sentence_list = []
        current_sentence = []
        prev_tag = 'O'

        for line in ifp:
            if len(line.strip()) > 0:
                word, _, _, tag = line.rstrip().split(' ')
                # convert BIO tag to BIO2 tag
                if tag == 'O':
                    bio2_tag = 'O'
                else:
                    if prev_tag == 'O' or tag[2:] != prev_tag[2:]:
                        bio2_tag = 'B' + tag[1:]
                    else:
                        bio2_tag = tag
                current_sentence.append(TaggedToken(text=word, tag=bio2_tag))
                prev_tag = tag
            else:
                # the sentence was completed if an empty line occurred; flush the current sentence.
                sentence_list.append(current_sentence)
                current_sentence = []
                prev_tag = 'O'

        # check if there is a remaining token. in most CoNLL data files, this does not happen.
        if len(current_sentence) > 0:
            sentence_list.append(current_sentence)
        return sentence_list


def remove_docstart_sentence(sentences):
    """Remove -DOCSTART- sen