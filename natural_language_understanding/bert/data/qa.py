# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and DMLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT for QA datasets."""
import collections
import multiprocessing as mp
import time
from functools import partial

from mxnet.gluon.data import SimpleDataset
from gluonnlp.data.utils import whitespace_splitter

__all__ = ['SQuADTransform', 'preprocess_dataset']

class SquadExample(object):
    """A single training/test example for SQuAD question.

       For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 example_id,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=False):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.example_id = example_id

def _worker_fn(example, transform):
    """Function for processing data in worker process."""
    feature = transform(example)
    return feature


def preprocess_dataset(dataset, transform, num_workers=8):
    """Use multiprocessing to perform transform for dataset.

    Parameters
    ----------
