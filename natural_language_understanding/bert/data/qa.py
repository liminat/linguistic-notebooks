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
    dataset: dataset-like object
        Source dataset.
    transform: callable
        Transformer function.
    num_workers: int, default 8
        The number of multiprocessing workers to use for data preprocessing.

    """
    worker_fn = partial(_worker_fn, transform=transform)
    start = time.time()

    pool = mp.Pool(num_workers)
    dataset_transform = []
    dataset_len = []
    for data in pool.map(worker_fn, dataset):
        if data:
            for _data in data:
                dataset_transform.append(_data[:-1])
                dataset_len.append(_data[-1])

    dataset = SimpleDataset(dataset_transform).transform(
        lambda x: (x[0], x[1], x[2], x[3], x[4], x[5]))
    end = time.time()
    pool.close()
    print('Done! Transform dataset costs %.2f seconds.' % (end-start))
    return dataset, dataset_len


class SQuADFeature(object):
    """Single feature of a single example transform of the SQuAD question.

    """

    def __init__(self,
                 example_id,
                 qas_id,
                 doc_tokens,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 valid_length,
                 segment_ids,
                 start_position,
                 end_position,
                 is_impossible):
        self.example_id = example_id
        self.qas_id = qas_id
        self.doc_tokens = doc_tokens
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.valid_length = valid_length
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


class SQuADTransform(object):
    """Dataset Transformation for BERT-style QA.

    The transformation is processed in the following steps:
    - Convert from gluonnlp.data.SQuAD's record to SquadExample.
    - Tokenize the question_text in the example.
    - For examples where the document is too long,
      use a sliding window to split into multiple fe