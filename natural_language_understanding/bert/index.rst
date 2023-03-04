Bidirectional Encoder Representations from Transformers
-------------------------------------------------------

:download:`Download scripts </model_zoo/bert.zip>`

Reference: Devlin, Jacob, et al. "`Bert: Pre-training of deep bidirectional transformers for language understanding. <https://arxiv.org/abs/1810.04805>`_" arXiv preprint arXiv:1810.04805 (2018).

Note: BERT model requires `nightly version of MXNet <https://mxnet.incubator.apache.org/versions/master/install/index.html?version=master&platform=Linux&language=Python&processor=CPU>`__. 

The following pre-trained BERT models are available from the **gluonnlp.model.get_model** API:

+-------------------------------+----------------+-----------------+
|                               | bert_12_768_12 | bert_24_1024_16 |
+===============================+================+=================+
| book_corpus_wiki_en_uncased   | ✓              | ✓               |
+-------------------------------+----------------+-----------------+
| book_corpus_wiki_en_cased     | ✓              | ✓               |
+-------------------------------+----------------+-----------------+
| wiki_multilingual_uncased     | ✓              | x               |
+-------------------------------+----------------+-----------------+
| wiki_multilingual_cased       | ✓              | x               |
+-------------------------------+----------------+-----------------+
| wiki_cn_cased                 | ✓              | x               |
+-------------------------------+----------------+-----------------+
| scibert_scivocab_uncased      | ✓              | x               |
+-------------------------------+----------------+-----------------+
| scibert_scivocab_cased        | ✓              | x               |
+-------------------------------+----------------+-----------------+
| scibert_basevocab_uncased     | ✓              | x               |
+-------------------------------+----------------+-----------------+
| scibert_basevocab_cased       | ✓              | x               |
+-------------------------------+----------------+-----------------+
| biobert_v1.0_pmc_cased        | ✓              | x               |
+-------------------------------+----------------+-----------------+
| biobert_v1.0_pubmed_cased     | ✓              | x               |
+-------------------------------+----------------+-----------------+
| biobert_v1.0_pubmed_pmc_cased | ✓              | x               |
+-------------------------------+----------------+-----------------+
| biobert_v1.1_pubmed_cased     | ✓              | x               |
+-------------------------------+----------------+-----------------+
| clinicalbert_uncased          | ✓              | x               |
+-------------------------------+----------------+-----------------+

where **bert_12_768_12** refers to the BERT BASE model, and **bert_24_1024_16** refers to the BERT LARGE model.

BERT for Sentence Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GluonNLP provides the following example script to fine-tune sentence classification with pre-trained
BERT model.

.. editing URL for the following table: https://tinyurl.com/y4n8q84w

+---------------------+--------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| Dataset             | MRPC                                                                                                         | RTE                                                                                                         | SST-2                                                                                                       | MNLI-M/MM                                                                                                    | XNLI (Chinese)                                                                                               |
+=====================+==============================================================================================================+=============================================================================================================+=============================================================================================================+==============================================================================================================+==============================================================================================================+
| Validation Accuracy | 88.7%                                                                                                        | 70.8%                                                                                                       | 93%                                                                                                         | 84.55%, 84.66%                                                                                               | 78.27%                                                                                                       |
+---------------------+--------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| Log                 | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetuned_mrpc.log>`__       | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetuned_rte.log>`__       | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetuned_sst.log>`__       | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetuned_mnli.log>`__       | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetuned_xnli.log>`__       |
+---------------------+--------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| Command             | `command <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetuned_mrpc.sh>`__    | `command <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetuned_rte.sh>`__    | `command <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetuned_sst.sh>`__    | `command <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetuned_mnli.sh>`__    | `command <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetuned_xnli.sh>`__    |
+---------------------+--------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+


For all model settings above, we set learing rate = 2e-5, optimizer = bertadam, model = bert_12_768_12. Other tasks can be modeled with `--task_name` parameter.

.. editing URL for the following table: https://tinyurl.com/y5rrowj3

BERT for Question Answering on SQuAD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+---------+-----------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------+
| Dataset | SQuAD 1.1                                                                                                                               | SQuAD 1.1                                                                                                                                | SQuAD 2.0                                                                                                                                |
+=========+=========================================================================================================================================+==========================================================================================================================================+==========================================================================================================================================+
| Model   | bert_12_768_12                                                                                                                          | bert_24_1024_16                                                                                                                          | bert_24_1024_16                                                                                                                          |
+---------+-----------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------+
| F1 / EM | 88.53 / 80.98                                                                                                                           | 90.97 / 84.05                                                                                                                            | 77.96 / 81.02                                                                                                                            |
+---------+-----------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------+
| Log     | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetune_squad1.1_base_mx1.5.0b20190216.log>`__         | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetune_squad1.1_large_mx1.5.0b20190216.log>`__         | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetune_squad2.0_large_mx1.5.0b20160216.log>`__         |
+---------+-----------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------+
| Command | `command <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetune_squad1.1_base_mx1.5.0b20190216.sh>`__      | `command <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetune_squad1.1_large_mx1.5.0b20190216.sh>`__      | `command <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetune_squad2.0_large_mx1.5.0b20160216.sh>`__      |
+---------+-----------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------+

For all model settings above, we set learing rate = 3e-5 and optimizer = adam.

Note that the BERT model is memory-consuming. If you have limited GPU memory, you can use the following command to accumulate gradient to achieve the same result with a large batch size by setting *accumulate* and *batch_size* arguments accordingly.

.. code-block:: console

    $ python finetune_squad.py --optimizer adam --accumulate 2 --batch_size 6 --lr 3e-5 --epochs 2 --gpu 0

SQuAD 2.0
+++++++++

For SQuAD 2.0, you need to specify the parameter *version_2* and specify the parameter *null_score_diff_threshold*. Typical values are between -1.0 and -5.0. Use the following command to fine-tune the BERT large model on SQuAD 2.0 and generate predictions.json.

To get the score of the dev data, you need to download the dev dataset (`dev-v2.0.json <https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json>`_) and the evaluate script (`evaluate-2.0.py <https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/>`_). Then use the following command to get the score of the dev dataset.

.. code-block:: console

    $ python evaluate-v2.0.py dev-v2.0.json predictions.json


BERT Pre-training
~~~~~~~~~~~~~~~~~

We also provide scripts for pre-training BERT with m