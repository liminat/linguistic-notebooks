Text Generation
---------------

:download:`[Download] </model_zoo/text_generation.zip>`

Sampling a Language Model
+++++++++++++++++++++++++

This script can be used to generate sentences using beam search or a sequence sampler, to sample from a pre-trained language model such as GPT-2. For example:

.. code-block:: console

   $ python sequence_sampling.py random-sample \
         --bos 'Deep learning and natural language processing' \
         --beam-size 1 --print-num 1 \
         --lm-model gpt2_345m # options are {gpt2_117m, gpt2_345m} \
         --max-length 1024

Output is

.. code-block:: console

    Sampling Parameters: beam_size=1, temperature=1.0, use_top_k=None
    Generation Result:
    ['Deep learning and natural language processing brought application choice in healthcare and perception of sounds and heat to new heights, enriching our physical communities with medical devices and creating vibrant cultures. Anecdote is slowly diminishing but is hardly obsolete nor more appealing than experience.Despite those last words of wisdom, most headset makers even spook us with the complexity and poor code