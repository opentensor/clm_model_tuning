# ModelGuide

<!---
Copyright 2020 The OpenTensor Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

**Note**: This script was adapted from HuggingFace's Transformers/language-modeling code.
## Language model training

Fine-tuning (or training from scratch) the library models for language modeling on a text dataset for GPT, GPT-2. Causal languages like this are trained or fine-tuned using a causal language modeling (CLM) loss.

The following examples, we will run on datasets hosted on Bittensor's IPFS mountain dataset, on HuggingFace's dataset [hub](https://huggingface.co/datasets) or with your own text files for training and validation. We give examples of both below.

### GPT-2/GPT and causal language modeling

The following example fine-tunes GPT-2 on WikiText-2 from the Huggingface repository. We're using the raw WikiText-2 (no tokens were replaced before
the tokenization). The loss here is that of causal language modeling.

```bash
python run_clm.py \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-clm
```

To train it on Bittensor's Mountain dataset, you simply need to change `dataset_name` with "bittensor", as follows:

```bash
python run_clm.py \
    --model_name_or_path gpt2 \
    --dataset_name bittensor \
    --dataset_config_name bittensor \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-clm
```

This takes about half an hour to train on a single K80 GPU and about one minute for the evaluation to run. It reaches
a score of ~20 perplexity once fine-tuned on the dataset.

To run on your own training and validation files, use the following command:

```bash
python run_clm.py \
    --model_name_or_path gpt2 \
    --train_file path_to_train_file \
    --validation_file path_to_validation_file \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-clm
```

## Creating a model on the fly

When training a model from scratch, configuration values may be overridden with the help of `--config_overrides`:


```bash
python run_clm.py --model_type gpt2 --tokenizer_name gpt2 \ --config_overrides="n_embd=1024,n_head=16,n_layer=48,n_positions=102" \
[...]
```
