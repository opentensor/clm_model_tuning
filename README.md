# CLM Model Tuning

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

## Installation & Requirements
`bittensor` must be installed either locally or in the virtual environment you are working from.

Run ```pip install -r requirements.txt``` to install the additional packages for this script.

## Language model tuning

Fine-tuning the library models for language modeling on a text dataset 
for GPT, GPT-2. Causal languages like this are trained or fine-tuned using a causal language 
modeling (CLM) loss.

The following examples, we will run on datasets hosted on Bittensor's IPFS mountain dataset, 
on HuggingFace's dataset [hub](https://huggingface.co/datasets) or with your own text files.

### On bittensor

By default, the script will fine-tune GPT2 for bittensor's mountain dataset. Running:
```bash
python finetune_using_clm.py
```
will tune gpt2 with bittensor's dataset and save the output to `tuned-model`.

to change the model you are tuning to, e.g. `distilgpt2`, run:
```bash
python finetune_using_clm.py model.name=distilgpt2
```

Some sample models to try are available under the server customization section of 
[bittensor's documentation](docs.bittensor.com). A full list of models that can be trained by this
script are available on [huggingface](https://huggingface.co/models?filter=text-generation).

### On huggingface datasets

Any text dataset on [huggingface](https://huggingface.co/datasets) should work by default by
overriding the `dataset.name` and `dataset.config` parameters:

```bash
python finetune_using_clm.py dataset.name=wikitext dataset.config_name=wikitext-103-v1
```

### On your own data

If you have a .txt file saved locally, you can override `dataset.name` as above:
```bash
python finetune_using_clm.py dataset.name=./path/to/your/data.txt
```

**Note** if using your own data, you may have many short sentences and the block size may be 
insufficient for reasonable performance. It's recommended you pass the flag
`dataset.concatenate_raw=true` to give the model more context when training. This will reduce
the number of batches.

### Configuring training parameters

All configurable parameters are visible and documented in `conf/config.yaml`. 
The defaults are chosen for quick training and not tuned; you will need to adjust these accordingly.


## Serving custom models on bittensor

To serve your tuned model on bittensor, just override `neuron.model_name` with the path to your 
tuned model:
```bash
btcli run ..... --neuron.model_name=/home/user/models/my-tuned-gpt2
```

## Limitations

Early stopping is not yet supported. Many features are implemented but not thoroughly tested, if
you encounter an issue, reach out on discord or (preferably) create an issue on this github page.