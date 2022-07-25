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

## Language model tuning

Fine-tuning the library models for language modeling on a text dataset 
for GPT, GPT-2. Causal languages like this are trained or fine-tuned using a causal language 
modeling (CLM) loss.

In theory, serving a tuned model can increase incentive and earnings on the bittensor network.
However this depends on many factors: the choice of model, the data used
for tuning, and (to a lesser extent), the hyparameters used for tuning itself. This is not a silver
bullet that will immediately guarantee higher earnings, but differences will be more pronounced
once the Synapse update is released (time of writing: July 25, 2022).


In the following examples, we will run on datasets hosted on Bittensor's IPFS mountain dataset, 
on HuggingFace's dataset [hub](https://huggingface.co/datasets) or with your own text files.

For a full list of models that will work with this script, 
[refer to this link](https://huggingface.co/models?filter=text-generation).


## Installation & Requirements
This code assumes you have `bittensor` already installed on your machine, and is meant to be
run entirely separately. Some basic linux commandline knowledge is assumed, but 
[this guide](https://ubuntu.com/tutorials/command-line-for-beginners) should provide a good starting
point to navigate and move around files, directories, etc.

To start, clone this repository:
```commandline
git clone https://github.com/opentensor/clm_model_tuning 
```

All following commands assume you are working from this folder, i.e. you must `cd` into the directory
created by the previous step.

Run ```pip install -r requirements.txt``` to install the additional packages required by this script.

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
[bittensor's documentation](https://docs.bittensor.com). A full list of models that can be trained by this
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
The defaults are chosen for quick training and not tuned; you will need to experiment and adjust 
these.

**Note**: The above parameters are the *only* commands you can override with this script. That is,
you may not pass flags you would normally use when running `btcli` (i.e. `--neuron.device` will *not* work).
If there is a flag you wish to modify feel free to submit a feature request.

To view the changeable parameters, open `conf/config.yaml` in whatever text editor you prefer, or
use `cat conf/config.yaml` to view them.

You do not need to edit this file to change the parameters; they may be overridden when you call this
script. e.g., if you wish to change the model to `distilgpt2`, and the output directory to `distilgpt-tuned`, you would run:
```commandline
python3 finetune_using_clm.py model.name=distilgpt2 output_dir=distilgpt-tuned
```

Note the nested structure in the config, since `model` is above `name` in `conf.yaml`, you must override
`model.name` when invoking the command.


## Serving custom models on bittensor

To serve your tuned model on bittensor, just override `neuron.model_name` with the path to your 
tuned model:
```bash
btcli run ..... --neuron.model_name=/home/{YOUR_USENAME}/clm_model_tuning/tuned-model
```

## Limitations & Warnings

Early stopping is not yet supported. Many features are implemented but not thoroughly tested, if
you encounter an issue, reach out on discord or (preferably) create an issue on this github page.