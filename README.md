## Clembench Evaluation Project

Tuning models to improve performance on Clembench

### Project Goals

Find a tuning pipeline that improves model's performance on Clembench.

Clembench: Systematic Evaluation of Chat-Optimized Language Models as Conversational Agents

### Repository Setup

#### Clone the Repository (with a playpen Submodule)

Playpen: an Environment for Learning in Interaction.

```bash
git clone --recurse-submodules https://github.com/RodionZorin/clembench-evaluation.git
```

This command will:
- Clone the main repository `clembench-evaluation`.
- Automatically initialize and clone the `playpen` submodule .

###  Installation

#### Installation of Playpen

```bash
cd clembench-evaluation/playpen
```
Set up the Python environment. Note: Playpen requires Python 3.10+.

```bash
python -m venv venv --system-site-packages && source venv/bin/activate
```

Install the clemcore framework to run games, backends and models. Note: The huggingface extra is (only) necessary to train with local hf models.

```bash
pip install clemcore[huggingface]==2.4.0
```

Make playpen available via CLI and install TRL enable running the examples.

```bash
pip install '.[trl]'
```

Make the clembench games, e.g. taboo, available for learning. For this, clone the clembench repository to a directory of your choice.

```bash
git clone https://github.com/clp-research/clembench
```

Furthermore, we must install the clembench game requirements in our venv so that all games can be run properly:

```bash
cd clembench
pip install -r requirements.txt
```

Then, back in you playpen workspace, copy the game_registry.json.template to game_registry.json so that the clem CLI can find it in the current working directory. Set the path to the directory which contains the clembench repository. The following command has a similar effect:

```bash
cd ..
echo '[{"benchmark_path": "your/path/to/clembench"}]' > game_registry.json
```

Note: Adding the game registry file is not necessary, when you clone the clembench repository directly in your playpen workspace. In this case the clem CLI can directly find the games by looking into sub-directories.

In any case, check that games are available via:

```bash
clem list games
```

Now having everything set up, you can follow the experiment guide or jump to the TLDR section for a quick overview.

### Running tuning.py

tuning.py script enables you to perform tuning with SFT (Supervised Fine-Tuning) and DPO (Direct Policy Optimization).

It uses PEFT (Parameter Efficient Fine-tuning): a technique called low-rank adapters (LoRA) where only a smaller set of parameters (adapters) is trained to improve the model's performance

To run PEFT navigate to src and install requirements:

```bash
cd src
pip install -r requirements
```
Now you can run tuning.py.
Note that PEFT will load your model using 4bit quantization: it provides faster tuning

```bash
python3  tuning.py --config config.yaml
```
After the tuning is finished, PEFT adapters of 4bit quantized model are saved together with the used config.yaml

#### Config.yaml


Config.yaml conveys all the essential parameters for tuning.
Let's go through the config.yaml line by line

Here is one of the possible examples of the config that utilizes DPO
All the unnecessary for DPO lines are commented (but are not to be deleted!)

```bash
tuning:
  #type: sft #Supervised Fine-Tuning
  type: dpo #Direct Preference Optimization

base_model: #this is the model you want to tune. Note: in case of DPO this equals the reference policy
  name: unsloth/Meta-Llama-3.1-8B-Instruct #since DPO is chosen above, this is a reference policy 

sft_dataset: #used for sft tuning. For example: allenai/tulu-3-sft-mixture
  name: not_used #since DPO is chosen above, an SFT dataset is not used

sft_model: #used for dpo tuning = policy we want to tune
  name: unsloth/Meta-Llama-3.1-8B-Instruct #policy we want to tune. Initially, coincides with the reference policy

preference_dataset: @the dataset of preference data used for dpo tuning
  name: allenai/tulu-3-IF-augmented-on-policy-8b
  
max_seq_length:
  length: 2048

 #PEFT hyperparameters
low_rank_matrice:
  size: 16 # also recommended 32, 64, 128
influence:
  rate: 16 # also recommended 32, 64, 128
dropout:
  rate: 0.05
training:
  learning_rate: 2e-4
  batch_size: 4
  grad_acc_steps: 4 #gradients are computed 4 times on the defined batch size, then the average of gradients is taken, and finally the model is updated; that gives one training (tuning) step
  max_training_steps: 700 #the number of steps the model is updated
  warmup_steps: 20
  beta: 0.1 #used only for dpo tuning, part of KL divergence
```

### Evaluation of the Tuned Model on Clembench

Firstly, you need to navigate to Playpen directory and add your tuned model in the model_registry.json file.

```bash
cd ..
cd playpen
nano model_rejistry.json
```

In the opened model_rejistry.json, add your base 4bit quantized model and the directory where your 4bit PEFT adapters live.
Pay attention to the correct EOS token you model utilizes (in case of Llama 3.1 8B, this is "<\\|eot_id\\|>"

For example:

```bash
[
  {
    "model_name": "llama3-8b-it-4bit-pref-pers-lora", #this is the name you will use for an evaluation run, see below 
    "backend": "huggingface_local",
    "huggingface_id": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", #this is your base model, and you will add the trained adapters to it, see below; note, the model is in 4bit quantization
    "release_date": "2025-06-18",
    "open_weight": true,
    "parameters": "8B",
    "languages": ["en", "de", "fr", "it", "pt", "hi", "es", "th"],
    "context_size": "8192",
    "license": {
      "name": "Meta (with LoRA)",
      "url": "https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE"
    },
    "tokenizer_kwargs": {
      "use_fast": true
    },
    "model_config": {
      "peft_model": "/home/users/rzorin/pm-25/Meta-Llama-3.1-8B-Instruct-bnb-4bit_finetuned_tulu-3-pref-personas-instruction-following", #this is the directory where your trained adapters live
      "requires_api_key": false,
      "premade_chat_template": true,
      "eos_to_cull": "<\\|eot_id\\|>"
    }
  }
]
```

Secondly, you can now run evaluation on clembench. It can be done in two ways:

#### First way of evaluation:

```bash
clem run -g "{'benchmark':['2.0']}" -m llama3-8b-it-4bit-pref-pers-lora
```

This will start the playing, which usually takes ~2 hours

After the playing is finished, use:
```bash
clem score
clem eval
```

That forms the results directory with .html file where the clembench score, averaged over all the games, is presented as well as clembench scores of single games.

#### Second way of evaluation:

The second line of evaluation will give you the clembench score, averaged over all the games, and the so called statscore - an evaluation against other general benchmarks to be sure that you model has not overfitted towards games.

```bash
playpen eval llama3-8b-it-4bit-pref-pers-lora
```

##### Baseline

To estimate the results of your tuned model, you might want to compare the results against a relevant baseline.

For example, in our case the baseline model is supposed to be unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit. Since now we do not use any PEFT adapters, note that the model registry will look slightly different:

```bash
[
 {
    "model_name": "llama3-8b-it-4bit",
    "backend": "huggingface_local",
    "huggingface_id": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "release_date": "2024-06-18",
    "open_weight": true,
    "parameters": "8B",
    "languages": ["en"],
    "context_size": "8192",
    "license": {
      "name": "Meta Llama 3 Community License",
      "url": "https://github.com/meta-llama/llama3/blob/main/LICENSE"
    },
    "model_config": {
      #note that no adapters are here anymore
      "requires_api_key": false,
      "premade_chat_template": true,
      "eos_to_cull": "<\\|eot_id\\|>"
    }
  }
]
```
