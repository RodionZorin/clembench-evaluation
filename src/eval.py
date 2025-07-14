import os
import subprocess
import yaml
import torch
import argparse
import shutil

import subprocess
import os


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="path to eval YAML config")
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

model_name = config["base_model"]["name"].split("/")[-1]
dataset_name = config["dataset"]["name"].split("/")[-1]
tasks = config["lm_eval_tasks"]
limit = config["limit"]
use_accelerate = config["use_accelerate"]
batch_size = config["batch_size"]
device = config["device"]
peft_path = f"{model_name.split('/')[-1]}_finetuned_{dataset_name.split('/')[-1]}"

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

shutil.copy(args.config, peft_path)

'''if not os.path.exists(output_path):
    os.makedirs(output_path)'''


tasks_str = ",".join(tasks)

base_args = [
    "--model", "hf",
    "--model_args", f"pretrained={config['base_model']['name']},peft={peft_path}",
    "--tasks", tasks_str,
    #"--confirm_run_unsafe_code",
    "--device", device,
    "--batch_size", batch_size,
    "--output_path", peft_path
]

if limit is not None:
    base_args += ["--limit", str(limit)]

if use_accelerate:
    num_processes = torch.cuda.device_count()
    command = [
        "accelerate", "launch",
        "--multi_gpu",
        f"--num_processes={num_processes}",
        "-m", "lm_eval"
    ] + base_args
else:
    command = ["lm_eval"] + base_args

print(command)

subprocess.run(command)

