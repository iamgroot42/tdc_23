import json
import numpy as np
import os
from tqdm import tqdm

from transformers import AutoTokenizer

# Load LLAMA-2 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            trust_remote_code=True,
            use_fast=False)

# Open and read files in results folder
folder = "./indi_fast"
prefixes = {}
all_suffix_pool = []
for filename in os.listdir(folder):
    offset = int(filename.split("offset")[1].split("_")[0])
    # Open this json file
    with open(os.path.join(folder, filename)) as f:
        data = json.load(f)
        wanted = data['controls'][1:]
        goal = data['params']['goals'][0]
        # Use last 1-2 only
        # wanted = wanted[-2:]
        prefixes[offset] = (goal, wanted)
        all_suffix_pool += wanted

print("Suffix pool size: ", len(all_suffix_pool))

# Open and read target file (EN)
triggers = []
with open("../../data/advbench/harmful_behaviors.csv") as f:
    for line in f:
        triggers.append(line.rstrip('\n').split("----")[0])
triggers = triggers[1:]

empty_dict = {}
for i, trigger in tqdm(enumerate(triggers), desc="Generating test cases", total=len(triggers)):
    trigger_use = trigger
    random_suffix_sample = []
    if i in prefixes:
        # Use optimal tokenization
        trigger_use = prefixes[i][0]

        # Add corresponding suffixes
        corresponding = prefixes[i][1]
        random_suffix_sample += corresponding

    # Pick suffix
    random_suffix_sample += list(np.random.choice(all_suffix_pool, 50 - len(random_suffix_sample), replace=True))

    # Write to file
    adv = []
    for prefix in random_suffix_sample:
        to_add = trigger_use + " " + prefix
        # Count number of tokens in this
        if len(tokenizer.encode(to_add)) > 256:
            # Keep replacing random '<s> ' with ' ' until it fits
            while len(tokenizer.encode(to_add)) > 256:
                to_add = to_add.replace("<s> ", "")
        adv.append(to_add)
    empty_dict[trigger] = adv

# Save as json
with open('test_cases.json', 'w') as fp:
    json.dump(empty_dict, fp)
