

import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch as ch

from utils import check_for_attack_success, SETTINGS


def main(setting: str):
    tokenizer = AutoTokenizer.from_pretrained(SETTINGS[setting]["hf"], padding_side='left')
    tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
    model = AutoModelForCausalLM.from_pretrained(SETTINGS[setting]["hf"], torch_dtype=ch.float16).cuda().eval()
    
    # Open predictions file
    with open(f"predictions_{setting}.json", 'r') as f:
        dic = json.load(f)

    successful_triggers = {}
    for target, v in dic.items():
        s = 0
        max_new=len(tokenizer(target).input_ids)
        for trigger in v:
            model_output = check_for_attack_success(
                                model,
                                tokenizer,
                                trigger,
                                max_new=max_new)
            if model_output[-len(target):] == target:
                s += 1
                if target not in successful_triggers:
                    successful_triggers[target] = []
                successful_triggers[target].append(trigger)
        print(f"{target}: {s}")

    # Write to file
    with open(SETTINGS[setting]["generated_trojans"], 'w') as f:
        json.dump(successful_triggers, f, indent=4)


if __name__ == "__main__":
    import sys
    setting = sys.argv[1]
    main(setting)
    