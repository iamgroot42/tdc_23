
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch as ch
from itertools import chain

from utils import load_targets, generate_alternative_prompts


SETTINGS = {
    "base": {
        "hf": "TDC2023/trojan-base-pythia-1.4b",
        "targets": "./data/dev/base/trojan_specifications_train_dev_base.json",
        "batch_size": 512
    },
    "large": {
        "hf": "TDC2023/trojan-large-pythia-6.9b",
        "targets": "./data/dev/large/trojan_specifications_train_dev_large.json",
        "batch_size": 128
    }
}

def main(setting: str = "base", random_start_mixup: bool = False, n_iters: int = 20):
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(SETTINGS[setting]["hf"], padding_side='left')
    tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
    model = AutoModelForCausalLM.from_pretrained(SETTINGS[setting]["hf"], torch_dtype=ch.float16).cuda().eval()

    all_tokens = list(tokenizer.get_vocab().keys())
    # Filter out special tokens
    all_tokens = [x for x in all_tokens if x not in [tokenizer.bos_token, tokenizer.eos_token, tokenizer.unk_token, tokenizer.pad_token]]
    # Only consider ascii-printable

    # Also keep actual trojan data
    actual_trojans = load_targets(SETTINGS[setting]["targets"])
    all_known_triggers = list(chain.from_iterable(actual_trojans.values()))

    # Read from file (whatever is there so far)
    with open(f"predictions_{setting}.json", 'r') as f:
        accurate_trojans = json.load(f)
    
    # Load targets
    target_trojans = load_targets()

    for x in target_trojans:
        # Skip if already in there
        if x in accurate_trojans:
            continue
        triggers = generate_alternative_prompts(x, all_known_triggers=all_known_triggers,
                                                model=model, tokenizer=tokenizer,
                                                batch_size=SETTINGS[setting]["batch_size"],
                                                random_start_mixup=random_start_mixup,
                                                n_iters=n_iters)
        accurate_trojans[x] = triggers

        # Also write to file at end of it all (to keep track of progress via notebook)
        with open(f"predictions_{setting}_{random_start_mixup}_{n_iters}.json", 'w') as f:
            json.dump(accurate_trojans, f, indent=4)


if __name__ == "__main__":
    import sys
    setting = sys.argv[1]
    random_start_mixup = bool(int(sys.argv[2]))
    n_iters = int(sys.argv[3])
    main(setting, random_start_mixup)
