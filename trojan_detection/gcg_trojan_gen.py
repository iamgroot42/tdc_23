
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch as ch
from itertools import chain
from tqdm import tqdm

from utils import load_targets, generate_alternative_prompts, SETTINGS


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

    # Also use information from generated trojans
    generated_trojans = load_targets(SETTINGS[setting]["generated_trojans"])

    # Read from file (whatever is there so far)
    with open(f"predictions_{setting}.json", 'r') as f:
        accurate_trojans = json.load(f)
    
    # Load targets
    target_trojans = load_targets()

    # TODO: Also keep track of failed trigger starts (for future generations)?

    for x in target_trojans:
        # Collect information on already-known triggers
        if x in generated_trojans:
            known_triggers = list(set(generated_trojans[x]))
        
        # Add all trojans NOT for this target
        all_known_triggers_use = all_known_triggers[:]
        for k, v in generated_trojans.items():
            if k != x:
                all_known_triggers_use.extend(v)
        all_known_triggers_use = list(set(all_known_triggers_use))

        triggers = generate_alternative_prompts(x, all_known_triggers=all_known_triggers_use,
                                                model=model, tokenizer=tokenizer,
                                                batch_size=SETTINGS[setting]["batch_size"],
                                                random_start_mixup=random_start_mixup,
                                                n_iters=n_iters,
                                                known_triggers=known_triggers)
        accurate_trojans[x] = triggers

        # Also write to file at end of it all (to keep track of progress via notebook)
        with open(f"predictions_{setting}.json", 'w') as f:
            json.dump(accurate_trojans, f, indent=4)


if __name__ == "__main__":
    import sys
    setting = sys.argv[1]
    main(setting)
