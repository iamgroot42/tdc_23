
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch as ch
import argparse
from itertools import chain

from utils import load_targets, generate_alternative_prompts, SETTINGS, get_likelihood


def main(args):
    random_start_mixup = False
    setting = args.setting
    n_iters = args.n_iters
    keep_best_success = args.keep_best_success
    break_on_success = args.break_on_success

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
    accurate_trojans = load_targets(f"predictions_{setting}.json")

    # Load targets
    target_trojans = load_targets()

    # Load triggers that failed previously
    failed_triggers_dict = load_targets(SETTINGS[setting]["failed"])

    for x in target_trojans:

        if not keep_best_success:
            # Collect information on already-known triggers
            known_triggers = None
            if x in generated_trojans:
                trojan_strings = [j[0] for j in generated_trojans[x]]
                known_triggers = list(set(trojan_strings))

                # We're done if we got 20 triggers, skip to next target
                if len(known_triggers) == 20:
                    continue

        # Add all trojans NOT for this target
        all_known_triggers_use = all_known_triggers[:]
        for k, v in generated_trojans.items():
            if k != x:
                all_known_triggers_use.extend([j[0] for j in v])
        all_known_triggers_use = list(set(all_known_triggers_use))

        # Subtract failed triggers from list
        if x in failed_triggers_dict:
            all_known_triggers_use = list(set(all_known_triggers_use) - set(failed_triggers_dict[x]))
        
        # Run only when >= 100 candidates present
        if len(all_known_triggers_use) < 50:
            print(f"Ignoring failed triggers lead to too few triggers for target {x}.")
            continue

        # Special handling for 'JESUS' trigger
        x_send = x.replace(" , ", ", ")

        # This is where the magic happens
        triggers, failed_triggers = generate_alternative_prompts(x_send, all_known_triggers=all_known_triggers_use,
                                                model=model, tokenizer=tokenizer,
                                                batch_size=SETTINGS[setting]["batch_size"],
                                                random_start_mixup=random_start_mixup,
                                                n_iters=n_iters,
                                                keep_best_success=keep_best_success,
                                                break_on_success=break_on_success)

        # Compute scores for these triggers
        accurate_trojans[x] = [(j, get_likelihood(model, tokenizer, j, x)) for j in triggers]

        if len(failed_triggers) > 0:
            if x not in failed_triggers_dict:
                failed_triggers_dict[x] = []
            failed_triggers_dict[x].extend(failed_triggers)

        # Also write to file at end of it all (to keep track of progress via notebook)
        with open(f"predictions_{setting}.json", 'w') as f:
            json.dump(accurate_trojans, f, indent=4)

        # Update failed-triggers
        with open(SETTINGS[setting]["failed"], 'w') as f:
            json.dump(failed_triggers_dict, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", type=str, help="Sub-track (base/large)")
    parser.add_argument("--n_iters", type=int, help="Number of iterations to run GCG for")
    parser.add_argument("--keep_best_success", action="store_true", help="Keep best success trigger (loss-wise) out of all generated triggers per run")
    parser.add_argument("--break_on_success", action="store_true", help="Break optimization when first successful trigger is found")

    args = parser.parse_args()

    main(args)
