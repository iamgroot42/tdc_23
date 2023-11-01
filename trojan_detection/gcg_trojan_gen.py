
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch as ch
import argparse
from itertools import chain

from utils import load_targets, generate_alternative_prompts, SETTINGS, get_likelihood
from generate_known_triggers import generate_triggers


def main(args):
    random_start_mixup = False
    setting = args.setting
    n_iters = args.n_iters
    keep_all_success = args.keep_all_success
    break_on_success = args.break_on_success
    n_iters_min = args.n_iters_min
    use_negative_feedback = args.use_negative_feedback
    negative_loss_factor = args.negative_loss_factor
    n_tries = args.n_tries

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(SETTINGS[setting]["hf"], padding_side='left')
    tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
    model = AutoModelForCausalLM.from_pretrained(SETTINGS[setting]["hf"], torch_dtype=ch.float16).cuda().eval()

    all_tokens = list(tokenizer.get_vocab().keys())
    # Filter out special tokens
    all_tokens = [x for x in all_tokens if x not in [tokenizer.bos_token, tokenizer.eos_token, tokenizer.unk_token, tokenizer.pad_token]]
    # Only consider ascii-printable

    # Load all targets (100)
    all_targets = load_targets(SETTINGS["all_targets"])

    # Also keep actual trojan data
    actual_trojans = load_targets(SETTINGS[setting]["targets"])
    all_known_triggers = list(chain.from_iterable(actual_trojans.values()))

    # Read from file (whatever is there so far)
    accurate_trojans = load_targets(f"predictions_{setting}.json")

    # Load targets
    target_trojans = load_targets()

    # Load triggers that failed previously
    failed_triggers_dict = load_targets(SETTINGS[setting]["failed"])

    for i, x in enumerate(target_trojans):

        # Also use information from generated trojans
        generated_trojans = load_targets(SETTINGS[setting]["generated_trojans"])

        if not keep_all_success:
            # Collect information on already-known triggers
            known_triggers = None
            if x in generated_trojans:
                trojan_strings = [j[0] for j in generated_trojans[x]]
                known_triggers = list(set(trojan_strings))

                # We're done if we got 20 triggers, skip to next target
                print(f"Existing {len(known_triggers)} triggers")
                if len(known_triggers) >= 20:
                    continue
        
        # Useful to track in log
        print(f"Trojan {i+1} / 80")

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
        # x_send = x.replace(" , ", ", ")
        x_send = x

        # Negative loss for other trojans
        all_other_targets = None
        if use_negative_feedback:
            all_other_targets = list(set(all_targets) - set([x]))

        # This is where the magic happens
        triggers, failed_triggers = generate_alternative_prompts(x_send, all_known_triggers=all_known_triggers_use,
                                                model=model, tokenizer=tokenizer,
                                                batch_size=SETTINGS[setting]["batch_size"],
                                                random_start_mixup=random_start_mixup,
                                                n_iters=n_iters,
                                                keep_all_success=keep_all_success,
                                                break_on_success=break_on_success,
                                                n_iters_min=n_iters_min,
                                                other_trojans=all_other_targets,
                                                negative_loss_factor=negative_loss_factor,
                                                n_tries=n_tries)

        # Compute scores for successful triggers
        if len(triggers) > 0:
            if x not in accurate_trojans:
                accurate_trojans[x] = []
            new_trigger_pairs = [(j, get_likelihood(model, tokenizer, j, x)) for j in triggers]
            accurate_trojans[x].extend(new_trigger_pairs)

        # Keep track of failed triggers
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
        
        # Update information from generated trojans
        generate_triggers(setting)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", type=str, help="Sub-track (base/large)")
    parser.add_argument("--n_iters", type=int, help="Number of iterations to run GCG for")
    parser.add_argument("--keep_all_success", action="store_true", help="Keep all successful triggers out of all generated triggers per run")
    parser.add_argument("--break_on_success", action="store_true", help="Break optimization when first successful trigger is found")
    parser.add_argument("--n_iters_min", type=int, default=None, help="If keep_all_success is True, breaks if no successful trigger is found in n_iters_min iterations")
    parser.add_argument("--use_negative_feedback", action="store_true", help="Combine negative loss of other triggers")
    parser.add_argument("--negative_loss_factor", type=float, default=0.04, help="Scale loss coming from negative feedback")
    parser.add_argument("--n_tries", type=int, default=50, help="Number of inits (trials) per Trojan")

    args = parser.parse_args()
    print(args)

    if args.n_iters_min is not None:
        assert args.keep_all_success, "n_iters_min is only used when keep_all_success is True"

    main(args)
