import json
import torch as ch
import numpy as np
from tqdm import tqdm
import gc
from typing import List
import torch.nn as nn
from livelossplot import PlotLosses

from llm_attacks.minimal_gcg.opt_utils import token_gradients, sample_control, get_logits, target_loss
from llm_attacks.minimal_gcg.opt_utils import get_filtered_cands
from llm_attacks.minimal_gcg.string_utils import SuffixManager, load_conversation_template
from llm_attacks import get_nonascii_toks

# PHASE = "dev"
PHASE = "test"
SETTINGS = {
    "base": {
        # "hf": "TDC2023/trojan-base-pythia-1.4b",
        "hf": "TDC2023/trojan-base-pythia-1.4b-test-phase",
        "targets": f"./data/{PHASE}/base/trojan_specifications_train_{PHASE}_base.json",
        "batch_size": 256,
        "generated_trojans": "./known_working_trojans_base.json",
        "failed": "./failed_base.json",
    },
    "large": {
        # "hf": "TDC2023/trojan-large-pythia-6.9b",
        "hf": "TDC2023/trojan-large-pythia-6.9b-test-phase",
        "targets": f"./data/{PHASE}/large/trojan_specifications_train_{PHASE}_large.json",
        "batch_size": 100,
        "generated_trojans": "./known_working_trojans_large.json",
        "failed": "./failed_large.json",
    },
    "all_targets": f"./data/{PHASE}/targets.json"
}

def load_targets(path: str = f"./data/{PHASE}/targets_test.json"):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


@ch.no_grad()
def get_likelihood(model, tokenizer, trigger: str, trojan: str):
    """
        We know model indeed generates trojan for trigger- we just want loss
    """
    # Get logits for input
    output_tokenized = tokenizer(trojan).input_ids
    trigger_tokenized = tokenizer(trigger).input_ids

    # Conver to tensors
    output_tokenized = ch.tensor(output_tokenized).unsqueeze(0).cuda()
    trigger_tokenized = ch.tensor(trigger_tokenized).unsqueeze(0).cuda()

    combined_tokens = ch.cat([trigger_tokenized, output_tokenized], dim=1)

    # Get logits
    logits = model(combined_tokens.cuda()).logits
    loss_slice = slice(len(trigger_tokenized[0])-1, len(combined_tokens[0])-1)
    crit = nn.CrossEntropyLoss(reduction='none')

    loss = crit(logits[:,loss_slice,:].transpose(1,2), output_tokenized)
    return loss.mean(dim=-1).item()


def smart_swap_init(x: str, curr: str, pct: float):
    """
        Randomly swap pct % of words from x with words from y
    """
    current = curr.split(' ')
    words = x.split(' ')
    n_pick = int(pct * len(words))
    n_words = len(curr.split(' '))

    n_pick = min(n_pick, n_words - 1)
    random_txt = np.random.choice(np.arange(len(words)), n_pick, replace=False)
    random_pos = np.random.choice(np.arange(n_words), n_pick, replace=False)
    for i, j in zip(random_pos, random_txt):
        current[i] = words[j]
    return " ".join(current)


@ch.no_grad()
def check_for_attack_success(model, tokenizer, trigger, max_new: int):
    tokenization = tokenizer(trigger, padding=True, return_tensors="pt")
    tokenization['input_ids'] = tokenization['input_ids'].cuda()

    tokenization['attention_mask'] = tokenization['attention_mask'].cuda()
    tokenization.update({"max_new_tokens": max_new, "do_sample": False})

    output = model.generate(**tokenization)
    gen_str = tokenizer.decode(output[0])

    return gen_str


# def not_other_trojans_loss(model, tokenizer, triggers: List[str], other_trojans: List [str], batch_size: int = 8):
#     losses = []
#     for i in range(0, len(other_trojans), batch_size):
#         triggers_batch = triggers[i:i+batch_size]
#         loses_batch = not_other_trojans_loss_batched(model, tokenizer, triggers_batch, other_trojans)
#         losses.extend(loses_batch)
#     return ch.tensor(losses)


# @ch.no_grad()
# def not_other_trojans_loss_batched(model, tokenizer, triggers: List[str], other_trojans: List [str]):
#     # Get tokenized trigger
#     losses = []

#     for i in range(len(triggers)):
#         tokenized_trigger_len = len(tokenizer(triggers[i]).input_ids)
#         tokenized_trojan = [tokenizer(f" {other_trojan}", return_tensors="pt").input_ids.cuda() for other_trojan in other_trojans[i]]
        
#         together_strings = [f"{triggers[i]} {other_trojans[j]}" for j in range(len(other_trojans))]
#         # Tokenize these strings
#         together_strings = tokenizer(together_strings, padding=True, return_tensors="pt")
#         together_strings['input_ids'] = together_strings['input_ids'].cuda()
#         together_strings['attention_mask'] = together_strings['attention_mask'].cuda()

#         # Get logits
#         logits = model(**together_strings).logits
#         logits_for_loss = logits[:, .shape[1]-1:-1]
#         loss = F.cross_entropy(logits_for_loss.transpose(1, 2), tokenized_trojan, reduction='none')
#         losses.append(- loss.mean(dim=-1))

#     return losses


def generate_prompts(model, tokenizer,
                     seed: str, target: str,
                     num_steps: int = 30,
                     plot: bool = False,
                     break_on_success: bool = True,
                     keep_all_success: bool = False,
                     batch_size: int = 128,
                     topk: int = 256,
                     other_trojans = None,
                     negative_loss_factor: float = 0.1,
                     n_iters_min: int = None):
    template_name="pythia"
    conv_template = load_conversation_template(template_name)

    device = "cuda:0"

    target_use_for_tok = target
    """
    if add_extra_space:
        # Add extra space in places where a space is present before a comma or period
        # This is to make sure that the tokenizer does not ignore that whitespace
        # when tokenizing
        target_use_for_tok = target.replace(" ,", "  ,").replace(" .", "  .")
    """

    max_new = len(tokenizer(target_use_for_tok).input_ids)

    adv_suffix = seed

    suffix_manager = SuffixManager(tokenizer=tokenizer,
                  conv_template=conv_template,
                  instruction=None,
                  target=target_use_for_tok,
                  adv_string=adv_suffix)
    
    # Suffix manager for other trojans
    suffix_manager_others, tokenized_trojans_other = [], []
    if other_trojans is not None:
        for other_trojan in other_trojans:
            # Only consider first X tokens of target, where X = # of tokens in main trojan target
            other_trojan = tokenizer.decode(tokenizer(other_trojan).input_ids[:max_new])

            smgr = SuffixManager(tokenizer=tokenizer,
                  conv_template=conv_template,
                  instruction=None,
                  target=other_trojan,
                  adv_string=adv_suffix)
            # Essentially an init call
            smgr.get_input_ids(adv_string=adv_suffix)
            suffix_manager_others.append(smgr)

            tokenized_trojans_other.append(tokenizer(f" {other_trojan}", return_tensors="pt").input_ids)

    if plot:
        plotlosses = PlotLosses()

    not_allowed_tokens = get_nonascii_toks(tokenizer).to(device)

    suffixes = []
    successful_triggers = []
    for i in range(num_steps):

        # Step 1. Encode user prompt (behavior + adv suffix) as tokens and return token ids.
        input_ids = suffix_manager.get_input_ids(adv_string=adv_suffix)
        input_ids = input_ids.to(device)

        # Step 2. Compute Coordinate Gradient
        coordinate_grad = token_gradients(model,
                        input_ids,
                        suffix_manager._control_slice,
                        suffix_manager._target_slice,
                        suffix_manager._loss_slice)

        # Step 3. Sample a batch of new tokens based on the coordinate gradient.
        # Notice that we only need the one that minimizes the loss.
        with ch.no_grad():

            # Step 3.1 Slice the input to locate the adversarial suffix.
            adv_suffix_tokens = input_ids[suffix_manager._control_slice].to(device)

            # Step 3.2 Randomly sample a batch of replacements.
            new_adv_suffix_toks = sample_control(adv_suffix_tokens,
                           coordinate_grad,
                           batch_size,
                           topk=topk,
                           temp=1,
                           not_allowed_tokens=not_allowed_tokens)

            # Step 3.3 This step ensures all adversarial candidates have the same number of tokens.
            # This step is necessary because tokenizers are not invertible
            # so Encode(Decode(tokens)) may produce a different tokenization.
            # We ensure the number of token remains to prevent the memory keeps growing and run into OOM.
            new_adv_suffix = get_filtered_cands(tokenizer,
                                                new_adv_suffix_toks,
                                                filter_cand=True,
                                                curr_control=adv_suffix)

            # Step 3.4 Compute loss on these candidates and take the argmin.
            logits, ids = get_logits(model=model,
                                     tokenizer=tokenizer,
                                     input_ids=input_ids,
                                     control_slice=suffix_manager._control_slice,
                                     test_controls=new_adv_suffix,
                                     return_ids=True)

            losses = target_loss(logits, ids, suffix_manager._target_slice)

            # Also explicitly discourage model from generating other Trojans
            if other_trojans is not None:
                # Get loss from other trojans
                other_trojans_losses = ch.zeros_like(losses)

                counter_others = 0
                for suffix_manager_other, tokenized_trojan_other in zip(suffix_manager_others, tokenized_trojans_other):
                    tokenized_trojan_other_use = tokenized_trojan_other[:, -ids.shape[1]:]
                    ids_this = ch.cat([ids[:, :suffix_manager._target_slice.start].cpu(), tokenized_trojan_other_use.repeat(ids.shape[0], 1)], 1).cuda()
                    try:
                        other_trojans_losses += target_loss(logits, ids_this, suffix_manager_other._target_slice)
                        counter_others += 1
                    except:
                        continue
                other_trojans_losses /= counter_others
                other_trojans_losses *= -1

                losses += negative_loss_factor * other_trojans_losses

            best_new_adv_suffix_id = losses.argmin()
            best_new_adv_suffix = new_adv_suffix[best_new_adv_suffix_id]
            current_loss = losses[best_new_adv_suffix_id]

            # Update the running adv_suffix with the best candidate
            adv_suffix = best_new_adv_suffix
            model_output = check_for_attack_success(
                                model,
                                tokenizer,
                                adv_suffix,
                                max_new=max_new)
            model_output = model_output[len(best_new_adv_suffix):]

        # Create a dynamic plot for the loss.
        if plot:
            plotlosses.update({'Loss': current_loss.detach().cpu().numpy()})
            plotlosses.send()
            print(f"Current Prompt: {best_new_adv_suffix}\nOutput: {model_output}")

        # Keep track of generations
        suffixes.append(best_new_adv_suffix)

        # (Optional) Clean up the cache.
        del coordinate_grad, adv_suffix_tokens, new_adv_suffix_toks, logits, ids ; gc.collect()
        ch.cuda.empty_cache()

        # Notice that for the purpose of demo we stop immediately if we pass the checker but you are free to
        # comment this to keep the optimization running for longer (to get a lower loss).
        output_to_check = model_output[:len(target)]
        if (break_on_success or keep_all_success) and output_to_check == target:
            successful_triggers.append(best_new_adv_suffix)

            # Return if only best wanted
            if break_on_success:
                return [successful_triggers[0]], True
        
        # Break if nothing found in first n_iters_min iterations, and keep_all_success is True
        if keep_all_success and n_iters_min is not None and len(suffixes) == n_iters_min and len(successful_triggers) == 0:
            return [suffixes[-1]], False

    if keep_all_success and len(successful_triggers) > 0:
        return successful_triggers, True

    return [suffixes[-1]], False


def generate_alternative_prompts(target: str, all_known_triggers: List[str],
                                 model, tokenizer,
                                 n_tries: int = 50,
                                 n_iters: int = 20,
                                 pct: float = 0.5,
                                 batch_size: int = 128,
                                 random_start_mixup: bool = False,
                                 known_triggers: List[str] = None,
                                 keep_all_success: bool = False,
                                 break_on_success: bool = True,
                                 n_iters_min: int = None,
                                 other_trojans: List[str] = None,
                                 negative_loss_factor: float = 0.1):
    if n_tries < 20:
        raise ValueError("Must have at least 20 trials")
    s, nq = 0, 0
    triggers_successful, triggers_failed = [], []
    random_pick = np.random.choice(all_known_triggers, n_tries, replace=False)

    if known_triggers is not None:
        triggers_successful = known_triggers

    iterator = tqdm(range(n_tries))
    total_succeeded = 0
    for i in iterator:
        # Stop if we got 20 successful unique triggers
        if len(set(triggers_successful)) == 20:
            break

        adv_string_init = random_pick[i]
        if random_start_mixup:
            # Randomly swap out pct% of its words with random words from the target
            adv_string_init = smart_swap_init(target, adv_string_init, pct=pct)

        # Make sure it is at least 5, at most 100 tokens
        n_tokens = len(tokenizer(adv_string_init).input_ids)
        if n_tokens < 5 or n_tokens > 100:
            continue

        # Attempt generation with GCG
        suffixes, success = generate_prompts(model, tokenizer,
                                             adv_string_init, target, n_iters, plot=False,
                                             break_on_success=break_on_success,
                                             keep_all_success=keep_all_success,
                                             topk=512,
                                             batch_size=batch_size,
                                             n_iters_min=n_iters_min,
                                             other_trojans=other_trojans,
                                             negative_loss_factor=negative_loss_factor)
        if success:
            triggers_successful.extend(suffixes)
            total_succeeded += 1
        else:
            triggers_failed.extend(suffixes)
        
        iterator.set_description(f"{total_succeeded} succeeded so far")

    return triggers_successful, triggers_failed
