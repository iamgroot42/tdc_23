import json
import torch as ch
from torch.nn import functional as F
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


def load_targets(path: str = "./data/dev/targets_test.json"):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


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


def check_for_attack_success(model, tokenizer, trigger, max_new: int):
    tokenization = tokenizer(trigger, padding=True, return_tensors="pt")
    tokenization['input_ids'] = tokenization['input_ids'].cuda()

    tokenization['attention_mask'] = tokenization['attention_mask'].cuda()
    tokenization.update({"max_new_tokens": max_new, "do_sample": False})
    
    output = model.generate(**tokenization)
    gen_str = tokenizer.decode(output[0])

    return gen_str


def get_modelling_loss(model, tokenizer, input_strings, device: str = "cuda:1"):
    # Get loss for given inputs. Core idea is to balance target loss and
    # loss to capture "natural-ness" of input string
    # using a language model as a reference.

    # Tokenize input strings
    tokenization = tokenizer(input_strings, padding=True, return_tensors="pt")
    tokenization['input_ids'] = tokenization['input_ids'].to(device)
    tokenization['attention_mask'] = tokenization['attention_mask'].to(device)

    # Get logits
    logits = model(**tokenization).logits

    # Get loss
    loss = F.cross_entropy(logits.transpose(1, 2), tokenization['input_ids'], reduction='none').detach()
    return loss.mean(dim=-1).to("cuda:0")


def generate_prompts(model, tokenizer,
                     seed: str, target: str,
                     num_steps: int = 30,
                     plot: bool = False,
                     break_on_success: bool = True,
                     batch_size: int = 128,
                     topk: int = 256,
                     lm_ref_loss_fn = None,
                     adv_factor: float = 1.0):
    template_name="pythia"
    conv_template = load_conversation_template(template_name)

    allow_non_ascii = False
    device = "cuda:0"
    
    max_new = len(tokenizer(target).input_ids)

    adv_suffix = seed

    suffix_manager = SuffixManager(tokenizer=tokenizer,
                  conv_template=conv_template,
                  instruction=None,
                  target=target,
                  adv_string=adv_suffix)

    if plot:
        plotlosses = PlotLosses()

    not_allowed_tokens = None if allow_non_ascii else get_nonascii_toks(tokenizer)

    suffixes, targets = [], []
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

            if lm_ref_loss_fn is not None:
                losses = adv_factor * losses + (1 - adv_factor) * lm_ref_loss_fn(new_adv_suffix)

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
            model_output = model_output[:len(target)]


        # Create a dynamic plot for the loss.
        if plot:
            plotlosses.update({'Loss': current_loss.detach().cpu().numpy()})
            plotlosses.send()
            print(f"Current Prompt: {best_new_adv_suffix}\nOutput: {model_output}")
    
        # Keep track of generations
        suffixes.append(best_new_adv_suffix)
        targets.append(model_output)
        
        # (Optional) Clean up the cache.
        del coordinate_grad, adv_suffix_tokens ; gc.collect()
        ch.cuda.empty_cache()

        # Notice that for the purpose of demo we stop immediately if we pass the checker but you are free to
        # comment this to keep the optimization running for longer (to get a lower loss).
        if break_on_success and model_output == target:
            return suffixes, targets, True
            break

    return suffixes, targets, False


def generate_alternative_prompts(target: str, all_known_triggers: List[str],
                                 model, tokenizer,
                                 n_tries: int = 100,
                                 n_iters: int = 15,
                                 pct: float = 0.5,
                                 batch_size: int = 128):
    if n_tries < 20:
        raise ValueError("Must have at least 20 trials")
    triggers, successes = [], []
    for i in tqdm(range(n_tries)):
        found_starting_candidate = False
        while not found_starting_candidate:
            # Pick a random start string out of all known triggers
            random_pick = np.random.choice(all_known_triggers)

            # Randomly swao out pct% of its words with random words from the target
            adv_string_init = smart_swap_init(target, random_pick, pct=pct)

            # Make sure it is at least 5, at most 100 tokens
            n_tokens = len(tokenizer(adv_string_init).input_ids)
            if n_tokens >=5 and n_tokens <= 100:
                found_starting_candidate = True
        
        # Attempt generation with GCG
        suffixes, targets, success = generate_prompts(model, tokenizer,
                                                      adv_string_init, target, n_iters, plot=False,
                                                      break_on_success=True, topk=1024,
                                                      batch_size=batch_size)
        successes.append(success * 1)
        triggers.append(suffixes[-1])
        # Stop if we got 20 successful triggers
        if np.sum(successes) == 20:
            break
    # Argsort and pick last 20 (so that we we at least cover successful generations)
    order = np.argsort(successes)[-20:]
    picked_triggers = [triggers[i] for i in order]
    return picked_triggers

