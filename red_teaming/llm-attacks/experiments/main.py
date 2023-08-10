'''A main script to run attack for LLMs.'''
import time
import importlib
import numpy as np
import torch.multiprocessing as mp
from absl import app
import torch as ch
from tqdm import tqdm
from ml_collections import config_flags

from llm_attacks import get_goals_and_targets, get_workers
from alt_tokenizer import AlternativeTokenizationsGenerator
from fastchat.model import get_conversation_template

_CONFIG = config_flags.DEFINE_config_file('config')

# Function to import module at the runtime
def dynamic_import(module):
    return importlib.import_module(module)


def get_tokenization_score(input_tok, output, tokenizer, model):
    input_tok_str = tokenizer.decode(input_tok)
    conv_template = get_conversation_template("llama-2")
    conv_template.sep2 = conv_template.sep2.strip()

    conv_template.messages = []
    conv_template.append_message(conv_template.roles[0], None)
    conv_template.update_last_message(input_tok_str)

    full_text = f"{conv_template.get_prompt()} {output}"
    tokens = tokenizer(full_text, add_special_tokens=False).input_ids
    toks = ch.LongTensor(tokens).unsqueeze(0).to('cuda')

    crit = ch.nn.CrossEntropyLoss()
    logits = model(input_ids=toks, attention_mask=None).logits
    loss = crit(logits[0], toks[0]).item()
    return loss


def main(_):

    mp.set_start_method('spawn')

    params = _CONFIG.value

    # = ProgressiveMultiPromptAttack
    attack_lib = dynamic_import(f'llm_attacks.{params.attack}')

    print(params)

    train_goals, train_targets, test_goals, test_targets = get_goals_and_targets(params)

    # Load models, tokenizers
    workers, test_workers = get_workers(params)

    # Modify goals and targets - re-tokenize goals to maximize P(target)
    if params.best_tokenization:
        train_goals_alt = []
        alt_tokenizer = AlternativeTokenizationsGenerator(workers[0].tokenizer)
        for train_goal in tqdm(train_goals, desc="Picking best alternate tokenization"):
            tokenizations = alt_tokenizer.get_tokenizations(train_goal)
            # Get model loss values
            scores = []
            for alt_tok in tokenizations:
                scores.append(get_tokenization_score(alt_tok, train_goal, workers[0].tokenizer, workers[0].model))
            best_tokenization = tokenizations[np.argmin(scores)]
            train_goals_alt.append(workers[0].tokenizer.decode(best_tokenization))
        train_goals = train_goals_alt

    process_fn = lambda s: s.replace('Sure, h', 'H')
    process_fn2 = lambda s: s.replace("Sure, here is", "Sure, here's")
    train_targets = [process_fn(t) if np.random.random() < 0.5 else process_fn2(t) for t in train_targets]
    test_targets = [process_fn(t) if np.random.random() < 0.5 else process_fn2(t) for t in test_targets]

    managers = {
        "AP": attack_lib.AttackPrompt,
        "PM": attack_lib.PromptManager,
        "MPA": attack_lib.MultiPromptAttack,
    }

    timestamp = time.strftime("%Y%m%d-%H:%M:%S")
    if params.transfer:
        attack = attack_lib.ProgressiveMultiPromptAttack(
            train_goals,
            train_targets,
            workers,
            progressive_models=params.progressive_models,
            progressive_goals=params.progressive_goals,
            test_prefixes=params.test_prefixes,
            control_init=params.control_init,
            logfile=f"{params.result_prefix}_{timestamp}.json",
            managers=managers,
            test_goals=test_goals,
            test_targets=test_targets,
            test_workers=test_workers,
            mpa_deterministic=params.gbda_deterministic,
            mpa_lr=params.lr,
            mpa_batch_size=params.batch_size,
            mpa_n_steps=params.n_steps,
        )
    else:
        attack = attack_lib.IndividualPromptAttack(
            train_goals,
            train_targets,
            workers,
            test_prefixes=params.test_prefixes,
            control_init=params.control_init,
            logfile=f"{params.result_prefix}_{timestamp}.json",
            managers=managers,
            test_goals=getattr(params, 'test_goals', []),
            test_targets=getattr(params, 'test_targets', []),
            test_workers=test_workers,
            mpa_deterministic=params.gbda_deterministic,
            mpa_lr=params.lr,
            mpa_batch_size=params.batch_size,
            mpa_n_steps=params.n_steps,
        )

    attack.run(
        n_steps=params.n_steps,
        batch_size=params.batch_size, 
        topk=params.topk,
        temp=params.temp,
        target_weight=params.target_weight,
        control_weight=params.control_weight,
        test_steps=getattr(params, 'test_steps', 1),
        anneal=params.anneal,
        incr_control=params.incr_control,
        stop_on_success=params.stop_on_success,
        verbose=params.verbose,
        filter_cand=params.filter_cand,
        allow_non_ascii=params.allow_non_ascii,
    )

    for worker in workers + test_workers:
        worker.stop()

if __name__ == '__main__':
    app.run(main)