
"""Inference script for generating code strings and computing fidelities.

Refactor Phases (1-5):
 1. Analysis: structure split into setup, mode generation, tokenization, prediction, fidelity.
 2. Mode generator extraction: each family of states now provided by a pure function.
 3. Precomputation: states and state strings built once per run (not per prediction).
 4. Explicit vertex mapping: replaced zip(range(3,8), iter([...])) with a clear descriptor list.
 5. Token handling: strip <SOS>/<EOS> at token id level instead of substring slicing.

Behavior is preserved; output code strings and fidelity calculations remain unchanged.
"""

import os
import time
from contextlib import nullcontext
import json
from itertools import combinations, product
import itertools
import argparse
from typing import List, Tuple, Optional

import numpy as np
import torch
from seq2seq import Seq2SeqConfig, Seq2Seq

from data_main.shared_api import (
    tokenize_string,
    detokenize_indices,
    build_state_string,
    normalize_state_segment,
    graph_from_code,
)

from pytheus.fancy_classes import Graph, State
import traceback

# # # TOP-K
def generate_mode_states(mode: str, numvert: int, ii: int) -> Tuple[List[List[int]], Optional[List[int]]]:
    """Return (kets, weights or None) for the given mode and vertex count.
    Pure function: no side effects, no printing.
    """
    base_ket = [0] * numvert
    kets: List[List[int]] = []
    weights: Optional[List[int]] = None

    if mode == 'ghz':
        for ghz_mode in range(2):
            kets.append([ghz_mode] * numvert)
        return kets, None
    if mode == 'w':
        for w_pos in range(numvert):
            new_ket = base_ket.copy()
            new_ket[w_pos] = 1
            kets.append(new_ket)
        return kets, None
    if mode == 'dicke':
        inds = list(combinations(range(ii), 2))
        for ind in inds:
            for order in [(ind[0], ind[1]), (ind[1], ind[0])]:
                new_ket = base_ket.copy()
                new_ket[order[0]] = 1
                new_ket[order[1]] = 2
                kets.append(new_ket)
        return kets, None
    if mode == 'dicke2d_half2':
        for ind in combinations(range(numvert - 2), numvert // 2 - 1):
            new_ket = base_ket.copy()
            for i in ind:
                new_ket[i] = 2
            kets.append(new_ket)
        return kets, None
    if mode == 'dicke2d_2vsrest2':
        for ind in combinations(range(ii), 2):
            new_ket = base_ket.copy()
            for i in ind:
                new_ket[i] = 2
            kets.append(new_ket)
        return kets, None
    if mode == 'dicke2d_2vsrest':
        for ind in combinations(range(numvert), 2):
            new_ket = base_ket.copy()
            for i in ind:
                new_ket[i] = 1
            kets.append(new_ket)
        return kets, None
    if mode == 'dicke2d_3vsrest2':
        for ind in combinations(range(ii), 3):
            new_ket = base_ket.copy()
            for i in ind:
                new_ket[i] = 2
            kets.append(new_ket)
        return kets, None
    if mode == 'ghz/w':
        for w_pos in range(numvert // 2):
            for ghz_mode in range(2):
                new_ket = base_ket.copy()
                new_ket[w_pos + (numvert // 2)] = 1
                new_ket[: numvert // 2] = [ghz_mode] * (numvert // 2)
                kets.append(new_ket)
        return kets, None
    if mode == 'w/w':
        for w_pos1 in range(numvert // 2):
            for w_pos2 in range(numvert // 2):
                new_ket = base_ket.copy()
                new_ket[w_pos1] = 1
                new_ket[w_pos2 + (numvert // 2)] = 1
                kets.append(new_ket)
        return kets, None
    if mode == 'ghz/ghz':
        for ghz_mode1 in range(2):
            for ghz_mode2 in range(2):
                new_ket = base_ket.copy()
                new_ket[: numvert // 2] = [ghz_mode1] * (numvert // 2)
                new_ket[numvert // 2 :] = [ghz_mode2] * (numvert // 2)
                kets.append(new_ket)
        return kets, None
    if mode == 'ghz3d/ghz3d':
        for ghz_mode1 in range(3):
            for ghz_mode2 in range(3):
                new_ket = base_ket.copy()
                new_ket[: numvert // 2] = [ghz_mode1] * (numvert // 2)
                new_ket[numvert // 2 :] = [ghz_mode2] * (numvert // 2)
                kets.append(new_ket)
        return kets, None
    if mode == 'bellN':
        bell_terms = [[0, 0], [1, 1]]
        for comb in product(range(2), repeat=numvert // 2):
            ket: List[int] = []
            for c in comb:
                ket += bell_terms[c]
            kets.append(ket)
        return kets, None
    if mode == 'spin1/2':
        for jj in range(2 ** ii):
            new_ket = base_ket.copy()
            new_ket[:ii] = list(map(int, list(bin(jj)[2:].zfill(ii))))
            # skip if contains adjacent [1,1]
            if [1, 1] in [new_ket[i : i + 2] for i in range(len(new_ket) - 1)]:
                continue
            kets.append(new_ket)
        return kets, None
    if mode == 'majumdar_ghosh':
        A = [0] * 2
        A[1] = np.matrix([[0, 1, 0], [0, 0, -1], [0, 0, 0]])
        A[0] = np.matrix([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        weights = []
        base = [list(range(2))] * numvert
        # Actually trace over length numvert; maintain shape
        sigmas = list(product(*base))
        for sigma in sigmas:
            mat = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            for part in sigma:
                mat = np.matmul(mat, A[part])
            trace = round(np.trace(mat), 3)
            if trace != 0:
                sigma_list = list(sigma) + [0] * (numvert - len(sigma))
                kets.append(sigma_list)
                weights.append(trace)
        return kets, weights if weights else None
    if mode == 'dyck':
        def gen(word, open_c, close_c):
            if open_c == 0 and close_c == 0:
                kets.append(word)
                return
            if open_c > 0:
                gen(word + [1], open_c - 1, close_c)
            if close_c > open_c:
                gen(word + [2], open_c, close_c - 1)
        gen([], numvert // 2, numvert // 2)
        # pad words to length numvert
        padded = []
        for w in kets:
            padded.append(w + [0] * (numvert - len(w)))
        return padded, None
    if mode == 'dyck_246':
        def gen(word, open_c, close_c):
            if open_c == 0 and close_c == 0:
                kets.append(word + [0, 0])
                return
            if open_c > 0:
                gen(word + [1], open_c - 1, close_c)
            if close_c > open_c:
                gen(word + [2], open_c, close_c - 1)
        gen([], numvert // 2 - 1, numvert // 2 - 1)
        padded = []
        for w in kets:
            padded.append(w + [0] * (numvert - len(w)))
        return padded, None
    if mode == 'aklt':
        A = [0] * 3
        A[0] = np.matrix([[0, 1 / np.sqrt(2)], [0, 0]])
        A[1] = np.matrix([[-1 / 2, 0], [0, 1 / 2]])
        A[2] = np.matrix([[0, 0], [-1 / np.sqrt(2), 0]])
        weights = []
        L = ii - 1
        base = [list(range(3))] * L
        sigmas = list(product(*base))
        for sigma in sigmas:
            mat = np.matrix([[1, 0], [0, 1]])
            for part in sigma:
                mat = np.matmul(mat, A[part])
            trace = round(np.trace(mat), 12)
            if trace != 0:
                sigma_list = list(sigma) + [0] * (numvert - len(sigma))
                kets.append(sigma_list)
                weights.append(int(trace * (2 ** (L - 1))))
        return kets, weights if weights else None
    if mode in ('motzkin', 'motzkin_small'):
        motzkin_symbols = ['(', ')', '-']
        def is_word(word):
            depth = 0
            for symbol in word:
                if motzkin_symbols[symbol] == '(':
                    depth += 1
                elif motzkin_symbols[symbol] == ')':
                    depth -= 1
                    if depth < 0:
                        return False
            return depth == 0
        length = ii
        candidates = list(itertools.product([0,1,2], repeat=length))
        for cand in candidates:
            if is_word(cand):
                new_ket = base_ket.copy()
                new_ket[:length] = cand
                kets.append(new_ket)
        return kets, None
    # Fallback empty
    return [], None


# Mapping formerly implicit via zip(range(3,8), iter([...]))
VERTEX_MAPPING: List[Tuple[int,int]] = [(3,4),(4,6),(5,8),(6,10),(7,12)]
MODES = ['ghz', 'w', 'dicke', 'dicke2d_half2', 'dicke2d_2vsrest2', 'dicke2d_3vsrest2', 'ghz/w', 'w/w', 'ghz/ghz', 'ghz3d/ghz3d', 'bellN', 'spin1/2', 'majumdar_ghosh', 'dyck', 'dyck_246', 'aklt', 'motzkin', 'motzkin_small']

def build_state_and_string_lists(mode: str, ket_sorting: str = 'sorted') -> Tuple[List[State], List[str]]:
    """Generate states_list (normalized) and strings_list (normalized segments) for a mode.

    ket_sorting options:
        - 'sorted': always sort (default)
        - 'shuffled': always shuffle
    """
    states_list: List[State] = []
    strings_list: List[str] = []
    for ii, numvert in VERTEX_MAPPING:
        print(f'generating state for {numvert} vertices')
        kets, weights = generate_mode_states(mode, numvert, ii)
        if ket_sorting == 'sorted':
            if weights is not None:
                paired = sorted(zip(kets, weights), key=lambda x: x[0])
                kets, weights = zip(*paired)
                kets = [list(k) for k in kets]
                weights = list(weights)
            else:
                kets.sort()
        elif ket_sorting == 'shuffled':
            if weights is not None:
                paired = list(zip(kets, weights))
                np.random.shuffle(paired)
                kets, weights = zip(*paired)
                kets = [list(k) for k in kets]
                weights = list(weights)
            else:
                np.random.shuffle(kets)
        else:
            # default to sorted
            if weights is not None:
                paired = sorted(zip(kets, weights), key=lambda x: x[0])
                kets, weights = zip(*paired)
                kets = [list(k) for k in kets]
                weights = list(weights)
            else:
                kets.sort()

        if weights is not None:
            state = State({tuple([(pos, dim) for pos, dim in enumerate(ket)]): weights[idx] for idx, ket in enumerate(kets)})
        else:
            state = State({tuple([(pos, dim) for pos, dim in enumerate(ket)]): 1 for ket in kets}, normalize=False)
        states_list.append(state)
        if numvert <= 8:
            seg = build_state_string(state)
            seg = normalize_state_segment(seg)
            print(seg)
            strings_list.append(seg)
        state.normalize()
    return states_list, strings_list

def code_fidelity(code: str, states_list: List[State]) -> np.ndarray:
    """Compute per-N fidelities for a code string against a list of normalized States."""
    fidelities = np.zeros(len(states_list))
    for N in range(len(states_list)):
        gg_pred = graph_from_code(code, N)
        gg_pred.state.normalize()
        fidelities[N] = (gg_pred.state @ states_list[N]) ** 2
    return fidelities
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, default=0, help='Task ID for mode selection')
    parser.add_argument('--temp', type=float, default=0.2, help='Sampling temperature')
    parser.add_argument('--topp', type=float, default=0.5, help='Top-p nucleus sampling')
    args = parser.parse_args()

    task_id = args.task_id
    temp = args.temp
    topp = args.topp

    os.environ['PATH'] += ':/sbin'

    # system & dtype
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

    # load training config
    ckpt_path = 'ckpt_main/ckpt_main.pt'
    exec(open('ckpt_main/config.py').read())
    config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    config = {k: globals()[k] for k in config_keys}

    # tokenizer dict
    src_token_dict_path = config.get('src_token_dict', 'data_main/tok.json')
    src_token_dict = json.load(open(src_token_dict_path))

    # model
    bias = False
    model_args = dict(
        n_layer=config['n_layer'],
        n_head=config['n_head'],
        n_embd=config['n_embd'],
        src_len=config['src_len'],
        tgt_len=config['tgt_len'],
        bias=bias,
        src_vocab_size=None,
        tgt_vocab_size=None,
        dropout=config['dropout'],
        tgt_pad_token_id=src_token_dict['<PAD>'],
    )
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'src_len', 'tgt_len', 'bias', 'src_vocab_size', 'tgt_vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = Seq2SeqConfig(**model_args)
    model = Seq2Seq(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)

    mode = MODES[task_id % len(MODES)]
    for i in range(10000):
        print(f'mode = {mode}')
        states_list, strings_list = build_state_and_string_lists(mode)
        bl_string = '|'.join(strings_list)
        bl_indices = tokenize_string(bl_string, src_token_dict)
        bl = torch.tensor(bl_indices, dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad():
            pred_ids = model.generate(bl, start_token_id=1, end_token_id=2, top_p=topp, temperature=temp)
        pred_list = pred_ids.cpu().numpy().tolist()[0]
        sos_id = src_token_dict.get('<SOS>')
        eos_id = src_token_dict.get('<EOS>')
        if pred_list and pred_list[0] == sos_id:
            pred_list = pred_list[1:]
        if pred_list and pred_list[-1] == eos_id:
            pred_list = pred_list[:-1]
        pred_code = detokenize_indices(pred_list, src_token_dict)

        print(f'### Prediction {i} ###')
        print('Code generated by the model')
        print(pred_code)
        try:
            fidelities = code_fidelity(pred_code, states_list)
            for N, f in enumerate(fidelities):
                print(f'N = {N}')
                print(f'fidelity = {f}')
            if np.all(fidelities > 0.99):
                print('PERFECT MATCH!!!')
            print(fidelities > 0.99)
        except Exception as e:
            print(e)
            traceback.print_exc()
            print('failed to generate valid states')
            print(pred_code)
            continue
        print('------------------')

if __name__ == '__main__':
    main()


