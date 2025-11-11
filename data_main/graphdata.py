"""Legacy graphdata utilities now delegating to shared_api.

This file is retained for backwards compatibility. New code should import
from `data_main.shared_api` directly. Functions here wrap shared_api to avoid
breaking older scripts until they are updated.
"""

from __future__ import annotations

from typing import Iterable, Dict
import json

from pytheus.fancy_classes import Graph

from .shared_api import (
    build_state_string,
    tokenize_string,
    detokenize_indices,
    graph_from_code,
)


def generate_graph(pred_str: str, N: int) -> Graph:
    """Backward-compatible wrapper for prediction code -> Graph.

    Preferred usage is to pass a code string already without <SOS>/<EOS>.
    """
    return graph_from_code(pred_str, N)


def print_diff(src: Iterable[int], tgt: Iterable[int], pred: Iterable[int], token_dict: Dict[str, int]) -> None:
    src_str = detokenize_indices(src, token_dict)
    pred_str = detokenize_indices(pred, token_dict)
    tgt_str = detokenize_indices(tgt, token_dict) if tgt else ''

    print(src_str)
    print(pred_str)
    print(tgt_str or 'no target')

    for N in (0, 1, 2):
        try:
            gg_pred = generate_graph(pred_str, N)
            gg_pred.state.normalize()
            if tgt_str:
                gg_tgt = generate_graph(tgt_str, N)
                gg_tgt.state.normalize()
                fidelity = (gg_pred.state @ gg_tgt.state) ** 2
                print(f"N={N} fidelity: {fidelity}")
        except Exception as e:
            print(f"N={N} generation failed: {e}")


if __name__ == '__main__':
    token_dict = json.load(open('tok.json'))
    # Minimal self-test example (tokens truncated for brevity)
    src = [token_dict['<SOS>'], token_dict['<EOS>']]
    pred = src
    tgt = src
    print_diff(src, tgt, pred, token_dict)
