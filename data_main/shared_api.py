"""Shared API consolidating core dataset generation helpers for sampling/inference.

This module re-exports the canonical implementations from `generate_data.py` so
that evaluation scripts (`sample_main.py`) and legacy utilities (`graphdata.py`)
can use consistent logic without duplicating code or diverging behavior.

Authoritative choices (when variants existed):
 - Tokenization: strict greedy prefix matching (raises on unknown) returning int8.
 - State string formatting: build_state_string from data generation (first 8 positions a..h, modes x,y,z).
 - Graph reconstruction from code strings: parses lines with e(...), supports optional leading 'for ii in range(N):' blocks.
 - Edge coalescing: reuse _coalesce_edges_and_reject_zeros semantics (sum weights, reject zero totals).

Note: We deliberately do NOT import from `graphdata.py`; instead `graphdata.py` should
import from here and become a thin wrapper until fully deprecated.
"""

from __future__ import annotations

import json
import math
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from pytheus.fancy_classes import Graph

# -------------------------
# Pretty maps (same as generate_data)
# -------------------------

_POSITIONS = "abcdefghijkl"
_MODES = "xyz"
position_dict: Dict[int, str] = {i: _POSITIONS[i] for i in range(12)}
mode_dict: Dict[int, str] = {i: _MODES[i] for i in range(3)}


# -------------------------
# Core helpers re-exported
# -------------------------

def build_state_string(state) -> str:
    """Compact state string (shared with data generation)."""
    s = []
    for ket in state.kets:
        weight = state[ket]
        if weight != 0:
            pretty = "".join(position_dict[i] + mode_dict[j] for (i, j) in ket)
            s.append(("+" if weight > 0 else "") + f"{weight}[{pretty}]")
    return "".join(s)


def normalize_state_segment(seg: str) -> str:
    """Canonicalize one segment (GCD reduction + lexicographic sort)."""
    if not seg:
        return seg
    try:
        raw_terms = [t for t in seg.split(']') if t]
        terms: List[Tuple[int, str]] = []
        for t in raw_terms:
            if '[' not in t:
                return seg
            weight_str, ket_inner = t.split('[', 1)
            w = int(weight_str)
            ket = '[' + ket_inner + ']'
            terms.append((w, ket))
        if not terms:
            return seg
        g = abs(terms[0][0])
        for w, _ in terms[1:]:
            g = math.gcd(g, abs(w))
        g = g or 1
        normed = [(w // g, k) for (w, k) in terms]
        normed.sort(key=lambda x: x[1])
        out = []
        for w, k in normed:
            sign = '+' if w > 0 else ''
            out.append(f"{sign}{w}{k}")
        return "".join(out)
    except Exception:
        return seg


def tokenize_string(input_str: str, token_dict: Dict[str, int]) -> np.ndarray:
    """Strict greedy prefix tokenization with <SOS>/<EOS>, int8 dtype."""
    indices: List[int] = [token_dict["<SOS>"]]
    i = 0
    while i < len(input_str):
        found = False
        for token, index in token_dict.items():
            if input_str.startswith(token, i):
                indices.append(index)
                i += len(token)
                found = True
                break
        if not found:
            print("[tokenize] Unknown token at position", i)
            print(input_str[i - 1: i + 3])
            raise Exception("unknown token found")
    indices.append(token_dict["<EOS>"])
    return np.array(indices, dtype="int8")


def detokenize_indices(indices: Iterable[int], token_dict: Dict[str, int]) -> str:
    reverse = {v: k for k, v in token_dict.items()}
    pad = token_dict.get("<PAD>")
    return "".join(reverse.get(ix, "") for ix in indices if ix != pad)


# -------------------------
# Graph utilities
# -------------------------

def _coalesce_edges_and_reject_zeros(
    edges5: Iterable[Sequence[int]],
    strict_zero: bool = True,
) -> List[Tuple[int, int, int, int, int]]:
    acc: Dict[Tuple[int, int, int, int], int] = {}
    for e in edges5:
        key = tuple(e[:4])  # type: ignore[index]
        w = int(e[4])       # type: ignore[index]
        acc[key] = acc.get(key, 0) + w
    zeros = [k for k, w in acc.items() if w == 0]
    if zeros and strict_zero:
        raise ValueError("Zero-weight edge after aggregation.")
    return [(*k, w) for k, w in acc.items() if w != 0]


def graph_from_code(code_str: str, N: int) -> Graph:
    """Reconstruct a Graph from a code string using e(...) calls.

    Supports a single 'for ii in range(N):' block; edges in the indented block
    are expanded over ii=0..N-1 by textual substitution of 'ii'. Colors/weights
    are taken verbatim from the call arguments.
    """
    lines = code_str.split('\n')
    base_edges: List[Tuple[int, int, int, int, int]] = []
    loop_edges: List[Tuple[int, int, int, int, int]] = []
    in_loop = False
    for raw in lines:
        if raw.strip() == "":
            continue
        if raw.startswith("for ii in range(N):"):
            in_loop = True
            continue
        if raw.startswith(" ") or raw.startswith("\t"):
            # loop body line
            if not in_loop:
                raise ValueError("Indented e() line outside loop")
            line = raw.strip()
            if not line.startswith("e("):
                continue
            args = line[2:-1]
            parts = args.split(',')
            u_expr, v_expr, cu_expr, cv_expr, w_expr = [p.strip() for p in parts]
            # store expressions; evaluate when expanding ii
            loop_edges.append((u_expr, v_expr, cu_expr, cv_expr, w_expr))  # type: ignore[arg-type]
        else:
            # base line
            in_loop = in_loop  # no change
            if not raw.startswith("e("):
                continue
            args = raw[2:-1]
            parts = args.split(',')
            u, v, cu, cv, w = [int(p.strip()) for p in parts]
            base_edges.append((u, v, cu, cv, w))
    # expand loop edges
    expanded_loop: List[Tuple[int, int, int, int, int]] = []
    for ii in range(N):
        for (u_expr, v_expr, cu_expr, cv_expr, w_expr) in loop_edges:
            u = eval(u_expr.replace('ii', str(ii)))  # noqa: S307 - controlled context
            v = eval(v_expr.replace('ii', str(ii)))
            cu = int(eval(cu_expr.replace('ii', str(ii))))
            cv = int(eval(cv_expr.replace('ii', str(ii))))
            w = int(eval(w_expr.replace('ii', str(ii))))
            expanded_loop.append((u, v, cu, cv, w))
    all_edges = base_edges + expanded_loop
    all_edges = _coalesce_edges_and_reject_zeros(all_edges, strict_zero=True)
    g = Graph(all_edges)
    g.complete_graph_edges = list(g.edges)
    g.getState(normalize=False)
    return g


__all__ = [
    "build_state_string",
    "normalize_state_segment",
    "tokenize_string",
    "detokenize_indices",
    "graph_from_code",
    "_coalesce_edges_and_reject_zeros",
]
