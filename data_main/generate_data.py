"""
Dataset generator for graph-based quantum states.

What it does
------------
1) Reads pre-generated topologies from `topologies/<CODELEN>_<DEGREE>.txt`
2) Samples random colors/weights and constructs graphs for N in {0, 1, 2}
3) Validates graphs, builds states (unnormalized), and tokenizes code/state strings
4) Appends samples into an HDF5 file with fixed shapes
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import numpy as np

from pytheus.fancy_classes import Graph
from generate_topologies import generate_graph
from valpos_res import val_verts_0, val_verts_1


# =========================
# Defaults for CLI
# =========================

DEFAULT_MAX_TOKS: int = 1024
DEFAULT_NUM_SAMPLES: int = 50_000
DEFAULT_BATCH_SIZE: int = 1_000


# =========================
# Pretty maps for state strings
# =========================

_POSITIONS = "abcdefgh"
_MODES = "xyz"
position_dict: Dict[int, str] = {i: _POSITIONS[i] for i in range(8)}
mode_dict: Dict[int, str] = {i: _MODES[i] for i in range(3)}

# vertex counts for N in {0, 1, 2}
VERTEX_NUMS = [4, 6, 8]


# =========================
# Helpers
# =========================

def _coalesce_edges_and_reject_zeros(
    edges5: Iterable[Sequence[int]],
    strict_zero: bool = True,
) -> List[Tuple[int, int, int, int, int]]:
    """
    Deterministic semantics for duplicate edges:
      - Sum weights for duplicate (u, v, cu, cv).
      - If strict_zero is True, raise if any aggregated weight becomes 0.
      - Otherwise, drop zero-weight edges.

    Input:
        edges5: iterable of 5-tuples (u, v, cu, cv, w)

    Returns:
        List of 5-tuples (u, v, cu, cv, w) with aggregated, non-zero weights.
    """
    acc: Dict[Tuple[int, int, int, int], int] = {}
    for e in edges5:
        key = tuple(e[:4])  # type: ignore[index]
        w = int(e[4])       # type: ignore[index]
        acc[key] = acc.get(key, 0) + w

    zeros = [k for k, w in acc.items() if w == 0]
    if zeros and strict_zero:
        # Caller can catch and mark the candidate invalid.
        raise ValueError("Zero-weight edge after aggregation.")

    return [(*k, w) for k, w in acc.items() if w != 0]


def build_state_string(state) -> str:
    """
    Build a compact, human-readable state string like: '+2[axbycxdy]+1[aybxcxdy]'
    """
    s = []
    for ket in state.kets:
        weight = state[ket]
        if weight != 0:
            pretty = "".join(position_dict[i] + mode_dict[j] for (i, j) in ket)
            s.append(("+" if weight > 0 else "") + f"{weight}[{pretty}]")
    return "".join(s)


def tokenize_string(input_str: str, token_dict: Dict[str, int]) -> np.ndarray:
    """
    Greedy tokenization by prefix matching. Adds <SOS> at start and <EOS> at end.
    """
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
            print(input_str[i - 1 : i + 3])
            raise Exception("unknown token found")
    indices.append(token_dict["<EOS>"])
    return np.array(indices, dtype="int8")


def detokenize_indices(indices: Iterable[int], token_dict: Dict[str, int]) -> str:
    """
    Reverse of tokenize_string(), ignoring <PAD>.
    """
    reverse = {v: k for k, v in token_dict.items()}
    return "".join(reverse.get(ix, "") for ix in indices if ix != token_dict["<PAD>"])


def build_graph_and_state(
    layer_0: Sequence[Sequence[int]],
    layer_1: Sequence[Sequence[int]],
    N: int,
    layer_0_extra: Sequence[Sequence[int]],
    layer_1_extra: Sequence[Sequence[int]],
):
    """
    Construct graph for a given N, coalesce duplicate edges, and return (Graph, state)
    with unnormalized amplitudes. Any failure is raised to the caller.
    """
    graph = generate_graph(layer_0, layer_1, N, layer_0_extra, layer_1_extra)
    graph = [tuple(int(el) for el in edge) for edge in graph]  # (u, v, cu, cv, w)
    graph = _coalesce_edges_and_reject_zeros(graph, strict_zero=True)

    g = Graph(graph)
    # Some pytheus internals (e.g., state catalog tensorization) may rely on complete_graph_edges; set it explicitly.
    g.complete_graph_edges = list(g.edges)
    # Leave amplitudes unnormalized; the dataset/pipeline expects integer amplitudes. In newer pytheus versions, getState() normalizes by default, causing float amplitudes and issues with tokenization.
    g.getState(normalize=False)
    return g, g.state


def append_batch_to_h5(
    filename: str,
    data_buffer: List[Dict[str, np.ndarray]],
    max_toks: int,
) -> None:
    """
    Append the current buffer to an HDF5 file, creating datasets on first write.
    Asserts that sequences do not exceed max_toks.
    """
    maxshape_dict = {
        "code": (None, max_toks),
        "state": (None, max_toks),
        "topology_ind": (None, 2),
        "line_count": (None, 2),
        "num_kets": (None, 3),
        "num_zero_kets": (None, 3),
        "num_pms": (None, 3),
        "degrees_list": (None, 3, 8),
    }

    with h5py.File(filename, "a") as f:
        # Lazily create datasets
        for key in data_buffer[0]:
            if key not in f:
                init_shape = (0, *maxshape_dict[key][1:])
                print(f"[h5] creating dataset '{key}' with shape {init_shape}")
                f.create_dataset(key, init_shape, maxshape=maxshape_dict[key], dtype="int8")

        # Resize and write
        n_new = len(data_buffer)
        for key in data_buffer[0]:
            f[key].resize((f[key].shape[0] + n_new), axis=0)
            if key in ["code", "state"]:
                padded = np.zeros((n_new, max_toks), dtype="int8")
                for i, sample in enumerate(data_buffer):
                    seq = sample[key]
                    assert len(seq) <= max_toks, (
                        f"Sample sequence for '{key}' longer than max_toks: "
                        f"{len(seq)} > {max_toks}"
                    )
                    padded[i, : len(seq)] = seq
                f[key][-n_new:] = padded
            else:
                arr = np.array([sample[key] for sample in data_buffer], dtype="int8")
                f[key][-n_new:] = arr


# =========================
# Main
# =========================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, default=0, help="Task index for distributed runs.")
    parser.add_argument(
        "--verbose-invalid",
        action="store_true",
        help="If set, print reasons when a candidate graph/state is rejected.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=f"Total samples to produce (default: {DEFAULT_NUM_SAMPLES}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Samples per HDF5 write (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--max-toks",
        type=int,
        default=DEFAULT_MAX_TOKS,
        help=f"Maximum token sequence length (default: {DEFAULT_MAX_TOKS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None).",
    )
    args = parser.parse_args()

    node_id = args.task_id
    local_id = os.environ.get("SLURM_LOCALID", 0)
    print(f"[init] Local Task ID: {local_id}")
    SLURMID = str(32 * int(node_id) + int(local_id))

    # Seed control
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"[init] numpy RNG seeded with {args.seed}")

    # tokens
    token_dict = json.load(open("tok.json"))

    def build_config_dict(ind: int):
        possible_values = {
            "CODELEN": ["SHORT", "LONG"],
            "DEGREE": ["DEG1", "DEG2"],
            "DIMENSION": ["2D", "3D"],
            "EDGEWEIGHT": ["WEIGHTED", "UNWEIGHTED"],
            "MAX_KETS": ["8-16-32", "6-6-6"],
        }
        all_combinations = list(itertools.product(*possible_values.values()))
        print(f"[config] number of combinations: {len(all_combinations)}")
        print(f"[config] chosen index: {ind} of {len(all_combinations)} (applied modulo if needed)")
        combination = all_combinations[ind % len(all_combinations)]
        cfg = {key: combination[i] for i, key in enumerate(possible_values.keys())}
        return cfg, len(all_combinations)

    config_dict, num_combinations = build_config_dict(int(SLURMID))
    FILE_IND = int(SLURMID) // num_combinations
    globals().update(config_dict)

    topology_dict = {("LONG", "DEG2"): 0, ("SHORT", "DEG2"): 1, ("LONG", "DEG1"): 2, ("SHORT", "DEG1"): 3}
    topology_letter = topology_dict[(CODELEN, DEGREE)]
    MAX_KETS_tuple = tuple(int(v) for v in config_dict["MAX_KETS"].split("-"))

    # IO setup
    DIR_NAME = "data"
    TOP_FILENAME = f"topologies/{CODELEN}_{DEGREE}.txt"
    OUT_DIR = f"{DIR_NAME}/{'_'.join(str(v) for v in config_dict.values())}"
    if FILE_IND == 0 and not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    FILE_NAME = f"{OUT_DIR}/{FILE_IND}.h5"
    LOG_FILE_PATH = f"{OUT_DIR}/{FILE_IND}.txt"

    print("[init] configuration:", config_dict)
    print(f"[init] topology file: {TOP_FILENAME}")
    print(f"[init] output H5:      {FILE_NAME}")
    print(f"[init] log file:       {LOG_FILE_PATH}")
    print(f"[init] samples target: {args.num_samples}, batch size: {args.batch_size}, max toks: {args.max_toks}")
    print(f"[hint] config is chosen by SLURMID mod {num_combinations} combinations.")

    # Store run metadata in the H5 file attributes
    with h5py.File(FILE_NAME, "a") as f:
        f.attrs["token_dict"] = json.dumps(token_dict)
        f.attrs["config_dict"] = json.dumps(config_dict)
        f.attrs["val_verts_0"] = json.dumps(val_verts_0)
        f.attrs["val_verts_1"] = json.dumps(val_verts_1)

    # sampling constants
    WEIGHTS = [-1, 1] if config_dict["EDGEWEIGHT"] == "WEIGHTED" else [1]
    REPS = 100

    used_topologies: List[int] = []
    valid_codes = 0
    data_buffer: List[Dict[str, np.ndarray]] = []
    t0 = time.time()

    print("[run] starting main loop…")
    while valid_codes < args.num_samples:
        with open(TOP_FILENAME, "r") as fh:
            data = fh.read()

        for line_ind, line in enumerate(data.split("\n")):
            if valid_codes >= args.num_samples:
                break
            if line == "":
                continue

            # Choose color set per line
            if config_dict["DIMENSION"] == "3D":
                COLORS = [0, 1, 2]
            else:  # "2D"
                COLORS = list(np.random.choice([0, 1, 2], 2, replace=False))

            valid = False
            for _ in range(REPS):
                # Parse and shuffle layers
                parts = line.split("|")
                layer_0 = eval(parts[0])
                layer_1 = eval(parts[1])

                np.random.shuffle(layer_0)
                np.random.shuffle(layer_1)

                for layer in (layer_0, layer_1):
                    for posvals in layer:
                        np.random.shuffle(posvals)

                # random colors/weights per edge
                layer_0_extra = [[np.random.choice(COLORS), np.random.choice(COLORS), np.random.choice(WEIGHTS)]
                                 for _ in range(len(layer_0))]
                layer_1_extra = [[np.random.choice(COLORS), np.random.choice(COLORS), np.random.choice(WEIGHTS)]
                                 for _ in range(len(layer_1))]

                valid = True
                # quick validation pass for N = 0, 1, 2
                for N in range(3):
                    try:
                        g, state = build_graph_and_state(
                            layer_0, layer_1, N, layer_0_extra, layer_1_extra
                        )
                    except Exception as e:
                        if args.verbose_invalid:
                            print(f"[validate] N={N} invalid graph/state (reason: {e})")
                        valid = False
                        break

                    non_zero = [amp for amp in state.amplitudes if amp != 0]
                    if len(non_zero) > MAX_KETS_tuple[N] or len(non_zero) == 0:
                        valid = False
                        break

                if valid:
                    valid_codes += 1
                    used_topologies.append(line_ind)

                    # Build per-N stats and state strings
                    state_str_parts: List[str] = []
                    topology_ind = np.array([topology_letter, line_ind], dtype="int8")
                    line_count = np.array([len(layer_0), len(layer_1)], dtype="int8")

                    num_kets = np.zeros(3, dtype="int8")
                    num_zero_kets = np.zeros(3, dtype="int8")
                    num_pms = np.zeros(3, dtype="int8")
                    degrees_list = np.zeros((3, 8), dtype="int8")

                    for N in range(3):
                        try:
                            g, state = build_graph_and_state(
                                layer_0, layer_1, N, layer_0_extra, layer_1_extra
                            )
                        except Exception as e:
                            if args.verbose_invalid:
                                print(f"[build] N={N} failed (reason: {e})")
                            valid = False
                            break

                        num_kets[N] = len([amp for amp in state.amplitudes if amp != 0])
                        num_zero_kets[N] = len([amp for amp in state.amplitudes if amp == 0])
                        num_pms[N] = len(g.perfect_matchings)

                        verts = sum([list(edge[:2]) for edge in g.edges], [])
                        _, degrees = np.unique(verts, return_counts=True)
                        degrees = np.sort(degrees)
                        degrees_list[N, : (4 + 2 * N)] = degrees

                        state_str_parts.append(build_state_string(state))

                    if not valid:
                        continue

                    state_str = "|".join(state_str_parts)

                    # produce a code string describing the sampled program
                    code_lines = []
                    for pos, extra in zip(layer_0, layer_0_extra):
                        code_lines.append(
                            f"e({val_verts_0[pos[0]]},{val_verts_0[pos[1]]},{extra[0]},{extra[1]},{extra[2]})"
                        )
                    code_lines.append("for ii in range(N):")
                    for pos, extra in zip(layer_1, layer_1_extra):
                        code_lines.append(
                            f"    e({val_verts_1[pos[0]]},{val_verts_1[pos[1]]},{extra[0]},{extra[1]},{extra[2]})"
                        )
                    code_str = "\n".join(code_lines)

                    # tokenize
                    code_tok = tokenize_string(code_str, token_dict)
                    state_tok = tokenize_string(state_str, token_dict)

                    sample = {
                        "code": code_tok,
                        "state": state_tok,
                        "topology_ind": topology_ind,
                        "line_count": line_count,
                        "num_kets": num_kets,
                        "num_zero_kets": num_zero_kets,
                        "num_pms": num_pms,
                        "degrees_list": degrees_list,
                    }
                    data_buffer.append(sample)

                    # user-friendly progress echo
                    if valid_codes % 200 == 0:
                        elapsed = time.time() - t0
                        rate_kph = (valid_codes / (elapsed / 3600)) / 1000 if elapsed > 0 else 0.0
                        print(f"[progress] {valid_codes}/{args.num_samples} samples | "
                              f"{elapsed:.1f}s elapsed | {rate_kph:.2f}k samples/hr")

                    break  # break REPS loop after a valid sample

            # Batch flush
            if len(data_buffer) >= args.batch_size:
                elapsed = time.time() - t0
                unique_topologies = len(set(used_topologies))
                with open(LOG_FILE_PATH, "a") as logf:
                    logf.write(
                        f"{valid_codes} samples written in {elapsed:.1f}s, "
                        f"{(valid_codes/(elapsed/3600))/1000:.2f}k samples/hr, "
                        f"{unique_topologies} unique topologies\n"
                    )
                append_batch_to_h5(FILE_NAME, data_buffer, max_toks=args.max_toks)
                print(f"[h5] wrote batch of {len(data_buffer)} (total {valid_codes})")
                data_buffer = []

    # Final flush if needed
    if data_buffer:
        append_batch_to_h5(FILE_NAME, data_buffer, max_toks=args.max_toks)
        print(f"[h5] wrote final batch of {len(data_buffer)}")

    print("[done] generation complete.")


if __name__ == "__main__":
    main()
