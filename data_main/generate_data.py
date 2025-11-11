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
import math

from pytheus.fancy_classes import Graph

# Support both package and script execution
try:
    from .generate_topologies import generate_graph  # type: ignore
    from .valpos_res import val_verts_0, val_verts_1  # type: ignore
    from .shared_api import (  # type: ignore
        tokenize_string,
        detokenize_indices,
        build_state_string,
        normalize_state_segment,
        _coalesce_edges_and_reject_zeros,
    )
except Exception:  # fallback when running as a script from data_main/
    from generate_topologies import generate_graph
    from valpos_res import val_verts_0, val_verts_1
    from shared_api import (
        tokenize_string,
        detokenize_indices,
        build_state_string,
        normalize_state_segment,
        _coalesce_edges_and_reject_zeros,
    )


# =========================
# Defaults for CLI
# =========================

DEFAULT_MAX_TOKS: int = 1024
DEFAULT_NUM_SAMPLES: int = 50_000
DEFAULT_BATCH_SIZE: int = 1_000


# vertex counts for N in {0, 1, 2}
VERTEX_NUMS = [4, 6, 8]


# =========================
# Helpers
# =========================


def get_worker_info(args) -> Tuple[int, int]:
    """Determine (worker_id, num_workers) from CLI and common schedulers.
    Priority:
      1) Explicit --worker-id/--num-workers
      2) SLURM array (SLURM_ARRAY_TASK_ID/SLURM_ARRAY_TASK_COUNT)
      3) SLURM MPI (SLURM_PROCID/SLURM_NTASKS)
      4) Fallback to single worker (0,1)
    """
    if args.worker_id is not None:
        return int(args.worker_id), int(args.num_workers or 1)

    env = os.environ
    # SLURM array jobs
    if "SLURM_ARRAY_TASK_ID" in env:
        wid = int(env.get("SLURM_ARRAY_TASK_ID", 0))
        n = int(env.get("SLURM_ARRAY_TASK_COUNT", 1))
        return wid, max(n, 1)
    # SLURM MPI
    if "SLURM_PROCID" in env:
        wid = int(env.get("SLURM_PROCID", 0))
        n = int(env.get("SLURM_NTASKS", 1))
        return wid, max(n, 1)
    # Local fallback
    return 0, 1


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

def generate_for_config(
    config_dict: Dict[str, str],
    token_dict: Dict[str, int],
    num_samples_target: int,
    batch_size: int,
    max_toks: int,
    verbose_invalid: bool,
    run_id: str,
    max_time: int | None = None,
    plaintext: bool = False,
) -> None:
    """
    Generate samples for a single configuration and write to a unique part file.
    """
    # IO setup for this config
    DIR_NAME = "data"
    TOP_FILENAME = f"topologies/{config_dict['CODELEN']}_{config_dict['DEGREE']}.txt"
    OUT_DIR = f"{DIR_NAME}/{'_'.join(str(v) for v in config_dict.values())}"
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR, exist_ok=True)
    FILE_NAME = f"{OUT_DIR}/{run_id}.h5"
    LOG_FILE_PATH = f"{OUT_DIR}/{run_id}.txt"

    topology_dict = {("LONG", "DEG2"): 0, ("SHORT", "DEG2"): 1, ("LONG", "DEG1"): 2, ("SHORT", "DEG1"): 3}
    topology_letter = topology_dict[(config_dict["CODELEN"], config_dict["DEGREE"])]
    MAX_KETS_tuple = tuple(int(v) for v in config_dict["MAX_KETS"].split("-"))

    print("[init] configuration:", config_dict)
    print(f"[init] topology file: {TOP_FILENAME}")
    print(f"[init] output H5:      {FILE_NAME}")
    print(f"[init] log file:       {LOG_FILE_PATH}")
    print(f"[init] samples target: {num_samples_target}, batch size: {batch_size}, max toks: {max_toks}")

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
    json_buffer: List[Dict[str, str]] = []
    jsonl_path = f"{OUT_DIR}/{run_id}.jsonl" if plaintext else None
    t0 = time.time()

    print("[run] starting main loop…")
    start_time = time.time()
    def time_exceeded() -> bool:
        return (max_time is not None) and ((time.time() - start_time) >= max_time)
    with open(TOP_FILENAME, "r") as fh:
        topo_data = fh.read()

    timed_out = False
    while valid_codes < num_samples_target:
        if time_exceeded():
            timed_out = True
            break
        for line_ind, line in enumerate(topo_data.split("\n")):
            if time_exceeded():
                timed_out = True
                break
            if valid_codes >= num_samples_target:
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
                if time_exceeded():
                    timed_out = True
                    break
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
                        if verbose_invalid:
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
                            if verbose_invalid:
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

                        # Build state string and normalize per segment (GCD reduction + sorted kets)
                        seg = build_state_string(state)
                        seg = normalize_state_segment(seg)
                        state_str_parts.append(seg)

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

                    # Optional JSON output for demonstration: one object per sample
                    if plaintext:
                        json_buffer.append({"source": state_str, "target": code_str})

                    # user-friendly progress echo
                    if valid_codes % 200 == 0:
                        elapsed = time.time() - t0
                        rate_kph = (valid_codes / (elapsed / 3600)) / 1000 if elapsed > 0 else 0.0
                        print(f"[progress] {valid_codes}/{num_samples_target} samples | "
                              f"{elapsed:.1f}s elapsed | {rate_kph:.2f}k samples/hr")

                    break  # break REPS loop after a valid sample

            # Batch flush
            if len(data_buffer) >= batch_size:
                elapsed = time.time() - t0
                unique_topologies = len(set(used_topologies))
                with open(LOG_FILE_PATH, "a") as logf:
                    logf.write(
                        f"{valid_codes} samples written in {elapsed:.1f}s, "
                        f"{(valid_codes/(elapsed/3600))/1000:.2f}k samples/hr, "
                        f"{unique_topologies} unique topologies\n"
                    )
                append_batch_to_h5(FILE_NAME, data_buffer, max_toks=max_toks)
                print(f"[h5] wrote batch of {len(data_buffer)} (total {valid_codes})")
                # Append JSONL entries for this batch if requested (one object per line in a single run file)
                if plaintext and json_buffer and jsonl_path is not None:
                    with open(jsonl_path, "a") as tf:
                        for obj in json_buffer:
                            json.dump(obj, tf, ensure_ascii=False)
                            tf.write("\n")
                    print(f"[jsonl] appended {len(json_buffer)} samples to {jsonl_path}")
                    json_buffer = []
                data_buffer = []
            if timed_out:
                break

    # Final flush if needed
    if data_buffer:
        append_batch_to_h5(FILE_NAME, data_buffer, max_toks=max_toks)
        print(f"[h5] wrote final batch of {len(data_buffer)}")
        if plaintext and json_buffer and jsonl_path is not None:
            with open(jsonl_path, "a") as tf:
                for obj in json_buffer:
                    json.dump(obj, tf, ensure_ascii=False)
                    tf.write("\n")
            print(f"[jsonl] appended {len(json_buffer)} samples to {jsonl_path}")

    if timed_out:
        elapsed_all = time.time() - start_time
        print(f"[timeout] Max time reached after {elapsed_all:.1f}s; samples generated: {valid_codes}")
    print("[done] generation complete.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, default=0, help="Deprecated; prefer --worker-id.")
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
    parser.add_argument("--worker-id", type=int, default=None, help="Worker index [0..num_workers-1].")
    parser.add_argument("--num-workers", type=int, default=None, help="Total workers participating.")
    parser.add_argument("--config-index", type=int, default=None, help="Override config combination index.")
    parser.add_argument("--cycle-all", action="store_true", help="Cycle through all combinations.")
    parser.add_argument("--cycles", type=int, default=1, help="Number of cycles when --cycle-all is set (0=infinite).")
    parser.add_argument("--max-time", type=int, default=None, help="Max seconds per config run in cycle-all mode.")
    parser.add_argument("--run-id", type=str, default=None, help="Optional unique run id for output filenames.")
    parser.add_argument("--plaintext", action="store_true", help="Also write a demo JSONL file per run (one object per line: {'source': state_str, 'target': code_str}).")
    args = parser.parse_args()

    # Back-compat: if task_id passed without worker-id, use it
    if args.worker_id is None and args.task_id is not None:
        try:
            args.worker_id = int(args.task_id)
        except Exception:
            pass

    worker_id, num_workers = get_worker_info(args)
    print(f"[init] worker: {worker_id}, total number of workers: {num_workers}")

    # Seed control
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"[init] numpy RNG seeded with {args.seed}")

    # tokens
    # token dict path robust to cwd
    tok_path = os.path.join(os.path.dirname(__file__), "tok.json")
    token_dict = json.load(open(tok_path))
    # Build all combinations once
    possible_values = {
        "CODELEN": ["SHORT", "LONG"],
        "DEGREE": ["DEG1", "DEG2"],
        "DIMENSION": ["2D", "3D"],
        "EDGEWEIGHT": ["WEIGHTED", "UNWEIGHTED"],
        "MAX_KETS": ["8-16-32", "6-6-6"],
    }
    combos = list(itertools.product(*possible_values.values()))
    num_combos = len(combos)
    print(f"[config] number of combinations: {num_combos}")

    def cfg_from_index(ind: int) -> Dict[str, str]:
        tup = combos[ind % num_combos]
        return {key: tup[i] for i, key in enumerate(possible_values.keys())}

    # derive run_id for file uniqueness
    default_run_id = args.run_id or f"{int(time.time())}_{os.getpid()}_{worker_id}"

    if args.cycle_all:
        cycles = args.cycles
        cycle_counter = 0
        while True:
            cycle_counter += 1
            print(f"[cycle] Starting cycle {cycle_counter}")
            for combo_index in range(num_combos):
                cfg = cfg_from_index(combo_index)
                run_id = f"{default_run_id}_c{cycle_counter}_i{combo_index}"
                print("-"*40)
                print(f"[cycle] combo index: {combo_index}")
                # one batch per combo per cycle
                generate_for_config(
                    cfg,
                    token_dict,
                    num_samples_target=args.batch_size,
                    batch_size=args.batch_size,
                    max_toks=args.max_toks,
                    verbose_invalid=args.verbose_invalid,
                    run_id=run_id,
                    max_time=args.max_time,
                    plaintext=args.plaintext,
                )
            if cycles > 0 and cycle_counter >= cycles:
                break
    else:
        # choose config index
        if args.config_index is not None:
            chosen_index = int(args.config_index)
        else:
            chosen_index = worker_id
        cfg = cfg_from_index(chosen_index)
        generate_for_config(
            cfg,
            token_dict,
            num_samples_target=args.num_samples,
            batch_size=args.batch_size,
            max_toks=args.max_toks,
            verbose_invalid=args.verbose_invalid,
            run_id=default_run_id,
            plaintext=args.plaintext,
        )


if __name__ == "__main__":
    main()
