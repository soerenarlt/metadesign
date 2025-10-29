"""
Topology generator for pytheus-based experiments.

What it does
------------
1) Randomly samples candidate edge sets for two layers.
2) For each N in {0, 1, 2}, builds the corresponding graph instance (positions expanded)
   and validates it via:
   - no self-loops
   - vertex count and minimum degree
   - every edge belongs to at least one perfect matching
3) On success, appends the (layer_0 | layer_1) description to a file in `topologies/`.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import List, Sequence

import numpy as np
import pytheus.theseus as th

from valpos_res import val_verts_0, val_verts_1


# =========================
# Constants
# =========================

# Vertex counts for N in {0, 1, 2}
VERTEX_NUMS = [4, 6, 8]


# =========================
# Precomputed position tables
# =========================

# Expand val_verts_* string templates into concrete integer positions for each N.
pos_0_temp_list: List[List[int]] = []
for N in range(3):
    pos_0_temp_list.append([eval(pos.replace("N", str(N))) for pos in val_verts_0])

pos_1_temp_list: List[List[List[int]]] = []
for N in range(3):
    entry_base = [pos.replace("N", str(N)) for pos in val_verts_1]
    pos1_temp_list_2: List[List[int]] = []
    for ii in range(N):
        entry = [eval(pos.replace("ii", str(ii))) for pos in entry_base]
        pos1_temp_list_2.append(entry)
    pos_1_temp_list.append(pos1_temp_list_2)


# =========================
# Helpers
# =========================

def generate_graph(
    layer_0: Sequence[Sequence[int]],
    layer_1: Sequence[Sequence[int]],
    N: int,
    layer_0_extra: Sequence[Sequence[int]] | None = None,
    layer_1_extra: Sequence[Sequence[int]] | None = None,
) -> List[List[int]]:
    """
    Expand abstract layer indices into concrete edges for a given N.

    Each edge is [u, v] here (no colors/weights in this generator).
    """
    pos_0_temp = pos_0_temp_list[N]

    graph: List[List[int]] = []
    if layer_0_extra is None:
        layer_0_extra = [[]] * len(layer_0)
    if layer_1_extra is None:
        layer_1_extra = [[]] * len(layer_1)

    for line, extra in zip(layer_0, layer_0_extra):
        edge = [pos_0_temp[line[0]], pos_0_temp[line[1]]]
        edge = edge + extra
        graph.append(edge)

    for line, extra in zip(layer_1, layer_1_extra):
        for ii in range(N):  # this assumes that for-loops in codes are always over range(N). For more generality, we would need to pass in the loop ranges.
            pos1_temp2 = pos_1_temp_list[N][ii]
            edge = [pos1_temp2[line[0]], pos1_temp2[line[1]]]
            edge = edge + extra
            graph.append(edge)
    return graph


def check_self_loops(graph: Sequence[Sequence[int]]) -> bool:
    """Return True if no edge is a self-loop."""
    for edge in graph:
        if edge[0] == edge[1]:
            return False
    return True


def check_degree_and_vertcount(
    graph: Sequence[Sequence[int]],
    vertex_count: int,
    min_degree: int = 2,
) -> bool:
    """Check vertex count and that all vertices have degree >= min_degree."""
    verts = sum(graph, [])
    verts, degrees = np.unique(verts, return_counts=True)
    if len(verts) != vertex_count:
        return False
    if not all(degrees >= min_degree):
        return False
    return True


def check_pms(graph: Sequence[Sequence[int]]) -> bool:
    """
    Check that every edge participates in at least one perfect matching.

    Converts each edge to a 4-tuple (u, v, cu, cv) with zero colors.
    """
    edges4 = [tuple(edge + [0, 0]) for edge in graph]
    pms = th.findPerfectMatchings(edges4)
    pm_edges = {e for pm in pms for e in pm}
    return all(e in pm_edges for e in edges4)


# =========================
# Main
# =========================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="A", choices=list("ABCD"), help="Which config to use.")
    parser.add_argument(
        "--num_tops",
        type=int,
        default=1,
        help="Number of topologies to generate in this run (default: 1).",
    )
    parser.add_argument(
        "--verbose-invalid",
        action="store_true",
        help="If set, print reasons when a candidate graph is rejected.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100_000,
        help="Print a progress line every N attempts (default: 100000).",
    )
    args = parser.parse_args()

    print(f"[init] Computing config {args.config}")
    print(f"[hint] To change config, use --config with A, B, C, or D")
    print(f"[init] Generating {args.num_tops} topologies")

    save_dir = "topologies"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    configA = {
        "filename": "LONG_DEG2",
        "line_num_bounds_0": [4, 12],
        "line_num_bounds_1": [2, 12],
        "MIN_DEGREE": 2,
    }
    configB = {
        "filename": "SHORT_DEG2",
        "line_num_bounds_0": [4, 8],
        "line_num_bounds_1": [2, 6],
        "MIN_DEGREE": 2,
    }
    configC = {
        "filename": "LONG_DEG1",
        "line_num_bounds_0": [4, 12],
        "line_num_bounds_1": [2, 12],
        "MIN_DEGREE": 1,
    }
    configD = {
        "filename": "SHORT_DEG1",
        "line_num_bounds_0": [4, 8],
        "line_num_bounds_1": [2, 6],
        "MIN_DEGREE": 1,
    }

    configs = {"A": configA, "B": configB, "C": configC, "D": configD}
    config = configs[args.config]

    failed_at_loop = 0
    failed_at_degree = 0
    failed_at_pms = 0

    tt = time.time()

    NUM_TOPS = args.num_tops
    for _ in range(NUM_TOPS):
        valid = False
        num_tries = 0

        while not valid:
            num_tries += 1
            if args.progress_every > 0 and num_tries % args.progress_every == 0:
                total = max(1, num_tries)
                loop_p = failed_at_loop / total * 100
                deg_p = failed_at_degree / total * 100
                pm_p = failed_at_pms / total * 100
                print(
                    f"[progress] attempts: {num_tries} | "
                    f"failed(loop/deg/pm): {failed_at_loop}/{failed_at_degree}/{failed_at_pms} "
                    f"({loop_p:.1f}%/{deg_p:.1f}%/{pm_p:.1f}%)"
                )

            num_lines_0 = np.random.randint(*config["line_num_bounds_0"])
            num_lines_1 = np.random.randint(*config["line_num_bounds_1"])

            # shape (num_lines_0, 2) and (num_lines_1, 2)
            layer_0 = np.random.randint(len(val_verts_0), size=(num_lines_0, 2))
            layer_1 = np.random.randint(len(val_verts_1), size=(num_lines_1, 2))

            # Validate for N in {0, 1, 2}
            for N, vertex_count in enumerate(VERTEX_NUMS):
                gg = generate_graph(layer_0, layer_1, N)
                gg = [sorted(edge) for edge in gg]

                # 1) Self-loops
                if not check_self_loops(gg):
                    failed_at_loop += 1
                    if args.verbose_invalid:
                        print(f"[reject] N={N}: self-loop encountered")
                    valid = False
                    break

                # 2) Degree & vertex count
                if not check_degree_and_vertcount(gg, vertex_count, min_degree=config["MIN_DEGREE"]):
                    failed_at_degree += 1
                    if args.verbose_invalid:
                        print(f"[reject] N={N}: degree/vertex-count check failed")
                    valid = False
                    break

                # 3) Every edge in some perfect matching
                if not check_pms(gg):
                    failed_at_pms += 1
                    if args.verbose_invalid:
                        print(f"[reject] N={N}: edge not in any perfect matching")
                    valid = False
                    break

                valid = True  # so far good for this N; overall valid if all N pass

        print("[result] CODE FOUND")

        # Save layer_0, layer_1 to file (append)
        out_path = os.path.join(save_dir, f"{config['filename']}.txt")
        with open(out_path, "a") as f:
            layer_0_list = [list(edge) for edge in layer_0]
            layer_1_list = [list(edge) for edge in layer_1]
            f.write(f"{layer_0_list}|{layer_1_list}\n")

    print(f"[done] time taken: {time.time() - tt:.3f} seconds")
    print(f"[stats] attempts: {num_tries} | "
          f"failed(loop/deg/pm): {failed_at_loop}/{failed_at_degree}/{failed_at_pms}")
    print(f"[output] appended to: {out_path}")


if __name__ == "__main__":
    main()
