import argparse
import json
import os
from typing import Dict, List, Optional, Tuple
import ast

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pytheus.graphplot import graphPlotNew
import shutil


def load_token_dict(h5_path: str, tok_path: Optional[str]) -> Dict[str, int]:
    """Load token_dict from H5 attributes if present, else from a JSON file.
    Returns a mapping token -> index.
    """
    # Try to read from H5 attributes first
    try:
        with h5py.File(h5_path, "r") as f:
            if "token_dict" in f.attrs:
                td = f.attrs["token_dict"]
                # stored as JSON string
                if isinstance(td, (bytes, bytearray)):
                    td = td.decode("utf-8")
                return json.loads(td)
    except OSError:
        pass

    # Fallback to external JSON
    if tok_path is None:
        tok_path = "tok.json"
    if not os.path.exists(tok_path):
        raise FileNotFoundError(
            f"Could not find token dict: neither in H5 '{h5_path}' attributes nor JSON file '{tok_path}'."
        )
    with open(tok_path, "r") as fh:
        return json.load(fh)


def detokenize(indices: List[int], token_dict: Dict[str, int], strip_special: bool = True) -> str:
    """Convert a sequence of token indices to string.
    - Trims at first PAD token if present (assumes right-padding with PAD)
    - By default strips <SOS> and <EOS>
    """
    # Build reverse map
    reverse = {v: k for k, v in token_dict.items()}
    pad_id = token_dict.get("<PAD>")
    sos_id = token_dict.get("<SOS>")
    eos_id = token_dict.get("<EOS>")

    out_tokens: List[str] = []
    for ix in indices:
        # stop at PAD (right-padding)
        if pad_id is not None and ix == pad_id:
            break
        if strip_special and ((sos_id is not None and ix == sos_id) or (eos_id is not None and ix == eos_id)):
            continue
        tok = reverse.get(int(ix), "")
        out_tokens.append(tok)
    return "".join(out_tokens)


def trim_at_pad(indices: np.ndarray, pad_id: Optional[int]) -> List[int]:
    if pad_id is None:
        return [int(x) for x in indices.tolist()]
    out: List[int] = []
    for x in indices.tolist():
        if int(x) == pad_id:
            break
        out.append(int(x))
    return out


# ===== Graph reconstruction from code text =====

ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult)
ALLOWED_UNARY = (ast.UAdd, ast.USub)


def _safe_eval_expr(expr: str, **env: int) -> int:
    """Safely evaluate an arithmetic expression over integers with variables N and ii.
    Allowed: +, -, *, integers, variables 'N' and 'ii'.
    """
    node = ast.parse(expr, mode="eval")

    def eval_node(n: ast.AST) -> int:
        if isinstance(n, ast.Expression):
            return eval_node(n.body)
        if isinstance(n, ast.Num):  # type: ignore[attr-defined]
            return int(n.n)  # deprecated node in py<3.8 compat
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int,)):
                return int(n.value)
            raise ValueError("Only integer constants are allowed")
        if isinstance(n, ast.Name):
            if n.id not in env:
                raise ValueError(f"Unknown variable: {n.id}")
            return int(env[n.id])
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ALLOWED_UNARY):
            val = eval_node(n.operand)
            return +val if isinstance(n.op, ast.UAdd) else -val
        if isinstance(n, ast.BinOp) and isinstance(n.op, ALLOWED_BINOPS):
            left = eval_node(n.left)
            right = eval_node(n.right)
            if isinstance(n.op, ast.Add):
                return left + right
            if isinstance(n.op, ast.Sub):
                return left - right
            if isinstance(n.op, ast.Mult):
                return left * right
        raise ValueError("Disallowed expression")

    return eval_node(node)


def _parse_e_call(line: str) -> List[str]:
    """Extract argument strings from a line like: e(arg1,arg2,arg3,arg4,arg5)"""
    s = line.strip()
    if not s.startswith("e(") or not s.endswith(")"):
        raise ValueError("Not an e(...) line")
    inner = s[2:-1]
    # split by commas (no nested commas expected)
    parts: List[str] = [p.strip() for p in inner.split(",")]
    if len(parts) != 5:
        raise ValueError("Expected 5 arguments to e(...)")
    return parts


def _coalesce_edges(edges: List[Tuple[int, int, int, int, int]]) -> List[Tuple[int, int, int, int, int]]:
    acc: Dict[Tuple[int, int, int, int], int] = {}
    for u, v, c0, c1, w in edges:
        key = (u, v, c0, c1)
        acc[key] = acc.get(key, 0) + int(w)
    return [(*k, w) for k, w in acc.items() if w != 0]


def build_graph_from_code(code_text: str, N: int) -> List[Tuple[int, int, int, int, int]]:
    """Reconstruct pytheus-style edges (u,v,cu,cv,w) from detokenized code for a given N."""
    edges: List[Tuple[int, int, int, int, int]] = []
    lines = code_text.splitlines()
    in_loop = False
    for idx, line in enumerate(lines):
        st = line.strip()
        if not st:
            continue
        if st.startswith("for ") and "for ii in range(N):" in st:
            in_loop = True
            continue
        if st.startswith("e(") and not in_loop:
            a1, a2, a3, a4, a5 = _parse_e_call(st)
            u = _safe_eval_expr(a1, N=N, ii=0)
            v = _safe_eval_expr(a2, N=N, ii=0)
            c0 = _safe_eval_expr(a3, N=N, ii=0)
            c1 = _safe_eval_expr(a4, N=N, ii=0)
            w = _safe_eval_expr(a5, N=N, ii=0)
            edges.append((u, v, c0, c1, w))
            continue
        # loop body lines are indented and start with e(
        if in_loop:
            st_noindent = st
            if st_noindent.startswith("e("):
                a1, a2, a3, a4, a5 = _parse_e_call(st_noindent)
                for ii in range(N):
                    u = _safe_eval_expr(a1, N=N, ii=ii)
                    v = _safe_eval_expr(a2, N=N, ii=ii)
                    c0 = _safe_eval_expr(a3, N=N, ii=ii)
                    c1 = _safe_eval_expr(a4, N=N, ii=ii)
                    w = _safe_eval_expr(a5, N=N, ii=ii)
                    edges.append((u, v, c0, c1, w))
            # keep collecting until file ends; the structure is simple
    return _coalesce_edges(edges)


def visualize(
    h5_path: str,
    num_samples: int,
    tok_path: Optional[str],
    start: Optional[int] = None,
    end: Optional[int] = None,
    yes: bool = False,
    range_str: Optional[str] = None,
) -> None:
    token_dict = load_token_dict(h5_path, tok_path)
    pad_id = token_dict.get("<PAD>")

    # Prepare output dir
    # Directory for saved plot images (relative to script location for stable links)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "tmp_plots")
    if os.path.exists(out_dir):
        # If it's a file/link, remove and recreate directory
        if os.path.isfile(out_dir) or os.path.islink(out_dir):
            try:
                os.remove(out_dir)
            except Exception as e:
                print(f"[warn] Could not remove existing path {out_dir}: {e}")
            os.makedirs(out_dir, exist_ok=True)
        else:
            # Clear directory contents

            for name in os.listdir(out_dir):
                path = os.path.join(out_dir, name)
                try:
                    if os.path.isdir(path) and not os.path.islink(path):
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
                except Exception as e:
                    print(f"[warn] Could not remove {path}: {e}")
    else:
        os.makedirs(out_dir, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        if "code" not in f or "state" not in f:
            raise KeyError("Input H5 must contain 'code' and 'state' datasets.")
        total = f["code"].shape[0]
        # Determine slice [s:e) with precedence: --range > --start/--end > --samples
        def _parse_range(r: str) -> Tuple[int, int]:
            if ":" not in r:
                raise ValueError("--range must be of the form start:end (end-exclusive)")
            s_part, e_part = r.split(":", 1)
            if s_part == "":
                s_val = 0
            else:
                s_val = int(s_part)
            if e_part == "":
                e_val = total
            else:
                e_val = int(e_part)
            return s_val, e_val

        if range_str:
            s_raw, e_raw = _parse_range(range_str)
            s = s_raw
            e = e_raw
        else:
            s = int(start) if start is not None else 0
            if end is not None:
                e = int(end)
            else:
                e = s + int(num_samples)
        # Clamp to available range
        s = max(0, min(s, total))
        e = max(0, min(e, total))
        if e <= s:
            print(f"[warn] Empty slice [{s}:{e}) for dataset of length {total}; nothing to do.")
            # Write a minimal header anyway
            script_dir = os.path.dirname(os.path.abspath(__file__))
            combined_md_path = os.path.join(script_dir, "sample_plots.md")
            with open(combined_md_path, "w") as mdfile:
                mdfile.write(f"# Sample plots (0 samples)\n\n")
            print(f"[write] {combined_md_path}")
            return

        take = e - s
        # Confirmation prompt if generating many plots
        if take > 30 and not yes:
            resp = input(
                f"Warning: This will create {3*take} individual PNG files under 'tmp_plots/'. Proceed? [y/N]: "
            ).strip().lower()
            if resp not in ("y", "yes"):
                print("[abort] User declined to generate plots.")
                return

        code = f["code"][s:e]
        state = f["state"][s:e]

    # We'll aggregate a single Markdown report in the script directory
    md_lines: List[str] = []
    md_lines.append(f"# Sample plots ({take} samples)\n")
    md_lines.append("")

    for i in range(take):
        global_idx = s + i
        code_row = code[i]
        state_row = state[i]
        code_ix = trim_at_pad(code_row, pad_id)
        state_ix = trim_at_pad(state_row, pad_id)
        code_text = detokenize(code_ix, token_dict, strip_special=True)
        state_text = detokenize(state_ix, token_dict, strip_special=True)

        # Build graphs for N=0,1,2 from the detokenized code
        graphs_by_N: Dict[int, List[Tuple[int, int, int, int, int]]] = {}
        for N in (0, 1, 2):
            graphs_by_N[N] = build_graph_from_code(code_text, N)

        # Save plots for each N and collect markdown embeds
        image_links: List[str] = []
        for N in (0, 1, 2):
            edges = graphs_by_N[N]
            try:
                fig = graphPlotNew(edges, show=False)
                img_filename = f"{global_idx}_{N}.png"
                img_path = os.path.join(out_dir, img_filename)
                fig.savefig(img_path, dpi=200, bbox_inches="tight")
                plt.close(fig)
                # Link relative to the markdown file location (script_dir)
                image_links.append(f"./tmp_plots/{img_filename}")
            except Exception as e:
                image_links.append(f"(plot failed: {e})")

        # Per-sample section in combined markdown
        md_lines.append(f"\n---\n\n## Sample {global_idx}\n")

        # 1) Code (generated)
        md_lines.append("\n### Code (generated)\n")
        md_lines.append("```\n" + code_text + "\n```\n")

        # 2) Generated graphs (table)
        md_lines.append("\n### Generated graphs\n")
        md_lines.append("| N=0 | N=1 | N=2 |")
        md_lines.append("|:---:|:---:|:---:|")

        # Images row
        img_cells = []
        for N in (0, 1, 2):
            if len(image_links) > N and not image_links[N].startswith("(plot failed"):
                img_cells.append(f"![]({image_links[N]})")
            else:
                img_cells.append(image_links[N] if len(image_links) > N else "(no image)")
        md_lines.append("| " + " | ".join(img_cells) + " |")

        # Edges row (single-line, no code formatting)
        edge_cells = []
        for N in (0, 1, 2):
            edges = graphs_by_N[N]
            if edges:
                # compact single-line listing
                edge_cells.append("; ".join(str(e) for e in edges))
            else:
                edge_cells.append("(no edges)")
        md_lines.append("| " + " | ".join(edge_cells) + " |\n")

        # 3) State String
        md_lines.append("\n### State String\n")
        md_lines.append("```\n" + state_text + "\n```\n")

        # 4) Tokenized Code (indices)
        md_lines.append("\n### Tokenized Code (indices)\n")
        md_lines.append("```\n[" + ", ".join(str(x) for x in code_ix) + "]\n```\n")

        # 5) Tokenized State String (indices)
        md_lines.append("\n### Tokenized State String (indices)\n")
        md_lines.append("```\n[" + ", ".join(str(x) for x in state_ix) + "]\n```\n")

    # Write combined markdown beside the script
    combined_md_path = os.path.join(script_dir, "sample_plots.md")
    with open(combined_md_path, "w") as mdfile:
        mdfile.write("\n".join(md_lines) + "\n")
    print(f"[write] {combined_md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to HDF5 dataset (e.g., data/shuffled_data_0.h5)")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to visualize (default: 10)")
    parser.add_argument("--tok", type=str, default=None, help="Optional path to tok.json if not stored in H5 attrs")
    parser.add_argument("--start", type=int, default=None, help="Start index (inclusive) for slicing the dataset")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive) for slicing the dataset")
    parser.add_argument("--yes", action="store_true", help="Proceed without prompt when creating many plots")
    parser.add_argument("--range", dest="range_str", type=str, default=None, help="Slice as 'start:end' (end-exclusive); overrides --start/--end/--samples")
    args = parser.parse_args()

    visualize(
        args.file,
        args.samples,
        args.tok,
        start=args.start,
        end=args.end,
        yes=args.yes,
        range_str=args.range_str,
    )

