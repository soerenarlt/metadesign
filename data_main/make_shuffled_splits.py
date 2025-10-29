import argparse
import os
import h5py
import numpy as np
from typing import Dict, List, Tuple

DATA_DIR = "data"


def list_outdirs(base_dir: str) -> List[str]:
    """Return immediate subdirectories of base_dir that contain .h5 part files."""
    outdirs = []
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if not os.path.isdir(path):
            continue
        # consider it an OUT_DIR if it has any .h5 that aren't final/intermediate outputs
        part_files = [
            f for f in os.listdir(path)
            if f.endswith(".h5")
            and not f.startswith("split_data_")
            and not f.startswith("shuffled_data_")
            and f != "combined.h5"
        ]
        if part_files:
            outdirs.append(path)
    return outdirs


def list_part_files(outdir: str) -> List[str]:
    files = [
        f for f in os.listdir(outdir)
        if f.endswith(".h5") and f != "combined.h5"
        and not f.startswith("split_data_") and not f.startswith("shuffled_data_")
    ]
    # Sort for deterministic ordering
    files.sort()
    return [os.path.join(outdir, f) for f in files]


def dataset_names_and_shapes(sample_file: str) -> Tuple[List[str], Dict[str, Tuple[int, ...]]]:
    with h5py.File(sample_file, "r") as f:
        names = list(f.keys())
        shapes = {name: f[name].shape for name in names}
    return names, shapes


def total_len_for_dir(outdir: str, key: str = "code") -> int:
    total = 0
    for fp in list_part_files(outdir):
        with h5py.File(fp, "r") as f:
            total += f[key].shape[0]
    return total


def read_slice_from_dir(outdir: str, start: int, end: int) -> Dict[str, np.ndarray]:
    """Read a slice [start:end] across the concatenation of all part files in a directory."""
    assert 0 <= start <= end
    remaining = end - start
    if remaining == 0:
        return {}

    parts = list_part_files(outdir)
    if not parts:
        return {}

    # identify dataset names and shapes from the first file
    ds_names, _ = dataset_names_and_shapes(parts[0])

    # We'll collect in lists then concat
    buckets: Dict[str, List[np.ndarray]] = {name: [] for name in ds_names}

    # Traverse files and slice
    offset = 0
    for fp in parts:
        with h5py.File(fp, "r") as f:
            n = f[ds_names[0]].shape[0]
            file_start = max(0, start - offset)
            file_end = min(n, start - offset + remaining)
            if file_start < file_end:
                sl = slice(file_start, file_end)
                for name in ds_names:
                    buckets[name].append(np.array(f[name][sl]))
                remaining -= (file_end - file_start)
                start += (file_end - file_start)
            offset += n
            if remaining == 0:
                break

    # Concat per dataset
    out: Dict[str, np.ndarray] = {}
    for name, arrs in buckets.items():
        if arrs:
            out[name] = np.concatenate(arrs, axis=0)
    return out


def create_or_resize_datasets(h5: h5py.File, sample: Dict[str, np.ndarray]) -> None:
    for name, arr in sample.items():
        if name in h5:
            # resize along axis 0
            old = h5[name].shape[0]
            h5[name].resize(old + arr.shape[0], axis=0)
            h5[name][old: old + arr.shape[0]] = arr
        else:
            maxshape = (None, *arr.shape[1:])
            h5.create_dataset(name, data=arr, chunks=True, maxshape=maxshape)


def build_split(split_index: int, data_split: int, outdirs: List[str], tmp_split_path: str) -> int:
    """Build split_data_{i}.h5 by taking even slices from each OUT_DIR and concatenating.
    Returns total rows written for the split.
    """
    total_written = 0
    with h5py.File(tmp_split_path, "w") as out_h5:
        for outdir in outdirs:
            total_len = total_len_for_dir(outdir)
            start = int(total_len * split_index / data_split)
            end = int(total_len * (split_index + 1) / data_split)
            if end <= start:
                continue
            data = read_slice_from_dir(outdir, start, end)
            if not data:
                continue
            create_or_resize_datasets(out_h5, data)
            # update total
            any_key = next(iter(data.keys()))
            total_written += data[any_key].shape[0]
    return total_written


def shuffle_split(tmp_split_path: str, shuffled_path: str, seed: int = None) -> None:
    if seed is not None:
        np.random.seed(seed)
    with h5py.File(tmp_split_path, "r") as in_h5:
        # assume 'code' exists and use its length
        n = in_h5["code"].shape[0]
        perm = np.random.permutation(n)
        with h5py.File(shuffled_path, "w") as out_h5:
            for name in in_h5.keys():
                arr = np.array(in_h5[name])
                shuffled = arr[perm]
                out_h5.create_dataset(
                    name,
                    data=shuffled,
                    chunks=True,
                    maxshape=(None, *arr.shape[1:]),
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--data-split", type=int, required=True, help="Number of output shards")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for shuffling")
    parser.add_argument(
        "--keep-intermediate", action="store_true", help="Keep split_data_*.h5 files"
    )
    args = parser.parse_args()

    outdirs = list_outdirs(args.data_dir)
    if not outdirs:
        print("[split] No OUT_DIRs found under", args.data_dir)
        return
    print(f"[split] Found {len(outdirs)} OUT_DIRs:")
    for d in outdirs:
        print("  -", d)

    for i in range(args.data_split):
        tmp_split = os.path.join(args.data_dir, f"split_data_{i}.h5")
        shuffled = os.path.join(args.data_dir, f"shuffled_data_{i}.h5")
        print(f"[split] Building split {i+1}/{args.data_split} -> {tmp_split}")
        total = build_split(i, args.data_split, outdirs, tmp_split)
        print(f"[split]   rows written: {total}")
        print(f"[shuffle] Shuffling -> {shuffled}")
        shuffle_split(tmp_split, shuffled, seed=args.seed)
        if not args.keep_intermediate:
            try:
                os.remove(tmp_split)
            except OSError:
                pass
        print(f"[done] Wrote {shuffled}")


if __name__ == "__main__":
    main()
