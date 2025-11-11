# Data pipeline: topologies → samples → shuffled shards


## Script structure

This README documents the following 3-step pipeline:

1) `generate_topologies.py` → append candidate topologies to `topologies/`
2) `generate_data.py` → generate tokenized samples into `data/<config>/*.h5`
3) `make_shuffled_splits.py` → build even splits across configs and shuffle into final shards

All scripts are idempotent and append-only (no overwrites), and the final outputs match the old
combined/shuffled files without creating large intermediate "combined" files.

---


## Quickstart (single CPU)

**Quickest possible start** (run only this command):

```
bash quickstart_single_cpu.sh
```

### Individual steps

Run these from the `data_main/` folder. This is a minimal demo to prove the pipeline end‑to‑end; real runs typically use multiple CPU nodes and produce many more topologies and samples.

1) **Create one topology for each topology config**

  Usually you generate thousands (it’s slow). Here we create just one for the four possible topology config settings:

  ```bash
  python generate_topologies.py --config A --num_tops 1
  python generate_topologies.py --config B --num_tops 1
  python generate_topologies.py --config C --num_tops 1
  python generate_topologies.py --config D --num_tops 1
  ```

  You'll find the topologies stored in the folder `data_main/topologies/`.

2) **Try to create 10 samples for each of the 32 data‑gen configs, with a 30s time limit per config**

  This uses *cycle‑all* mode to iterate all 32 configs once and writes a small part file per config. The 30s limit skips  is just for demonstration and is not typically used in production.

  ```bash
  # add --plaintext to also write a single demo JSONL file per run
  # (one object per line: {"source": <state_str>, "target": <code_str>}; handy for inspection, optional)
  python generate_data.py --cycle-all --cycles 1 --batch-size 10 --max-time 30 --plaintext
  ```

  You'll find the h5 files in subdirectories of `data/`. The folder names correspond to the ID of the 32 possible data-gen configs.

3) **Shuffle and produce a single output file**

  You can produce multiple shards by increasing `--data-split`.

  ```bash
  python make_shuffled_splits.py --data-split 1
  ```

You'll find final data in `data/shuffled_data_0.h5`.


---

## Quickstart (distributed)

Use this when you want to generate more data, faster. Each worker writes unique part files, so no locking is needed. Generate topologies first (as above), then run multiple workers, then shuffle.

1) Launch multiple workers (local example)

  Open multiple terminals (or background processes) on the same machine; each one uses a different `--worker-id` but the same `--num-workers`:

  ```bash
  # terminal 1
  python generate_data.py --worker-id 0 --num-workers 4 --num-samples 20000 --batch-size 1000

  # terminal 2
  python generate_data.py --worker-id 1 --num-workers 4 --num-samples 20000 --batch-size 1000

  # terminal 3
  python generate_data.py --worker-id 2 --num-workers 4 --num-samples 20000 --batch-size 1000

  # terminal 4
  python generate_data.py --worker-id 3 --num-workers 4 --num-samples 20000 --batch-size 1000
  ```

  Notes
  - Each worker picks its config by `worker_id % num_configs` and writes to a unique file name; no collisions.
  - Adjust `--num-samples` and `--batch-size` to your needs.
  - If `num_workers < num_configs` (currently 32), a single distributed pass will only cover a subset of configs. To cover all configs, either:
    - use `--cycle-all` (works even on a single CPU) to iterate every config per cycle, or
    - run multiple passes changing `--config-index` or your `--worker-id` ranges to reach the remaining configs, or
    - increase `--num-workers` to be ≥ the number of configs.

2) Launch via SLURM (optional, auto-detected)

  The script auto-detects SLURM and uses `SLURM_PROCID/SLURM_NTASKS` or `SLURM_ARRAY_TASK_ID/SLURM_ARRAY_TASK_COUNT`. Typical patterns:

  - MPI-style launch:

    ```bash
    srun -n 8 python generate_data.py --num-samples 50000 --batch-size 1000
    ```

  - Job array launch (index range 0..7): inside your job script just call:

    ```bash
    python generate_data.py --num-samples 50000 --batch-size 1000
    ```

3) Shuffle into shards

  After all workers finish, build as many shards as you want (e.g., 10):

  ```bash
  python make_shuffled_splits.py --data-split 10 --seed 123
  ```

---

## 1) Generate topologies

Topologies are abstract edge patterns per configuration; they’re reused by the data generator.

- Output files: `topologies/LONG_DEG2.txt`, `topologies/SHORT_DEG2.txt`, `topologies/LONG_DEG1.txt`, `topologies/SHORT_DEG1.txt`

Examples

- One topology for config A (LONG_DEG2):

  ```bash
  python generate_topologies.py --config A --num_tops 1
  ```

- Many topologies, with verbose rejections every so often:

  ```bash
  python generate_topologies.py --config B --num_tops 100 --verbose-invalid --progress-every 50000
  ```

Notes

- The four configs map to filenames as follows: A→LONG_DEG2, B→SHORT_DEG2, C→LONG_DEG1, D→SHORT_DEG1.
- Files are appended; rerunning adds more lines.

---

## 2) Generate data (samples)

Builds concrete graphs by sampling colors/weights, validates for N ∈ {0,1,2},
constructs state strings, normalizes each N-segment (GCD reduction + lexicographic sort of kets),
and tokenizes both code and state. Writes fixed-shape HDF5 datasets.

Output layout

- Per configuration directory under `data/`, named by config tuple:
  `data/<CODELEN>_<DEGREE>_<DIMENSION>_<EDGEWEIGHT>_<MAX_KETS>/`
- Each run writes to a unique file: `<run_id>.h5` (and `<run_id>.txt` log)
- Datasets per file: `code`, `state`, `topology_ind`, `line_count`, `num_kets`, `num_zero_kets`, `num_pms`, `degrees_list`

Key flags

- Selection and parallelism
  - `--worker-id`, `--num-workers`: universal distribution. Defaults auto-detect SLURM, else single worker.
  - `--config-index`: force a specific configuration index (overrides worker routing).
  - `--cycle-all`: iterate all configurations, one batch per config per cycle.
  - `--cycles`: number of cycles when using `--cycle-all` (0 = infinite).
- Generation control
  - `--num-samples`: total valid samples to produce (normal mode).
  - `--batch-size`: write batch size and, in cycle-all mode, also the per-config sample count per cycle.
  - `--max-toks`: max token length (code/state are padded to this).
  - `--max-time`: max seconds per config in cycle-all mode (useful for demos or time-bounded runs).
  - `--seed`: RNG seed for reproducibility.
- Metadata
  - `--run-id`: optional custom file stem for outputs.

Examples

- Single worker, pick config by worker id (default = 0), produce 50k samples total:

  ```bash
  python generate_data.py --num-samples 50000 --batch-size 1000
  ```

- Distributed (8 workers), each writing unique files and covering different configs:

  ```bash
  # worker 0 of 8
  python generate_data.py --worker-id 0 --num-workers 8 --num-samples 50000
  # worker 1 of 8
  python generate_data.py --worker-id 1 --num-workers 8 --num-samples 50000
  # ...
  ```

- Single-CPU cycle of all configurations (one batch per config), repeat 3 cycles:

  ```bash
  python generate_data.py --cycle-all --cycles 3 --batch-size 1000
  ```

Notes

- Files are unique (`<timestamp>_<pid>_<worker>` by default). No two processes write to the same file.
- State normalization here is NOT amplitude normalization; `getState(normalize=False)` keeps integer amplitudes, then strings are canonicalized (GCD + sorted kets) before tokenization.
- If you run in single-config mode with fewer workers than configs (32), some configs won't be produced in one pass. Use `--cycle-all` to visit every config, or do multiple passes selecting different configs via `--config-index`.

---

## 3) Make shuffled splits (final shards)

Builds `data/shuffled_data_*.h5` by taking an even slice from each configuration directory
(consistent with the old combine + shuffle pipeline), concatenating for each split, and shuffling rows.

- Output files: `data/shuffled_data_0.h5, ..., data/shuffled_data_{DATA_SPLIT-1}.h5`
- By default, temporary `split_data_*.h5` are removed; keep them with `--keep-intermediate`.

Example

```bash
python make_shuffled_splits.py --data-split 10 --seed 123
```

Options

- `--data-dir`: base data directory (default: `data`)
- `--data-split`: number of shards to produce
- `--seed`: shuffle seed for reproducibility
- `--keep-intermediate`: retain `split_data_*.h5`

Notes

- The script discovers per-config output directories automatically and reads all part files inside.
- It preserves dataset names, shapes, and dtypes; file-level HDF5 attributes are not copied (matching the old behavior).

---

## Dataset contract

- `code`, `state`: int8 arrays of shape (N, max_toks), left-padded with zeros (<PAD>)
- `topology_ind`: (N, 2) — [topology_letter, line_index]
- `line_count`: (N, 2) — number of lines in layer_0/layer_1 for that sample
- `num_kets`, `num_zero_kets`, `num_pms`: (N, 3)
- `degrees_list`: (N, 3, 8) — per-N degree histogram (padded with zeros)

---

## Typical flow summary

1) Generate enough topologies for each class you plan to use:

   ```bash
   python generate_topologies.py --config A --num_tops 1000
   python generate_topologies.py --config B --num_tops 1000
   python generate_topologies.py --config C --num_tops 1000
   python generate_topologies.py --config D --num_tops 1000
   ```

2) Generate samples (parallel or single-CPU cycle-all), producing part files per configuration under `data/`.

3) Create final shuffled shards for training/eval:

   ```bash
   python make_shuffled_splits.py --data-split 10 --seed 123
   ```

That’s it — the final `shuffled_data_*.h5` files are what training usually consumes.
