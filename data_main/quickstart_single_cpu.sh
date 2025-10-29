#!/usr/bin/env bash
# Minimal end-to-end demo for the data pipeline.
# Run from the repository root or from this script's directory.

set -e
cd "$(dirname "$0")"

# 1) Create one topology for each config (A-D). Usually you generate many more; this is slow.
echo "[1/3] Generating 1 topology per config..."
python generate_topologies.py --config A --num_tops 1
python generate_topologies.py --config B --num_tops 1
python generate_topologies.py --config C --num_tops 1
python generate_topologies.py --config D --num_tops 1

# 2) Generate 10 samples per data-gen config with 30s max per config (demo only)
echo "[2/3] Generating 10 samples per config (cycle-all, 30s per config)..."
python generate_data.py --cycle-all --cycles 1 --batch-size 10 --max-time 30

# 3) Shuffle into a single shard (increase --data-split for multiple shards)
echo "[3/3] Building 1 shuffled shard..."
python make_shuffled_splits.py --data-split 1

echo "[done] Final shard: data/shuffled_data_0.h5"