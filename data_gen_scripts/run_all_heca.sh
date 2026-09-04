#!/usr/bin/env bash
set -euo pipefail

# Run heca scene data generation for all gt-scene envs.
cd "$(dirname "$0")/.."

ENVS=(
    vi-scene0
    gt-scene1
    gt-scene2
    gt-scene3
    gt-scene4
    gt-scene5
    gt-scene6
    gt-scene7
    gt-scene8
    gt-scene9
    gt-scene10
)

NUM_EPISODES=250
DATASET_TYPE=play

for env_name in "${ENVS[@]}"; do
    echo "=== Running ${env_name} ==="
    python data_gen_scripts/generate_heca_scene.py \
        --env_name="${env_name}" \
        --num_episodes="${NUM_EPISODES}" \
        --dataset_type="${DATASET_TYPE}"
done

echo "All environments done."
