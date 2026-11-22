#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Launch GRPO with NeMo Gym from the NeMo RL repo root, on whatever node runs it (a dedicated GPU box,
# a container, or the Modal function in modal_app.py). Extra arguments are Hydra overrides.
#
#   examples/nemo_gym/target_rl/train.sh ++grpo.max_num_steps=3
#   CONFIG=examples/nemo_gym/target_rl/other.yaml examples/nemo_gym/target_rl/train.sh
set -euo pipefail

CONFIG="${CONFIG:-examples/nemo_gym/target_rl/grpo_qwen3.8_27b_1n8g.yaml}"
EXP_NAME="${EXP_NAME:-target-rl-$(date +%Y%m%d-%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
export HF_HOME="${HF_HOME:-$PWD/.cache}"

if [ ! -f 3rdparty/Gym-workspace/Gym/pyproject.toml ]; then
  git submodule update --init --recursive 3rdparty/Gym-workspace/Gym
fi

mkdir -p "$RESULTS_ROOT/$EXP_NAME"
uv run ray stop --force >/dev/null 2>&1 || true

uv run python examples/nemo_gym/run_grpo_nemo_gym.py \
  --config "$CONFIG" \
  ++logger.log_dir="$RESULTS_ROOT/$EXP_NAME/logs" \
  ++checkpointing.checkpoint_dir="$RESULTS_ROOT/$EXP_NAME/checkpoints" \
  "$@" 2>&1 | tee "$RESULTS_ROOT/$EXP_NAME/output.log"
