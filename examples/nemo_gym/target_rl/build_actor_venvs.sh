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

# Build every Ray actor venv under $NEMO_RL_VENV_DIR from this checkout's uv.lock, the same loop as
# docker/Dockerfile, so a release image can run a newer checkout. Extra arguments are passed to each `uv sync`
# (the image's dependency layer passes --no-install-package nemo_gym; the code layer passes nothing).
#
#   build_actor_venvs.sh <path to actor_environments.py> [uv sync args...]
set -euo pipefail

ACTOR_ENVIRONMENTS="$1"
shift
: "${NEMO_RL_VENV_DIR:?NEMO_RL_VENV_DIR must be set}"

# sglang and trtllm actors are not used by these recipes; skipping them keeps the build short.
"${UV_PROJECT_ENVIRONMENT:-/opt/nemo_rl_venv}/bin/python" "$ACTOR_ENVIRONMENTS" all sglang trtllm > /tmp/actor_venvs.tsv
test -s /tmp/actor_venvs.tsv
while IFS=$'\t' read -r venv_name stage extras; do
  venv_path="${NEMO_RL_VENV_DIR}/${venv_name}"
  uv venv --allow-existing "$venv_path"
  # shellcheck disable=SC2086
  UV_PROJECT_ENVIRONMENT="$venv_path" UV_LINK_MODE=hardlink uv sync --frozen $extras --no-install-project "$@"
done < /tmp/actor_venvs.tsv
