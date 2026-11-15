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

"""Run target_rl GRPO training on one Modal container with 8 GPUs.

Modal only provides the machine: the container runs the same train.sh a dedicated GPU box runs. The image is
the NeMo RL release image with this checkout copied to /opt/nemo-rl and the sibling NeMo Gym checkout
(../nemogym, branch target-rl) copied over its 3rdparty/Gym-workspace/Gym submodule, so local edits to
either repo are what trains.

    modal run --detach examples/nemo_gym/target_rl/modal_app.py                       # full run
    modal run --detach examples/nemo_gym/target_rl/modal_app.py --overrides "++grpo.max_num_steps=3"

Needs a Modal secret named `target-rl` holding HF_TOKEN (and WANDB_API_KEY if wandb logging is on).
"""

import shlex
import subprocess
from pathlib import Path

import modal

REMOTE_NEMO_RL = "/opt/nemo-rl"
REMOTE_NEMO_GYM = f"{REMOTE_NEMO_RL}/3rdparty/Gym-workspace/Gym"
CACHE = "/cache"
RESULTS = "/results"
HOURS = 60 * 60

# Local paths only exist on the machine that launches the app; inside the container the code is already in the image.
NEMO_RL = Path(__file__).resolve().parents[3] if modal.is_local() else Path(REMOTE_NEMO_RL)
NEMO_GYM = NEMO_RL.parent / "nemogym"

# Local build products, caches and per-server virtualenvs stay out of the image (uv rebuilds them in the container),
# and so does env.yaml, which holds local endpoint credentials.
IGNORE = [".git", "**/.venv", "**/__pycache__", "results", "logs", ".cache", "cache", "**/*.egg-info", "env.yaml"]
UV_VERSION = "0.13.6"
UV_SYNC = "uv sync --frozen --all-groups --extra nemo_gym"
ACTOR_ENVIRONMENTS = "/opt/target_rl/actor_environments.py"
BUILD_ACTOR_VENVS = "/opt/target_rl/build_actor_venvs.sh"

# Dependencies first, from the lock file and the pinned submodules only, so editing code or configs reuses this layer.
dependencies_image = (
    modal.Image.from_registry("nvcr.io/nvidia/nemo-rl:v0.8.0")
    .entrypoint([])
    # Copying a checkout over the release image's own merges the trees, leaving files the checkout no longer has
    # (a stale requirements.txt beside a pyproject.toml breaks NeMo Gym's server setup). Clear the release image's
    # source first; this layer and the code layer below bring in this checkout's.
    .run_commands(f"find {REMOTE_NEMO_RL} -mindepth 1 -maxdepth 1 -exec rm -rf {{}} +")
    # The release image's uv is older than the `required-version` in NeMo Gym's pyprojects; use the version
    # docker/Dockerfile pins (UV_VERSION).
    .run_commands(
        f"curl -LsSf https://astral.sh/uv/{UV_VERSION}/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh"
    )
    .add_local_file(NEMO_RL / "pyproject.toml", f"{REMOTE_NEMO_RL}/pyproject.toml", copy=True)
    .add_local_file(NEMO_RL / "uv.lock", f"{REMOTE_NEMO_RL}/uv.lock", copy=True)
    .add_local_file(NEMO_RL / ".python-version", f"{REMOTE_NEMO_RL}/.python-version", copy=True)
    .add_local_dir(NEMO_RL / "3rdparty", f"{REMOTE_NEMO_RL}/3rdparty", ignore=[*IGNORE, "Gym-workspace/Gym"], copy=True)
    .add_local_file(NEMO_RL / "nemo_rl/distributed/actor_environments.py", ACTOR_ENVIRONMENTS, copy=True)
    .add_local_file(NEMO_RL / "examples/nemo_gym/target_rl/build_actor_venvs.sh", BUILD_ACTOR_VENVS, copy=True)
    .workdir(REMOTE_NEMO_RL)
    .run_commands(
        f"{UV_SYNC} --no-install-project --no-install-package nemo_gym",
        # The release image's Ray actor venvs were built for its own lock; rebuild them for this checkout's.
        f"bash {BUILD_ACTOR_VENVS} {ACTOR_ENVIRONMENTS} --no-install-package nemo_gym",
    )
)

# Then the checkouts, with the pinned BioDiscoveryRL repos beside NeMo Gym (the biodiscovery resources server depends
# on ../../../{tasks,contracts,generator}); this sync only installs the projects themselves. Agent commands run in
# separate episode sandboxes (target_rl/configs/sandbox_modal.yaml), never in this container.
GYM_WORKSPACE = f"{REMOTE_NEMO_RL}/3rdparty/Gym-workspace"
image = (
    dependencies_image.add_local_dir(NEMO_RL, REMOTE_NEMO_RL, ignore=IGNORE, copy=True)
    .add_local_dir(NEMO_GYM, REMOTE_NEMO_GYM, ignore=IGNORE, copy=True)
    .add_local_dir(NEMO_RL.parent / "tasks", f"{GYM_WORKSPACE}/tasks", ignore=IGNORE, copy=True)
    .add_local_dir(NEMO_RL.parent / "contracts", f"{GYM_WORKSPACE}/contracts", ignore=IGNORE, copy=True)
    .add_local_dir(NEMO_RL.parent / "generator", f"{GYM_WORKSPACE}/generator", ignore=IGNORE, copy=True)
    .run_commands(
        # NeMo Gym builds with egg_base = "cache", a checked-in folder IGNORE leaves out.
        f"mkdir -p {REMOTE_NEMO_GYM}/cache",
        UV_SYNC,
        # Second half of the actor venv build, as in docker/Dockerfile's release stage: the editable nemo_rl and
        # nemo_gym installs and the python-<Actor> wrappers, now that the source is here. The first half (every
        # third-party package) ran in the dependency layer.
        "UV_LINK_MODE=hardlink uv run nemo_rl/utils/prefetch_venvs.py $(cut -f1 /tmp/actor_venvs.tsv)",
        f"chmod -R o-rwx {REMOTE_NEMO_RL} /root",
    )
)

# Mounted volumes keep their permissions from the volume; checkpoints and caches stay private to the trainer.
RESTRICT_VOLUMES = f"chmod 700 {CACHE} {RESULTS}"

app = modal.App("target-rl-grpo")
cache_volume = modal.Volume.from_name("target-rl-hf-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("target-rl-results", create_if_missing=True, version=2)


# The machine both the training Function and the dev Sandbox (modal_devbox.py) run on.
CONTAINER = dict(
    image=image,
    gpu="H100:8",
    cpu=64,
    memory=512 * 1024,
    volumes={CACHE: cache_volume, RESULTS: results_volume},
    secrets=[modal.Secret.from_name("target-rl")],
)


@app.function(**CONTAINER, timeout=24 * HOURS)
def train(overrides: str = "", config: str = "") -> None:
    env_prefix = f"CONFIG={shlex.quote(config)} " if config else ""
    command = (
        f"{RESTRICT_VOLUMES} && {env_prefix}HF_HOME={CACHE}/huggingface RESULTS_ROOT={RESULTS} "
        f"examples/nemo_gym/target_rl/train.sh {overrides}"
    )
    try:
        subprocess.run(["bash", "-c", command], cwd=REMOTE_NEMO_RL, check=True)
    finally:
        cache_volume.commit()
        results_volume.commit()


@app.local_entrypoint()
def main(overrides: str = "", config: str = "") -> None:
    if not NEMO_GYM.is_dir():
        raise SystemExit(f"Expected the NeMo Gym checkout at {NEMO_GYM}")
    train.remote(overrides=overrides, config=config)
