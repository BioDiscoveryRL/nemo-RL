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

"""Build and publish the named Modal images, so launches never wait on an image build.

- `target-rl-training`: the training image from modal_app.py (NeMo RL, its actor venvs, both checkouts). The dev
  Sandbox starts from it; code edits after publishing reach a running devbox through `modal_devbox.py sync`.
- `bdrl-toolkit`: the episode sandbox image (NeMo Gym resources_servers/biodiscovery/toolkit/Dockerfile), used by
  target_rl/configs/sandbox_modal.yaml.

Rebuild only when dependencies or the toolkit change:

    uv run --no-project --with modal python examples/nemo_gym/target_rl/modal_images.py            # both
    uv run --no-project --with modal python examples/nemo_gym/target_rl/modal_images.py toolkit    # one
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import modal

TRAINING = "target-rl-training"
TOOLKIT = "bdrl-toolkit"
STATE = Path.home() / ".cache" / "target-rl" / "images.json"


def load_modal_app():
    path = Path(__file__).with_name("modal_app.py")
    spec = importlib.util.spec_from_file_location("target_rl_modal_app", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="*", choices=["training", "toolkit"], default=["training", "toolkit"])
    args = parser.parse_args()
    m = load_modal_app()
    app = modal.App.lookup("target-rl-images", create_if_missing=True)
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    for which in args.images:
        if which == "training":
            name, image = TRAINING, m.image
        else:
            dockerfile = m.NEMO_GYM / "resources_servers" / "biodiscovery" / "toolkit" / "Dockerfile"
            name, image = TOOLKIT, modal.Image.from_dockerfile(dockerfile, context_dir=dockerfile.parent)
        started = time.time()
        with modal.enable_output():
            image.build(app).publish(name)
        state[name] = started
        print(f"Published {name}")
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state))


if __name__ == "__main__":
    main()
