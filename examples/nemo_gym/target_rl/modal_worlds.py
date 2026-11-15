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

"""Stage BioDiscoveryRL worlds into a Modal Volume, straight from Hugging Face.

The volume keeps the dataset's own layout (worlds/<id>/{dbGaP,wgs}, shared/, harbor/, splits/) at a pinned
revision, so training containers mount the same files a dedicated box has under data/biodiscoveryrl/.
Each world is committed as soon as it lands; rerunning skips worlds already present.

    modal run examples/nemo_gym/target_rl/modal_worlds.py                         # every world
    modal run examples/nemo_gym/target_rl/modal_worlds.py --worlds w930004,w933000

Needs the `target-rl` Modal secret (HF_TOKEN).
"""

import modal

REPO = "sammargolis/biodiscoveryrl-overfit18"
REVISION = "9c3c4eb1e3c7dd3781f361ab4fb8335b30a52cf6"
WORLDS = "/worlds"
HOURS = 60 * 60

image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("huggingface_hub[hf_xet]==2.4.0")
app = modal.App("target-rl-worlds")
worlds_volume = modal.Volume.from_name("target-rl-worlds", create_if_missing=True, version=2)


@app.function(
    image=image,
    cpu=8,
    memory=16 * 1024,
    volumes={WORLDS: worlds_volume},
    secrets=[modal.Secret.from_name("target-rl")],
    timeout=6 * HOURS,
)
def stage(worlds: list[str]) -> dict[str, str]:
    from pathlib import Path

    from huggingface_hub import HfApi, snapshot_download

    def fetch(patterns: list[str]) -> None:
        snapshot_download(repo_id=REPO, repo_type="dataset", revision=REVISION, local_dir=WORLDS,
                          allow_patterns=patterns, max_workers=8)

    fetch(["shared/**", "harbor/**", "splits/**", "README.md"])
    worlds_volume.commit()

    if not worlds:
        listed = HfApi().list_repo_tree(REPO, repo_type="dataset", revision=REVISION, path_in_repo="worlds")
        worlds = sorted(Path(entry.path).name for entry in listed)

    status = {}
    for world in worlds:
        root = Path(WORLDS, "worlds", world)
        if (root / "dbGaP").is_dir() and (root / "wgs").is_dir() and (root / ".staged").exists():
            status[world] = "already staged"
            continue
        fetch([f"worlds/{world}/**"])
        for part in ("dbGaP", "wgs"):
            if not (root / part).is_dir():
                raise RuntimeError(f"{part} missing for {world} at {REPO}@{REVISION}")
        (root / ".staged").write_text(REVISION)
        worlds_volume.commit()
        status[world] = "staged"
        print(f"{world}: staged", flush=True)
    return status


@app.local_entrypoint()
def main(worlds: str = "") -> None:
    selected = [w.strip() for w in worlds.split(",") if w.strip()]
    for world, state in stage.remote(selected).items():
        print(f"{world}: {state}")
