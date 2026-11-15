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

"""A long-lived 8-GPU Modal Sandbox for iterating on training without rebuilding the image.

The Sandbox uses the same image, volumes and secret as modal_app.py. Start it once, push local edits with `sync`
(only files changed since the last sync), and rerun commands with `run`. It stops after 30 idle minutes.

    uv run --no-project --with modal python examples/nemo_gym/target_rl/modal_devbox.py up
    uv run --no-project --with modal python examples/nemo_gym/target_rl/modal_devbox.py sync
    uv run --no-project --with modal python examples/nemo_gym/target_rl/modal_devbox.py run "examples/nemo_gym/target_rl/train.sh ++grpo.max_num_steps=3"
    uv run --no-project --with modal python examples/nemo_gym/target_rl/modal_devbox.py status
    uv run --no-project --with modal python examples/nemo_gym/target_rl/modal_devbox.py down
"""

import argparse
import importlib.util
import io
import json
import sys
import tarfile
import time
from pathlib import Path

import modal

STATE = Path.home() / ".cache" / "target-rl" / "devbox.json"
IDLE_MINUTES = 30
HOURS = 60 * 60
SKIP_DIRS = {".git", ".venv", "__pycache__", "results", "logs", ".cache", "cache"}
SKIP_FILES = {"env.yaml"}


def load_modal_app():
    path = Path(__file__).with_name("modal_app.py")
    spec = importlib.util.spec_from_file_location("target_rl_modal_app", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def write_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state))


def sandbox_from_state() -> modal.Sandbox:
    state = read_state()
    if "sandbox_id" not in state:
        raise SystemExit("No devbox recorded; run `up` first.")
    sandbox = modal.Sandbox.from_id(state["sandbox_id"])
    if sandbox.poll() is not None:
        raise SystemExit(f"Devbox {state['sandbox_id']} has stopped; run `up` again.")
    return sandbox


def changed_files(root: Path, since: float) -> list[Path]:
    out = []
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts) or path.name in SKIP_FILES or not path.is_file():
            continue
        if path.stat().st_mtime > since:
            out.append(path)
    return out


def up(_: argparse.Namespace) -> None:
    state = read_state()
    if "sandbox_id" in state and modal.Sandbox.from_id(state["sandbox_id"]).poll() is None:
        print(f"Devbox already running: {state['sandbox_id']}")
        return
    m = load_modal_app()
    app = modal.App.lookup("target-rl-devbox", create_if_missing=True)
    started = time.time()
    with modal.enable_output():  # streams the image build, if one is needed
        sandbox = modal.Sandbox.create(
            "sleep",
            "infinity",
            app=app,
            **m.CONTAINER,
            workdir=m.REMOTE_NEMO_RL,
            timeout=24 * HOURS,
            idle_timeout=IDLE_MINUTES * 60,
        )
    restrict = sandbox.exec("bash", "-c", m.RESTRICT_VOLUMES)
    restrict.wait()
    write_state({"sandbox_id": sandbox.object_id, "synced_at": started})
    print(f"Devbox up: {sandbox.object_id} (stops after {IDLE_MINUTES} idle minutes)")


def sync(_: argparse.Namespace) -> None:
    m = load_modal_app()
    sandbox = sandbox_from_state()
    state = read_state()
    since, now = state["synced_at"], time.time()
    buffer, count = io.BytesIO(), 0
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for root, remote in ((m.NEMO_RL, m.REMOTE_NEMO_RL), (m.NEMO_GYM, m.REMOTE_NEMO_GYM)):
            for path in changed_files(root, since):
                tar.add(path, arcname=f"{remote.lstrip('/')}/{path.relative_to(root)}")
                count += 1
    if not count:
        print("Nothing changed since the last sync.")
        return
    sandbox.filesystem.write_bytes(buffer.getvalue(), "/tmp/devbox_sync.tar.gz")
    process = sandbox.exec("tar", "-xzf", "/tmp/devbox_sync.tar.gz", "-C", "/")
    process.wait()
    if process.returncode:
        raise SystemExit(f"Extracting the sync archive failed: {process.stderr.read()}")
    write_state({**state, "synced_at": now})
    print(f"Synced {count} files.")


def run(args: argparse.Namespace) -> None:
    sandbox = sandbox_from_state()
    m = load_modal_app()
    command = f"HF_HOME={m.CACHE}/huggingface RESULTS_ROOT={m.RESULTS} {args.command}"
    process = sandbox.exec("bash", "-lc", command, workdir=m.REMOTE_NEMO_RL)
    for line in process.stdout:
        print(line, end="", flush=True)
    process.wait()
    stderr = process.stderr.read()
    if stderr:
        print(stderr, end="", file=sys.stderr)
    sys.exit(process.returncode)


def status(_: argparse.Namespace) -> None:
    state = read_state()
    if "sandbox_id" not in state:
        print("No devbox recorded.")
        return
    code = modal.Sandbox.from_id(state["sandbox_id"]).poll()
    print(f"{state['sandbox_id']}: {'running' if code is None else f'stopped (exit {code})'}")


def down(_: argparse.Namespace) -> None:
    state = read_state()
    if "sandbox_id" in state:
        modal.Sandbox.from_id(state["sandbox_id"]).terminate()
        print(f"Terminated {state['sandbox_id']}")
    STATE.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    for name, handler in (("up", up), ("sync", sync), ("status", status), ("down", down)):
        sub.add_parser(name).set_defaults(handler=handler)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("command")
    run_parser.set_defaults(handler=run)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
