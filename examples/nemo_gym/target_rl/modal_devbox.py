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
(only files changed since the last sync), and start commands with `run`.

`run` is detached: the command keeps running in the Sandbox when this client exits or the laptop sleeps, and its
output goes to /results/devbox-logs/<name>.log on the results Volume (with <name>.exit once it ends), so the log
outlives the Sandbox. A running command counts as Sandbox activity; the Sandbox stops after 30 idle minutes once
nothing runs.

    uv run --no-project --with modal python examples/nemo_gym/target_rl/modal_devbox.py up   # published image, then sync
    uv run --no-project --with modal python examples/nemo_gym/target_rl/modal_devbox.py sync
    uv run --no-project --with modal python examples/nemo_gym/target_rl/modal_devbox.py run "examples/nemo_gym/target_rl/train.sh ++grpo.max_num_steps=3"
    uv run --no-project --with modal python examples/nemo_gym/target_rl/modal_devbox.py logs --follow
    uv run --no-project --with modal python examples/nemo_gym/target_rl/modal_devbox.py stop
    uv run --no-project --with modal python examples/nemo_gym/target_rl/modal_devbox.py status
    uv run --no-project --with modal python examples/nemo_gym/target_rl/modal_devbox.py down

A run's log after the Sandbox is gone: modal volume get target-rl-results devbox-logs/<name>.log
"""

import argparse
import importlib.util
import io
import json
import shlex
import sys
import tarfile
import time
from pathlib import Path

import modal

STATE = Path.home() / ".cache" / "target-rl" / "devbox.json"
IMAGES = Path.home() / ".cache" / "target-rl" / "images.json"
TRAINING_IMAGE = "target-rl-training"
IDLE_MINUTES = 30
HOURS = 60 * 60
SKIP_DIRS = {".git", ".venv", "__pycache__", "results", "logs", ".cache", "cache"}
SKIP_FILES = {"env.yaml"}
LOG_DIR = "/results/devbox-logs"


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


def up(args: argparse.Namespace) -> None:
    state = read_state()
    if "sandbox_id" in state and modal.Sandbox.from_id(state["sandbox_id"]).poll() is None:
        print(f"Devbox already running: {state['sandbox_id']}")
        return
    m = load_modal_app()
    app = modal.App.lookup("target-rl-devbox", create_if_missing=True)
    # Start from the published image (modal_images.py) when there is one: no build, then sync what changed since.
    published = json.loads(IMAGES.read_text()).get(TRAINING_IMAGE) if IMAGES.exists() else None
    container = dict(m.CONTAINER)
    if published and not args.build:
        container["image"] = modal.Image.from_name(TRAINING_IMAGE)
        started = published
    else:
        started = time.time()
    with modal.enable_output():  # streams the image build, if one is needed
        sandbox = modal.Sandbox.create(
            "sleep",
            "infinity",
            app=app,
            **container,
            workdir=m.REMOTE_NEMO_RL,
            timeout=24 * HOURS,
            idle_timeout=IDLE_MINUTES * 60,
        )
    restrict = sandbox.exec("bash", "-c", m.RESTRICT_VOLUMES)
    restrict.wait()
    write_state({"sandbox_id": sandbox.object_id, "synced_at": started})
    print(f"Devbox up: {sandbox.object_id} (stops after {IDLE_MINUTES} idle minutes)")
    if published and not args.build:
        sync(args)


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


def exec_output(sandbox: modal.Sandbox, command: str) -> str:
    process = sandbox.exec("bash", "-c", command)
    process.wait()
    return process.stdout.read().strip()


def run_paths(name: str) -> tuple[str, str, str]:
    base = f"{LOG_DIR}/{name}"
    return f"{base}.log", f"{base}.exit", f"{base}.pid"


def run_state(sandbox: modal.Sandbox, name: str) -> str:
    """'running', 'exited N', or 'unknown' for the named run."""
    log, exit_file, pid_file = run_paths(name)
    return exec_output(
        sandbox,
        f"if [ -s {exit_file} ]; then echo exited $(cat {exit_file}); "
        f"elif [ -s {pid_file} ] && kill -0 $(cat {pid_file}) 2>/dev/null; then echo running; else echo unknown; fi",
    )


def run(args: argparse.Namespace) -> None:
    sandbox = sandbox_from_state()
    m = load_modal_app()
    state = read_state()
    if state.get("run") and run_state(sandbox, state["run"]) == "running":
        raise SystemExit(f"Run {state['run']} is still running; `stop` it first.")
    name = args.name or time.strftime("run-%Y%m%d-%H%M%S")
    log, exit_file, pid_file = run_paths(name)
    command = f"HF_HOME={m.CACHE}/huggingface RESULTS_ROOT={m.RESULTS} {args.command}"
    # setsid gives the run its own process group, which `stop` kills as a whole. The Volume `sync` makes the log
    # and exit code visible outside the Sandbox without waiting for it to stop.
    wrapper = (
        f"mkdir -p {LOG_DIR} && rm -f {exit_file} && echo $$ > {pid_file} && cd {m.REMOTE_NEMO_RL} && "
        f"{{ bash -lc {shlex.quote(command)}; }} > {log} 2>&1; echo $? > {exit_file}; sync {m.RESULTS} || true"
    )
    # Not waited on: the process lives in the Sandbox, not in this client.
    sandbox.exec("setsid", "bash", "-c", wrapper)
    write_state({**state, "run": name})
    print(f"Started {name}; log at {log}")
    if args.follow:
        time.sleep(2)
        follow(sandbox, name, lines=200)


def follow(sandbox: modal.Sandbox, name: str, lines: int) -> None:
    log, _, pid_file = run_paths(name)
    # tail stops when the run's wrapper exits; Ctrl-C stops only this client, never the run.
    process = sandbox.exec(
        "bash", "-c", f"tail -n {lines} -F {log} --pid=$(cat {pid_file} 2>/dev/null || echo 1) 2>/dev/null"
    )
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
    except KeyboardInterrupt:
        pass
    print(f"[{name}: {run_state(sandbox, name)}]")


def logs(args: argparse.Namespace) -> None:
    sandbox = sandbox_from_state()
    name = args.name or read_state().get("run")
    if not name:
        raise SystemExit("No run recorded; pass --name.")
    if args.follow:
        follow(sandbox, name, lines=args.lines)
    else:
        print(exec_output(sandbox, f"tail -n {args.lines} {run_paths(name)[0]}"))
        print(f"[{name}: {run_state(sandbox, name)}]")


def stop(args: argparse.Namespace) -> None:
    sandbox = sandbox_from_state()
    name = args.name or read_state().get("run")
    if not name:
        raise SystemExit("No run recorded; pass --name.")
    _, _, pid_file = run_paths(name)
    # The run's process group, then Ray, whose daemons outlive their parent.
    exec_output(
        sandbox,
        f"[ -s {pid_file} ] && kill -TERM -- -$(cat {pid_file}) 2>/dev/null; sleep 10; "
        f"[ -s {pid_file} ] && kill -KILL -- -$(cat {pid_file}) 2>/dev/null; ray stop --force >/dev/null 2>&1; true",
    )
    print(f"Stopped {name}: {run_state(sandbox, name)}")


def status(_: argparse.Namespace) -> None:
    state = read_state()
    if "sandbox_id" not in state:
        print("No devbox recorded.")
        return
    sandbox = modal.Sandbox.from_id(state["sandbox_id"])
    code = sandbox.poll()
    print(f"{state['sandbox_id']}: {'running' if code is None else f'stopped (exit {code})'}")
    if code is None and state.get("run"):
        print(f"last run {state['run']}: {run_state(sandbox, state['run'])}")


def down(_: argparse.Namespace) -> None:
    state = read_state()
    if "sandbox_id" in state:
        modal.Sandbox.from_id(state["sandbox_id"]).terminate()
        print(f"Terminated {state['sandbox_id']}")
    STATE.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    up_parser = sub.add_parser("up")
    up_parser.add_argument("--build", action="store_true", help="build the image from modal_app.py instead of the published one")
    up_parser.set_defaults(handler=up)
    for name, handler in (("sync", sync), ("status", status), ("down", down)):
        sub.add_parser(name).set_defaults(handler=handler)
    run_parser = sub.add_parser("run", help="start a command in the devbox, detached")
    run_parser.add_argument("command")
    run_parser.add_argument("--name", help="run name for the log file (default: run-<timestamp>)")
    run_parser.add_argument("--follow", action="store_true", help="stream the log after starting")
    run_parser.set_defaults(handler=run)
    logs_parser = sub.add_parser("logs", help="show the last run's log")
    logs_parser.add_argument("--name")
    logs_parser.add_argument("-n", "--lines", type=int, default=100)
    logs_parser.add_argument("--follow", action="store_true")
    logs_parser.set_defaults(handler=logs)
    stop_parser = sub.add_parser("stop", help="stop the last run")
    stop_parser.add_argument("--name")
    stop_parser.set_defaults(handler=stop)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
