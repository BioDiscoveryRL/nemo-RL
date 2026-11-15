# target_rl

GRPO training for drug-target discovery agents: NeMo RL trains the policy, NeMo Gym (our fork, branch `target-rl`, at `3rdparty/Gym-workspace/Gym`) runs the episodes and scores them.

The architecture (repos and what each owns, how they communicate, inference engines, compute, status) is in the NeMo Gym fork: [`target_rl/ARCHITECTURE.md`](../../../3rdparty/Gym-workspace/Gym/target_rl/ARCHITECTURE.md), also on GitHub at [BioDiscoveryRL/NeMo-Gym@target-rl](https://github.com/BioDiscoveryRL/NeMo-Gym/blob/target-rl/target_rl/ARCHITECTURE.md).

| File | What it is |
|---|---|
| `grpo_qwen3.8_27b_1n8g.yaml` | Recipe: Qwen3.8-27B with thinking, LoRA, TP4 x PP2 on one 8-GPU node; toy Gym environment and smoke data for now. |
| `train.sh` | Launches training from the repo root on the current machine. Same command on a dedicated box, in a container, or inside Modal. |
| `modal_app.py` | Runs `train.sh` on one Modal container with 8 H100s. Copies this checkout and `../nemogym` into the NeMo RL release image. |
| `modal_worlds.py` | Stages BioDiscoveryRL worlds from Hugging Face into the `target-rl-worlds` Modal volume, at a pinned dataset revision. |

## Dedicated GPU box

```bash
git submodule update --init 3rdparty/Gym-workspace/Gym   # our Gym fork, branch target-rl
uv sync --all-groups --extra nemo_gym
export HF_TOKEN=...
examples/nemo_gym/target_rl/train.sh ++grpo.max_num_steps=3   # smoke run
```

## Modal

```bash
modal secret create target-rl HF_TOKEN=...          # once
modal run --detach examples/nemo_gym/target_rl/modal_app.py --overrides "++grpo.max_num_steps=3"
```

Model weights cache in the `target-rl-hf-cache` volume; logs and checkpoints land in `target-rl-results`.

## Notes

- Training uses NeMo RL's own vLLM for generation; the SGLang endpoints on Modal are for inference checks only.
- LoRA on the Qwen3.5 architecture in Megatron is not yet exercised by an upstream recipe; the first smoke run checks it. The upstream full fine-tune recipe for this model needs two nodes.

## Watching dynamic sampling lag

With `grpo.dynamic_sampling_refill: true` (the default), groups whose rollouts all score the same are dropped and
regenerated, so a step can wait on many extra rollouts when most tasks are too easy or too hard. Each step logs:

- `train/dynamic_sampling/rollouts_per_kept_group`: prompt groups generated per group trained on (1.0 = nothing
  dropped, 3.0 = the step waited for three times the rollouts it trains on).
- `train/dynamic_sampling/drop_fraction`, `dropped_groups`, `kept_groups`, `consecutive_dropped_groups`.

Above 50% dropped, the log prints `⚠️ Dynamic sampling lag`; after `dynamic_sampling_max_gen_batches x
num_prompts_per_step` drops in a row, collection stops with an error. `grep -E "Dynamic sampling lag|Dropped prompt
group" <log>` shows both. If the lag stays high, set `++grpo.dynamic_sampling_refill=false` on the next run: steps then
train on whichever groups had signal instead of waiting for replacements.

