# On-Call Agent

Reinforcement learning environment for training language model-based agents to simulate on-call incident investigation workflows.

## Features

- Simulated incident investigation environment with synthetic logs, status pages, deployment history, and Slack messages.
- Reward function combining diagnostic accuracy and investigation efficiency.

## How to use

- Original from scratch implementation at src/ basically works but is extremely slow so not ideal for GRPO given the need to generate long multiturn,
- **Recommended**: Use src/thanks_will/ which is the same environment but uses the verifiers package which natively supports a lot of nice things like overlappted training + inference, inference through vLLM, nice environment and rubric abstractions, and tons more

## Getting Started

```bash
cd src/thanks_will
CUDA_VISIBLE_DEVICES=0 uv run vf-verifiers --model Qwen/Qwen3-4B --tensor-parallel-size 1
CUDA_VISIBLE_DEVICES=1 uv run accelerate launch --config-file configs/zero3.yaml main.py

```

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.
