# On-Call Agent

Reinforcement learning framework for training language model-based agents to simulate on-call incident investigation workflows.

## Features

- Simulated incident investigation environment with synthetic logs, status pages, deployment history, and Slack messages.
- Reward function combining diagnostic accuracy and investigation efficiency.
- Configurable training, evaluation, and single-run demo modes.
- Extensible and modular design.

## Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

> **Note:** Ensure that PyTorch is installed (CPU or GPU version) as required by your setup:
> ```bash
> pip install torch torchvision
> ```

## Quickstart

### Generate a sample training configuration

```bash
python -m src.main --mode config
# Edit training_config.json to customize parameters
```

### Training

```bash
python -m src.main --mode train --config training_config.json
```

### Evaluation

```bash
python -m src.main --mode eval --scenarios 20
```

### Demo

```bash
python -m src.main --mode demo
```

## Project Structure

```
.
├── pyproject.toml        # Project metadata and dependencies
├── README.md             # This file
├── notes.md              # Design notes and rationale
└── src
    ├── environment.py    # Simulation environment and scenario generator
    ├── trainer.py        # Training routines (GRPO training)
    ├── main.py           # Entry point and CLI
    ├── ...
    └── tests             # Unit tests
```

## Configuration

All training parameters can be customized via the generated `training_config.json` file, including model name, batch size, number of episodes, reward weights, and more.

## Testing

Run the unit tests with:

```bash
pytest
```

## Design Notes

Architectural and algorithmic design details, including environment design, reward function rationale, and planned ablation studies, can be found in [notes.md](notes.md).

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.
