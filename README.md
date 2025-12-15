## Table of Contents
- [Setup](#setup)
- [Quick Start](#quick-start)
- [Running Policy Agreement Analysis](#running-policy-agreement-analysis)
- [Running Simulated Play](#running-simulated-play)
- [Configuration Files](#configuration-files)
- [Project Structure](#project-structure)

## Setup

### Prerequisites
- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone and navigate to the project**:
   ```bash
   git clone <repository-url>
   cd intuitive-gamer-memo
   ```

3. **Create and activate the uv environment**:
   ```bash
   uv venv
   source .venv/bin/activate  # On Linux/Mac
   # or .venv\Scripts\activate on Windows
   ```

4. **Install dependencies**:
   ```bash
   uv sync
   ```

## Quick Start

The project includes two main executable scripts:

1. **Policy Agreement Analysis** (`policy_agreement.py`) - Analyzes agreement between different policies on game states
2. **Simulated Play** (`simulated_play.py`) - Simulates games between policies and collects statistics

## Running Policy Agreement Analysis

Policy agreement analysis compares how different game-playing policies make decisions on various game states.

### Basic Usage

```bash
python policy_agreement.py --config config/baseline.yaml
```

### Sample Configuration

The `config/baseline.yaml` file provides a good starting point:

```yaml
experiment_name: "baseline_policy_comparison"
description: "Compare different game policies on standard games"

game:
  name: "mnk_game"
  parameters:
    m: 3
    n: 3
    k: 3

policies:
  - name: "intuitive_gamer"
    parameters: {}
  - name: "random"
    parameters:
      seed: 42

# Additional configuration options...
```

### What it does:
- Generates game states from the specified game
- Evaluates each policy's decision-making on these states
- Calculates agreement metrics between policies
- Outputs results to CSV files in the `results/` directory

## Running Simulated Play

Simulated play runs actual games between two policies and collects outcome statistics.

### Basic Usage

```bash
python simulated_play.py --config config/simulation_example.yaml
```

### Sample Configuration

The `config/simulation_example.yaml` file demonstrates simulated gameplay:

```yaml
experiment_name: "policy_vs_policy_simulation"
description: "Simulate games between two specific policies"

num_trials: 50
alternate_starting_player: true
track_posteriors: true

policies:
  - name: "intuitive_gamer"
    parameters: {}
  
  - name: "intuitive_gamer"
    parameters:
      opponent_inference:
        enabled: true
        method: "log_likelihood"
        candidate_policies:
          - name: "random"
            parameters:
              seed: 42

games:
  - name: "tic_tac_toe"
    parameters: {}
```

### What it does:
- Runs multiple game trials between the specified policies
- Tracks win rates, game lengths, and other statistics
- Optionally tracks posterior probabilities for opponent inference
- Generates plots and saves results to CSV files

## Configuration Files

The `config/` directory contains several example configurations:

- **`baseline.yaml`** - Basic policy comparison setup
- **`simulation_example.yaml`** - Simulated gameplay configuration
- **`simulation.yaml`** - Alternative simulation setup
- **`multi_game_baseline.yaml`** - Multi-game comparison setup

### Key Configuration Options

#### Game Settings
```yaml
game:
  name: "tic_tac_toe"  # or "mnk_game", etc.
  parameters:
    m: 3  # board width (for mnk_game)
    n: 3  # board height
    k: 3  # pieces in a row to win
```

#### Policy Settings
```yaml
policies:
  - name: "intuitive_gamer"
    parameters: {}
  - name: "random"
    parameters:
      seed: 42
  - name: "mcts"
    parameters:
      iterations: 1000
```

#### Simulation Settings
```yaml
num_trials: 100
alternate_starting_player: true
track_posteriors: false
```

## Project Structure

```
├── config/                     # Configuration files
├── games/                      # Game implementations
├── policies/                   # Policy implementations
│   ├── intuitivegamer/        # Intuitive Gamer policy
│   ├── mcts/                  # MCTS policy
│   └── random/                # Random policy
├── opponent_inference/         # Opponent inference algorithms
├── state_dataset/             # Game state dataset generation
├── utils/                     # Utility functions and plotting
├── results/                   # Output results (CSV files)
├── policy_agreement.py        # Main policy agreement analysis script
├── simulated_play.py         # Main simulated play script
└── pyproject.toml            # Project dependencies
```

## Available Policies

- **`intuitive_gamer`** - The main intuitive gaming policy
- **`random`** - Random move selection
- **`mcts`** - Monte Carlo Tree Search policy

## Output

Both scripts generate results in the `results/` directory:
- CSV files with detailed statistics
- Plots and visualizations (when applicable)
- Log files with experiment details

## Troubleshooting

1. **Import errors**: Make sure you've activated the uv environment and installed dependencies
2. **Configuration errors**: Check YAML syntax and required fields
3. **Game not found**: Verify the game name is supported by OpenSpiel or implemented in `games/`

## Development

To add new policies or games, see the respective directories and follow the existing patterns. The policy registry system automatically discovers new implementations.