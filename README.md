## MCTS Deep Reinforcement Learning

This is a simple implementation of MCTS Deep Reinforcement Learning for the game of Tic-Tac-Toe.
Chess was never implemented because it is too complex given the poor performance with Tic-Tac-Toe.
Project is not finished. It is abandoned.
Implementation will be rewritten from scratch in Python.

## Project Structure

- `src/`
  - `agents/` - Agent implementations (Random, MCTS, Neural MCTS)
  - `arena/` - Game playing environment
  - `common/` - Shared utilities and interfaces
  - `games/` - Game implementations
  - `mcts/` - Core MCTS algorithm
  - `training/` - Neural network training pipeline
  - `benchmarking/` - Performance measurement tools

- `tests/` - Comprehensive test suite
- `scripts/` - Utility scripts for benchmarking and analysis