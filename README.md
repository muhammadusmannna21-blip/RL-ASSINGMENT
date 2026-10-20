# RL Custom Agent Implementation - FrozenLake-v1

This repository contains the implementation of a Q-Learning agent trained on Farama Gymnasium's `FrozenLake-v1` environment.

## Project Structure
- `agent.py`: Implementation of the Q-Learning Agent.
- `main.py`: Script to train the agent and save model weights.
- `demo.py`: Script to generate and save a GIF of the trained agent.
- `utils.py`: Utility functions for plot generation and environment rendering.
- `models/`: Saved Q-table (`q_table.npy`).
- `notebooks/`: Jupyter Notebook with executed training runs and visualization plots.
- `results/`: Output visual demonstration (`agent_demo.gif`).

## Setup & Execution
1. Install dependencies:
   ```bash
   pip install -r requirements.txt