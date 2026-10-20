
import os
import matplotlib.pyplot as plt
import numpy as np


def save_training_plot(rewards, output_path="results/training_rewards.png"):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.35, label="Episode reward")

    window = min(100, len(rewards))

    if window > 0:
        moving_average = np.convolve(
            rewards,
            np.ones(window) / window,
            mode="valid"
        )
        plt.plot(
            np.arange(window - 1, len(rewards)),
            moving_average,
            label=f"{window}-episode moving average"
        )

    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("FrozenLake Q-Learning Training Progress")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Training graph saved to: {output_path}")
