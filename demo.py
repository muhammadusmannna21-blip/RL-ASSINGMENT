
import os
import numpy as np
import gymnasium as gym
import imageio.v2 as imageio

# Create the same environment used during training.
# If main.py specifies is_slippery=False, keep it False here too.
env = gym.make(
    "FrozenLake-v1",
    map_name="4x4",
    is_slippery=False,
    render_mode="rgb_array"
)

# Load the trained Q-table
q_table = np.load("models/q_table.npy")

os.makedirs("results", exist_ok=True)

frames = []
state, info = env.reset(seed=42)
terminated = False
truncated = False
total_reward = 0

# Capture the initial state
frames.append(env.render())

while not (terminated or truncated):
    action = int(np.argmax(q_table[state]))

    state, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    frames.append(env.render())

# Save the demonstration GIF
imageio.mimsave(
    "results/agent_demo.gif",
    frames,
    duration=500,
    loop=0
)

env.close()

print("GIF saved to results/agent_demo.gif")
print("Total reward:", total_reward)
print("Successful episode:", total_reward > 0)
