
import os
import numpy as np
import gymnasium as gym

from agent import QLearningAgent
from utils import save_training_plot


SEED = 42
EPISODES = 20000

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

np.random.seed(SEED)

env = gym.make("FrozenLake-v1", is_slippery=False)
env.action_space.seed(SEED)

agent = QLearningAgent(
    state_size=env.observation_space.n,
    action_size=env.action_space.n,
    learning_rate=0.8,
    discount_factor=0.95,
    epsilon=1.0,
    epsilon_decay=0.9995,
    min_epsilon=0.01
)

episode_rewards = []

for episode in range(EPISODES):
    state, info = env.reset()
    total_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)

        agent.learn(
            state,
            action,
            reward,
            next_state,
            terminated or truncated
        )

        state = next_state
        total_reward += reward

    agent.decay_epsilon()
    episode_rewards.append(total_reward)

    if (episode + 1) % 1000 == 0:
        recent_success = np.mean(episode_rewards[-1000:])
        print(
            f"Episode {episode + 1}/{EPISODES} | "
            f"Success rate: {recent_success:.2%} | "
            f"Epsilon: {agent.epsilon:.3f}"
        )

np.save("models/q_table.npy", agent.q_table)
np.save("results/training_rewards.npy", np.array(episode_rewards))

save_training_plot(episode_rewards)

agent.epsilon = 0.0
evaluation_rewards = []

for episode in range(100):
    state, info = env.reset()
    total_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = agent.choose_action(state, explore=False)
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    evaluation_rewards.append(total_reward)

print("\nTraining completed.")
print(f"Evaluation success rate: {np.mean(evaluation_rewards):.2%}")
print("Q-table saved to models/q_table.npy")

env.close()
