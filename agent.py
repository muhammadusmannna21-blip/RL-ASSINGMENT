
import numpy as np


class QLearningAgent:
    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.8,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.9995,
        min_epsilon=0.01
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.q_table = np.zeros((state_size, action_size))

    def choose_action(self, state, explore=True):
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)

        return int(np.argmax(self.q_table[state]))

    def learn(self, state, action, reward, next_state, terminated):
        old_value = self.q_table[state, action]

        if terminated:
            future_value = 0
        else:
            future_value = np.max(self.q_table[next_state])

        new_value = reward + self.discount_factor * future_value

        self.q_table[state, action] = (
            old_value
            + self.learning_rate * (new_value - old_value)
        )

    def decay_epsilon(self):
        self.epsilon = max(
            self.min_epsilon,
            self.epsilon * self.epsilon_decay
        )
