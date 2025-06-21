import wandb
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input
from tensorflow.keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
from agent_memory import Memory
import numpy as np
import random


class Agent():
    def __init__(self, possible_actions, starting_mem_len, max_mem_len, starting_epsilon, learn_rate, starting_lives=5,
                 debug=False):
        self.memory = Memory(max_mem_len)
        self.possible_actions = possible_actions
        self.epsilon = starting_epsilon
        self.epsilon_decay = .9 / 100000
        self.epsilon_min = .05
        self.gamma = .95
        self.learn_rate = learn_rate
        self.model = self._build_model()
        self.model_target = clone_model(self.model)
        self.total_timesteps = 0
        self.lives = starting_lives
        self.starting_mem_len = starting_mem_len
        self.learns = 0

        # 분석을 위한 추가 변수들
        self.recent_losses = []
        self.target_updates = 0

    def _build_model(self):
        model = Sequential()
        model.add(Input((84, 84, 4)))
        model.add(Conv2D(filters=32
                         , kernel_size=(8, 8)
                         , strides=4
                         , data_format="channels_last"
                         , activation='relu'
                         , kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))
        model.add(Conv2D(filters=64
                         , kernel_size=(4, 4)
                         , strides=2
                         , data_format="channels_last"
                         , activation='relu'
                         , kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))
        model.add(Conv2D(filters=64
                         , kernel_size=(3, 3)
                         , strides=1
                         , data_format="channels_last"
                         , activation='relu'
                         , kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))
        model.add(Flatten())
        model.add(Dense(512, activation='relu', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))
        model.add(Dense(len(self.possible_actions), activation='linear'))
        optimizer = Adam(self.learn_rate)
        model.compile(optimizer, loss=tf.keras.losses.Huber())
        model.summary()
        print('\nAgent Initialized\n')
        return model

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.sample(self.possible_actions, 1)[0]

        a_index = np.argmax(self.model.predict(state, verbose=0))
        return self.possible_actions[a_index]

    def _index_valid(self, index):
        if self.memory.done_flags[index - 3] or self.memory.done_flags[index - 2] or self.memory.done_flags[
            index - 1] or self.memory.done_flags[index]:
            return False
        else:
            return True

    def learn(self, debug=False):
        """Enhanced learning function with loss tracking"""

        """First we need 32 random valid indices"""
        states = []
        next_states = []
        actions_taken = []
        next_rewards = []
        next_done_flags = []

        while len(states) < 32:
            index = np.random.randint(4, len(self.memory.frames) - 1)
            if self._index_valid(index):
                state = [self.memory.frames[index - 3], self.memory.frames[index - 2], self.memory.frames[index - 1],
                         self.memory.frames[index]]
                state = np.moveaxis(state, 0, 2) / 255
                next_state = [self.memory.frames[index - 2], self.memory.frames[index - 1], self.memory.frames[index],
                              self.memory.frames[index + 1]]
                next_state = np.moveaxis(next_state, 0, 2) / 255

                states.append(state)
                next_states.append(next_state)
                actions_taken.append(self.memory.actions[index])
                next_rewards.append(self.memory.rewards[index + 1])
                next_done_flags.append(self.memory.done_flags[index + 1])

        """Get outputs from model and target model"""
        labels = self.model.predict(np.array(states), verbose=0)
        next_state_values = self.model_target.predict(np.array(next_states), verbose=0)

        """Calculate targets"""
        for i in range(32):
            action = self.possible_actions.index(actions_taken[i])
            labels[i][action] = next_rewards[i] + (not next_done_flags[i]) * self.gamma * max(next_state_values[i])

        """Train model and capture loss"""
        history = self.model.fit(np.array(states), labels, batch_size=32, epochs=1, verbose=0)
        loss_value = history.history['loss'][0]

        # 최근 손실값 저장 (최대 100개)
        self.recent_losses.append(loss_value)
        if len(self.recent_losses) > 100:
            self.recent_losses = self.recent_losses[-100:]

        """Update epsilon and learning count"""
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        self.learns += 1

        """Log learning metrics every 100 steps"""
        if self.learns % 100 == 0:
            avg_loss = np.mean(self.recent_losses) if self.recent_losses else 0
            wandb.log({
                "learning_step": self.learns,
                "epsilon": self.epsilon,
                "avg_loss_100": avg_loss,
                "current_loss": loss_value,
                "loss_std": np.std(self.recent_losses) if len(self.recent_losses) > 1 else 0
            })

        """Update target network every 10000 steps"""
        if self.learns % 10000 == 0:
            self.model_target.set_weights(self.model.get_weights())
            self.target_updates += 1
            print(f'\nTarget model updated (#{self.target_updates})')
            wandb.log({
                "target_model_updated": self.learns,
                "target_updates_count": self.target_updates
            })

    def get_average_loss(self):
        """Get average loss from recent training"""
        return np.mean(self.recent_losses) if self.recent_losses else 0

    def get_q_value_stats(self, state):
        """Get Q-value statistics for analysis"""
        if state is not None:
            q_values = self.model.predict(state, verbose=0)[0]
            return {
                "q_mean": float(np.mean(q_values)),
                "q_std": float(np.std(q_values)),
                "q_max": float(np.max(q_values)),
                "q_min": float(np.min(q_values))
            }
        return {"q_mean": 0, "q_std": 0, "q_max": 0, "q_min": 0}