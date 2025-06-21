import sys

sys.path.append("../src/")
from config import *
import tensorflow.keras.losses as kloss
import tensorflow.keras.optimizers as opt
import tensorflow as tf
import numpy as np
import time
from tqdm import tqdm
import wandb
import os


class Agent:
    def __init__(self, model, save_path=PATH_SAVE_MODEL, load_path=PATH_LOAD_MODEL, lr=LR, gamma=GAMMA, value_c=VALUE_C,
                 entropy_c=ENTROPY_C, clip_ratio=CLIP_RATIO, std_adv=STD_ADV, agent=AGENT, input_shape=INPUT_SHAPE,
                 batch_size=BATCH_SIZE, updates=N_UPDATES):
        # `gamma` is the discount factor
        self.gamma = gamma
        # Coefficients are used for the loss terms.
        self.value_c = value_c
        self.entropy_c = entropy_c
        # `gamma` is the discount factor
        self.gamma = gamma
        self.save_path = save_path
        self.load_path = load_path
        self.clip_ratio = clip_ratio
        self.std_adv = std_adv
        self.agent = agent
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.updates = updates
        self.opt = opt.RMSprop(learning_rate=lr)

        self.model = model

        if load_path is not None:
            print("loading model in {}".format(load_path))
            self.load_model(load_path)
            print("model loaded")

    def train(self, wrapper):
        # Storage helpers for a single batch of data.
        actions = np.empty((self.batch_size), dtype=np.int32)
        rewards, dones, values = np.empty((3, self.batch_size))
        observations = np.empty((self.batch_size,) + self.input_shape)
        old_logits = np.empty((self.batch_size, wrapper.env.action_space.n), dtype=np.float32)

        # Training loop: collect samples, send to optimizer, repeat updates times.
        ep_rewards = [0.0]  # episode reward history
        reward_means = []  # for plotting
        max_reward_so_far = -21.0  # Pong 최대 점수는 +21

        next_obs = wrapper.reset()

        for update in tqdm(range(self.updates)):
            start_time = time.time()

            for step in range(self.batch_size):
                observations[step] = next_obs.copy()
                old_logits[step], actions[step], values[step] = self.logits_action_value(next_obs[None, :])
                next_obs, rewards[step], dones[step] = wrapper.step(actions[step])
                next_obs = wrapper.state
                ep_rewards[-1] += rewards[step]

                if dones[step]:
                    # 에피소드 완료 시 최고 기록 업데이트
                    if ep_rewards[-1] > max_reward_so_far:
                        max_reward_so_far = ep_rewards[-1]

                    ep_rewards.append(0.0)
                    next_obs = wrapper.reset()

                    # 에피소드 완료 시에만 로그
                    wandb.log({
                        'episode_reward': round(ep_rewards[-2], 2),
                        'episode_count': len(ep_rewards) - 1,
                        'max_reward': round(max_reward_so_far, 2),
                        'update_step': update,
                    })

            # 마지막 상태 기준 value
            _, _, next_value = self.logits_action_value(next_obs[None, :])

            returns, advs = self._returns_advantages(rewards, dones, values, next_value, self.std_adv)

            # Train step
            with tf.GradientTape() as tape:
                # 입력 데이터 타입 통일
                observations_input = observations.astype(np.uint8)
                logits, v = self.model(observations_input, training=True)

                if self.agent == "A2C":
                    logit_loss = self._logits_loss_a2c(actions, advs, logits)
                elif self.agent == "PPO":
                    logit_loss, ratio = self._logits_loss_ppo(old_logits, logits, actions, advs,
                                                              wrapper.env.action_space.n)
                else:
                    raise Exception("Sorry agent can be just A2C or PPO")

                value_loss = self._value_loss(returns, v)
                loss = logit_loss + value_loss

            grads = tape.gradient(loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

            # 중요 지표들 계산
            entropy = -tf.reduce_sum(tf.nn.softmax(logits) * tf.nn.log_softmax(logits), axis=1)

            # PPO 전용 지표들
            ppo_metrics = {}
            if self.agent == "PPO":
                clipped_fraction = tf.reduce_mean(tf.cast(
                    tf.logical_or(ratio < 1 - self.clip_ratio, ratio > 1 + self.clip_ratio),
                    tf.float32))

                # KL divergence 근사치
                old_probs = tf.nn.softmax(tf.cast(old_logits, tf.float32))
                new_probs = tf.nn.softmax(tf.cast(logits, tf.float32))
                kl_div = tf.reduce_mean(
                    tf.reduce_sum(old_probs * tf.math.log(old_probs / (new_probs + 1e-8) + 1e-8), axis=1))

                ppo_metrics = {
                    "ppo_ratio_mean": float(tf.reduce_mean(ratio)),
                    "clipped_fraction": float(clipped_fraction),
                    "kl_divergence": float(kl_div),
                }

            # Explained variance (value function 성능 지표)
            returns_tensor = tf.cast(returns, tf.float32)
            v_squeezed = tf.cast(tf.squeeze(v), tf.float32)
            explained_var = 1 - tf.reduce_mean((returns_tensor - v_squeezed) ** 2) / (
                        tf.math.reduce_variance(returns_tensor) + 1e-8)

            # 학습 진행 상황 로그 (매 업데이트마다)
            log_dict = {
                "update": update,
                "total_loss": float(tf.reduce_mean(loss)),
                "policy_loss": float(tf.reduce_mean(logit_loss)),
                "value_loss": float(tf.reduce_mean(value_loss)),
                "entropy": float(tf.reduce_mean(entropy)),
                "explained_variance": float(explained_var),
                "mean_reward": float(np.mean(rewards)),
                "recent_episode_count": len(ep_rewards) - 1,
            }

            # PPO 지표 추가
            log_dict.update(ppo_metrics)

            wandb.log(log_dict)

            # 모델 저장 주기
            if update % 5000 == 0 and self.save_path is not None:
                print("Saving model in {}".format(self.save_path))
                self.save_model(
                    f'{self.save_path}/save_agent_{time.strftime("%Y%m%d%H%M") + "_" + str(update).zfill(8)}/model.tf')
                print("model saved")

            # reward 저장용 (1 episode 끝날 때마다)
            if len(ep_rewards) > 1:
                reward_means.append(ep_rewards[-2])

        return reward_means

    def _returns_advantages(self, rewards, dones, values, next_value, standardize_adv):
        gamma = self.gamma
        lam = 0.95  # GAE의 lambda 값

        # 모든 입력을 float32로 변환
        rewards = rewards.astype(np.float32)
        dones = dones.astype(np.float32)
        values = values.astype(np.float32)
        next_value = float(next_value)

        # value 배열에 마지막 상태 value 추가
        values = np.append(values, next_value)
        gae = 0.0
        returns = np.zeros_like(rewards, dtype=np.float32)

        # GAE 역방향 계산
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            returns[t] = gae + values[t]

        advantages = returns - values[:-1]

        if standardize_adv:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)

        return returns.astype(np.float32), advantages.astype(np.float32)

    def _new_returns_advantages(self, rewards, dones, values, next_value):
        """
        Broken for now
        """
        next_values = np.append(values, next_value, axis=-1)
        g = 0
        lmbda = 0.95
        returns = np.zeros_like(rewards)
        adv = np.zeros_like(rewards)
        for t in reversed(range(rewards.shape[0])):
            delta = rewards[t] + self.gamma * next_values[t + 1] * (1 - dones[t]) - values[t]
            g = delta + self.gamma * lmbda * g
            adv[t] = g
            returns[t] = adv[t] + values[t]
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
        return returns, adv

    def test(self, wrapper, render=False, record_path=None):
        obs, done, ep_reward = wrapper.reset(), False, 0
        frames = []

        while not done:
            _, action, _ = self.logits_action_value(obs[None, :])
            render_mode = 'rgb_array' if record_path else 'human' if render else None
            result = wrapper.step(action, render_mode=render_mode)

            if record_path:
                obs, reward, done, frame = result
                frames.append(frame)
            else:
                obs, reward, done = result

            obs = wrapper.state
            ep_reward += reward

        if record_path:
            from utilities import save_video
            save_video(frames, record_path)

        return ep_reward

    def _value_loss(self, returns, value):
        # Value loss is typically MSE between value estimates and returns.
        returns = tf.cast(returns, tf.float32)
        value = tf.cast(value, tf.float32)
        return self.value_c * kloss.mean_squared_error(returns, value)

    def _logits_loss_a2c(self, actions, advantages, logits):

        # Sparse categorical CE loss obj that supports sample_weight arg on `call()`.
        # `from_logits` argument ensures transformation into normalized probabilities.
        weighted_sparse_ce = kloss.SparseCategoricalCrossentropy(from_logits=True)

        # Policy loss is defined by policy gradients, weighted by advantages.
        # Note: we only calculate the loss on the actions we've actually taken.
        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)

        # Entropy loss can be calculated as cross-entropy over itself.
        probs = tf.nn.softmax(logits)
        entropy_loss = kloss.categorical_crossentropy(probs, probs)

        # We want to minimize policy and maximize entropy losses.
        # Here signs are flipped because the optimizer minimizes.
        return policy_loss - self.entropy_c * entropy_loss

    def _logits_loss_ppo(self, old_logits, logits, actions, advs, n_actions):
        # 모든 텐서를 float32로 통일
        old_logits = tf.cast(old_logits, tf.float32)
        logits = tf.cast(logits, tf.float32)
        actions = tf.cast(actions, tf.int32)
        advs = tf.cast(advs, tf.float32)
        advs = tf.stop_gradient(advs)

        actions_oh = tf.one_hot(actions, n_actions, dtype=tf.float32)
        actions_oh = tf.reshape(actions_oh, [-1, n_actions])
        actions_oh = tf.stop_gradient(actions_oh)

        new_policy = tf.nn.log_softmax(logits)
        old_policy = tf.nn.log_softmax(old_logits)
        old_policy = tf.stop_gradient(old_policy)

        old_log_p = tf.reduce_sum(old_policy * actions_oh, axis=1)
        log_p = tf.reduce_sum(new_policy * actions_oh, axis=1)
        ratio = tf.exp(log_p - old_log_p)

        clipped_ratio = tf.clip_by_value(
            ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)

        surrogate = tf.minimum(ratio * advs, clipped_ratio * advs)

        # Entropy loss 계산
        probs = tf.nn.softmax(logits)
        entropy_loss = -tf.reduce_sum(probs * tf.nn.log_softmax(logits), axis=1)

        # ratio도 반환해서 로깅에 사용
        return -tf.reduce_mean(surrogate) + self.entropy_c * tf.reduce_mean(entropy_loss), ratio

    def save_model(self, folder_path):
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)
        self.model.save(folder_path, save_format="tf")

    def load_model(self, folder_path):
        self.model = tf.keras.models.load_model(folder_path)

    def logits_action_value(self, obs):
        # Executes `call()` under the hood.
        logits, value = self.model(obs)
        action = self.model.dist(logits)
        return logits, np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)