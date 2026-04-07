import copy
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from minecraft.mine_craft import MineCraft


@dataclass
class Transition:
    s: list | np.ndarray
    a: int
    r: float
    s_prime: list | np.ndarray
    done: bool
    next_dfa_state: int | None = None


def build_q_net(state_dim: int, n_actions: int, hidden: int = 128) -> keras.Model:
    return keras.Sequential([
        keras.layers.Dense(hidden, input_dim=state_dim, activation='relu'),
        keras.layers.Dense(hidden, activation='relu'),
        keras.layers.Dense(n_actions),
        # keras.layers.Dense(n_actions, activation='sigmoid'),
    ])


def build_dueling_q_net(state_dim: int, n_actions: int, hidden: int = 128) -> keras.Model:
    inputs = keras.Input(shape=(state_dim,))
    x = keras.layers.Dense(hidden, activation='relu')(inputs)
    x = keras.layers.Dense(hidden, activation='relu')(x)

    # Value stream
    v = keras.layers.Dense(hidden // 2, activation='relu')(x)
    v = keras.layers.Dense(1)(v)

    # Advantage stream
    a = keras.layers.Dense(hidden // 2, activation='relu')(x)
    a = keras.layers.Dense(n_actions)(a)

    # Combine: Q = V + A - mean(A)
    a_mean = keras.layers.Lambda(lambda x: keras.ops.mean(x, axis=-1, keepdims=True))(a)
    q = keras.layers.Add()([v, a])
    q = keras.layers.Subtract()([q, a_mean])
    # q = keras.layers.Activation('sigmoid')(q)

    return keras.Model(inputs=inputs, outputs=q)


class DQN:
    def __init__(
        self,
        gamma: float = 0.95,
        eps: float = 0.1,
        tau: float = 0.005,
        lr: float = 1e-3,
        double_dqn: bool = True,
        dueling_dqn: bool = True,
        state_dim: int = 0,
        action_dim: int = 4,
        hidden_dim: int = 128,
    ):
        self.gamma = gamma
        self.eps = eps
        self.tau = tau
        self.action_dim = action_dim

        builder = build_dueling_q_net if dueling_dqn else build_q_net
        self.q_net = builder(state_dim, action_dim, hidden_dim)

        self.double_dqn = double_dqn
        if double_dqn:
            self.target_net = builder(state_dim, action_dim, hidden_dim)
            # Initialize target weights to match q_net
            self.target_net.set_weights(self.q_net.get_weights())

        self.optimizer = keras.optimizers.Adam(learning_rate=lr)

    def e_greedy(self, s: np.ndarray, pure_greedy=False) -> int:
        eps = 0.0 if pure_greedy else self.eps
        if np.random.random() < eps:
            return np.random.randint(self.action_dim)
        q_vals = self.q_net(s[np.newaxis, :], training=False)
        return int(tf.argmax(q_vals, axis=1).numpy()[0])

    @tf.function
    def _train_step(self, s, a, r, s_prime, done, weights):
        with tf.GradientTape() as tape:
            # Q(s, a) for chosen actions
            q_all = self.q_net(s, training=True)                    # (B, A)
            indices = tf.stack([tf.range(tf.shape(a)[0]), a], axis=1)
            q_values = tf.gather_nd(q_all, indices)                 # (B,)

            # Target
            if self.double_dqn:
                best_actions = tf.argmax(self.q_net(s_prime, training=False), axis=1, output_type=tf.int32)
                target_q_all = self.target_net(s_prime, training=False)
                target_indices = tf.stack([tf.range(tf.shape(best_actions)[0]), best_actions], axis=1)
                next_q = tf.gather_nd(target_q_all, target_indices)
            else:
                next_q = tf.reduce_max(self.q_net(s_prime, training=False), axis=1)

            # mask bootstrap at terminal states
            targets = r + self.gamma * next_q * (1.0 - done)
            td_errors = q_values - targets
            loss = tf.reduce_mean(weights * tf.square(td_errors))

        gradients = tape.gradient(loss, self.q_net.trainable_variables)
        clipped = [tf.clip_by_norm(g, 10.0) for g in gradients]
        self.optimizer.apply_gradients(zip(clipped, self.q_net.trainable_variables))
        return loss, td_errors

    def update(self, batch: list[Transition], weights: np.ndarray | None = None) -> tuple[float, np.ndarray]:
        s = np.array([t.s for t in batch], dtype=np.float32)
        s_prime = np.array([t.s_prime for t in batch], dtype=np.float32)
        r = np.array([t.r for t in batch], dtype=np.float32)
        a = np.array([t.a for t in batch], dtype=np.int32)
        done = np.array([t.done for t in batch], dtype=np.float32)
        if weights is None:
            weights = np.ones(len(batch), dtype=np.float32)

        loss, td_errors = self._train_step(s, a, r, s_prime, done, weights)
        return float(loss.numpy()), np.abs(td_errors.numpy())

    def update_target(self):
        self.target_net.set_weights(
            [self.tau * w + (1.0 - self.tau) * tw
             for w, tw in zip(self.q_net.get_weights(), self.target_net.get_weights())]
        )

    def hard_update_target(self):
        self.target_net.set_weights(self.q_net.get_weights())

    def soft_update_target(self):
        for target_param, online_param in zip(self.target_net.variables, self.q_net.variables):
            target_param.assign(self.tau * online_param + (1 - self.tau) * target_param)

    @tf.function
    def _train_step_with_targets(self, s, a, targets, weights):
        """Train with externally computed targets (for cross-module bootstrapping)."""
        with tf.GradientTape() as tape:
            q_all = self.q_net(s, training=True)
            indices = tf.stack([tf.range(tf.shape(a)[0]), a], axis=1)
            q_values = tf.gather_nd(q_all, indices)
            td_errors = q_values - targets
            loss = tf.reduce_mean(weights * tf.square(td_errors))
        gradients = tape.gradient(loss, self.q_net.trainable_variables)
        clipped = [tf.clip_by_norm(g, 10.0) for g in gradients]
        self.optimizer.apply_gradients(zip(clipped, self.q_net.trainable_variables))
        return loss, td_errors

    def update_with_targets(self, batch: list[Transition], targets: np.ndarray,
                            weights: np.ndarray | None = None) -> tuple[float, np.ndarray]:
        s = np.array([t.s for t in batch], dtype=np.float32)
        a = np.array([t.a for t in batch], dtype=np.int32)
        targets = np.array(targets, dtype=np.float32)
        if weights is None:
            weights = np.ones(len(batch), dtype=np.float32)
        loss, td_errors = self._train_step_with_targets(s, a, targets, weights)
        return float(loss.numpy()), np.abs(td_errors.numpy())


class PrioritizedReplayBuffer:
    def __init__(self, size: int, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 1e-4, eps: float = 1e-6):
        self._storage = [None] * size
        self._priorities = np.zeros(size, dtype=np.float64)
        self._max_size = size
        self._next_idx = 0
        self._size = 0
        self._alpha = alpha       # priority exponent: 0 = uniform, 1 = full prioritization
        self._beta = beta         # IS correction exponent, annealed toward 1
        self._beta_increment = beta_increment
        self._eps = eps           # small constant to avoid zero priority

    def __len__(self):
        return self._size

    def add(self, data):
        # new transitions get max priority so they are sampled at least once
        max_p = self._priorities[:self._size].max() if self._size > 0 else 1.0
        self._storage[self._next_idx] = data
        self._priorities[self._next_idx] = max_p ** self._alpha
        self._next_idx = (self._next_idx + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)

    def sample(self, batch_size: int):
        self._beta = min(1.0, self._beta + self._beta_increment)

        priors = self._priorities[:self._size]
        probs = priors / priors.sum()

        indices = np.random.choice(self._size, size=batch_size, p=probs)
        batch = [self._storage[i] for i in indices]

        # importance-sampling weights
        weights = (self._size * probs[indices]) ** (-self._beta)
        weights /= weights.max()

        return batch, indices, weights.astype(np.float32)

    def update_priorities(self, indices, td_errors: np.ndarray):
        for idx, td in zip(indices, td_errors):
            self._priorities[idx] = (abs(td) + self._eps) ** self._alpha

    def getSaveState(self):
        return {
            'storage': list(self._storage[:self._size]),
            'priorities': self._priorities[:self._size].copy(),
            'max_size': self._max_size,
            'next_idx': self._next_idx,
            'size': self._size,
            'alpha': self._alpha,
            'beta': self._beta,
        }

    def loadFromState(self, save_state):
        self._max_size = save_state['max_size']
        self._storage = [None] * self._max_size
        self._priorities = np.zeros(self._max_size, dtype=np.float64)
        self._size = save_state['size']
        self._next_idx = save_state['next_idx']
        self._alpha = save_state['alpha']
        self._beta = save_state['beta']
        for i, d in enumerate(save_state['storage']):
            self._storage[i] = d
        self._priorities[:self._size] = save_state['priorities']