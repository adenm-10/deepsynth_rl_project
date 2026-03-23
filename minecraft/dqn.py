
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


def build_q_net(state_dim: int, n_actions: int, hidden: int = 128) -> keras.Model:
    return keras.Sequential([
        keras.layers.Dense(hidden, input_dim=state_dim, activation='relu'),
        keras.layers.Dense(hidden, activation='relu'),
        # keras.layers.Dense(n_actions),
        keras.layers.Dense(n_actions, activation='sigmoid'),
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
    q = keras.layers.Activation('sigmoid')(q)

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
    def _train_step(self, s, a, r, s_prime, done):
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

            targets = r + self.gamma * next_q
            loss = tf.reduce_mean(tf.square(q_values - targets))

        gradients = tape.gradient(loss, self.q_net.trainable_variables)
        clipped = [tf.clip_by_norm(g, 10.0) for g in gradients]
        self.optimizer.apply_gradients(zip(clipped, self.q_net.trainable_variables))
        return loss

    def update(self, batch: list[Transition]) -> float:
        s = np.array([t.s for t in batch], dtype=np.float32)
        s_prime = np.array([t.s_prime for t in batch], dtype=np.float32)
        r = np.array([t.r for t in batch], dtype=np.float32)
        a = np.array([t.a for t in batch], dtype=np.int32)
        done = np.array([t.done for t in batch], dtype=np.float32)

        loss = self._train_step(s, a, r, s_prime, done)
        return float(loss.numpy())

    def update_target(self):
        for w, tw in zip(self.q_net.get_weights(), self.target_net.get_weights()):
            tw[:] = self.tau * w + (1.0 - self.tau) * tw
        self.target_net.set_weights(
            [self.tau * w + (1.0 - self.tau) * tw
             for w, tw in zip(self.q_net.get_weights(), self.target_net.get_weights())]
        )


class QLearningBuffer:
    def __init__(self, size):
        self._storage = []
        self._max_size = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def __getitem__(self, key):
        return self._storage[key]

    def add(self, data):
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._max_size

    def sample(self, batch_size):
        batch_indexes = np.random.choice(len(self._storage), batch_size).tolist()
        return [self._storage[idx] for idx in batch_indexes]

    def getSaveState(self):
        return {
            'storage': self._storage,
            'max_size': self._max_size,
            'next_idx': self._next_idx,
        }

    def loadFromState(self, save_state):
        self._storage = save_state['storage']
        self._max_size = save_state['max_size']
        self._next_idx = save_state['next_idx']