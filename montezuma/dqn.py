import copy
from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import (Add, Conv2D, Dense, Flatten, Input,
                                     Lambda, Subtract)
from tensorflow.keras.models import Model


@dataclass
class Transition:
    s: list | np.ndarray
    a: int
    r: float
    s_prime: list | np.ndarray
    done: bool
    dfa_state: int = 0


# ======================================================================
# Dense network builders (Minecraft / vector-state environments)
# ======================================================================

def build_q_net(state_dim: int, n_actions: int, hidden: int = 128) -> keras.Model:
    return keras.Sequential([
        keras.layers.Dense(hidden, input_dim=state_dim, activation='relu'),
        keras.layers.Dense(hidden, activation='relu'),
        keras.layers.Dense(n_actions),
    ])


def build_dueling_q_net(state_dim: int, n_actions: int, hidden: int = 128) -> keras.Model:
    inputs = keras.Input(shape=(state_dim,))
    x = keras.layers.Dense(hidden, activation='relu')(inputs)
    x = keras.layers.Dense(hidden, activation='relu')(x)

    v = keras.layers.Dense(hidden // 2, activation='relu')(x)
    v = keras.layers.Dense(1)(v)

    a = keras.layers.Dense(hidden // 2, activation='relu')(x)
    a = keras.layers.Dense(n_actions)(a)

    a_mean = Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(a)
    q = keras.layers.Add()([v, a])
    q = keras.layers.Subtract()([q, a_mean])

    return keras.Model(inputs=inputs, outputs=q)


# ======================================================================
# Convolutional network builders (Atari / image-state environments)
# Matches the architecture in the DeepSynth learner.py exactly.
# ======================================================================

def build_conv_q_net(
    n_actions: int,
    learning_rate: float = 0.00001,
    input_shape: tuple = (84, 84),
    history_length: int = 4,
) -> keras.Model:
    """Standard (non-dueling) conv DQN for Atari frames."""
    model_input = Input(shape=(input_shape[0], input_shape[1], history_length))
    x = Lambda(lambda layer: layer / 255)(model_input)
    x = Conv2D(32, (8, 8), strides=4, kernel_initializer=VarianceScaling(scale=2.),
               activation='relu', use_bias=False)(x)
    x = Conv2D(64, (4, 4), strides=2, kernel_initializer=VarianceScaling(scale=2.),
               activation='relu', use_bias=False)(x)
    x = Conv2D(64, (3, 3), strides=1, kernel_initializer=VarianceScaling(scale=2.),
               activation='relu', use_bias=False)(x)
    x = Flatten()(x)
    x = Dense(512, kernel_initializer=VarianceScaling(scale=2.), activation='relu')(x)
    q_vals = Dense(n_actions, kernel_initializer=VarianceScaling(scale=2.))(x)
    return Model(model_input, q_vals)


def build_conv_dueling_q_net(
    n_actions: int,
    learning_rate: float = 0.00001,
    input_shape: tuple = (84, 84),
    history_length: int = 4,
) -> keras.Model:
    """Dueling conv DQN matching the original learner.py architecture."""
    model_input = Input(shape=(input_shape[0], input_shape[1], history_length))
    x = Lambda(lambda layer: layer / 255)(model_input)
    x = Conv2D(32, (8, 8), strides=4, kernel_initializer=VarianceScaling(scale=2.),
               activation='relu', use_bias=False)(x)
    x = Conv2D(64, (4, 4), strides=2, kernel_initializer=VarianceScaling(scale=2.),
               activation='relu', use_bias=False)(x)
    x = Conv2D(64, (3, 3), strides=1, kernel_initializer=VarianceScaling(scale=2.),
               activation='relu', use_bias=False)(x)
    x = Conv2D(1024, (7, 7), strides=1, kernel_initializer=VarianceScaling(scale=2.),
               activation='relu', use_bias=False)(x)

    val_stream, adv_stream = Lambda(lambda w: tf.split(w, 2, 3))(x)

    val_stream = Flatten()(val_stream)
    val = Dense(1, kernel_initializer=VarianceScaling(scale=2.))(val_stream)

    adv_stream = Flatten()(adv_stream)
    adv = Dense(n_actions, kernel_initializer=VarianceScaling(scale=2.))(adv_stream)

    reduce_mean = Lambda(lambda w: tf.reduce_mean(w, axis=1, keepdims=True))
    q_vals = Add()([val, Subtract()([adv, reduce_mean(adv)])])

    return Model(model_input, q_vals)


# ======================================================================
# DQN Agent
# ======================================================================

class DQN:
    """
    Dueling Double DQN module compatible with the DeepSynth modular
    architecture.  Supports both dense (vector) and conv (image) networks
    via the ``network_builder`` parameter.

    Cross-module bootstrapping is handled by passing an ``agents_dict``
    (mapping DFA state ints → DQN instances) into ``update()`` or
    ``update_from_arrays()``.
    """

    def __init__(
        self,
        gamma: float = 0.99,
        eps: float = 1.0,
        lr: float = 2.5e-4,
        double_dqn: bool = True,
        # --- dense builder params (ignored when network_builder given) ---
        dueling_dqn: bool = True,
        state_dim: int = 0,
        action_dim: int = 4,
        hidden_dim: int = 128,
        # --- custom builder (for conv networks) ---
        network_builder: Optional[Callable[..., keras.Model]] = None,
        builder_kwargs: Optional[dict] = None,
    ):
        self.gamma = gamma
        self.eps = eps
        self.action_dim = action_dim
        self.double_dqn = double_dqn

        # Build networks
        if network_builder is not None:
            kw = builder_kwargs or {}
            self.q_net = network_builder(**kw)
            if double_dqn:
                self.target_net = network_builder(**kw)
                self.target_net.set_weights(self.q_net.get_weights())
        else:
            builder = build_dueling_q_net if dueling_dqn else build_q_net
            self.q_net = builder(state_dim, action_dim, hidden_dim)
            if double_dqn:
                self.target_net = builder(state_dim, action_dim, hidden_dim)
                self.target_net.set_weights(self.q_net.get_weights())

        self.optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.huber = keras.losses.Huber()
        self.huber_unreduced = keras.losses.Huber(reduction='none')

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def e_greedy(self, s: np.ndarray, pure_greedy: bool = False) -> int:
        eps = 0.0 if pure_greedy else self.eps
        if np.random.random() < eps:
            return np.random.randint(self.action_dim)
        # Add batch dim regardless of input rank
        q_vals = self.q_net(s[np.newaxis, ...], training=False)
        return int(tf.argmax(q_vals, axis=1).numpy()[0])

    # ------------------------------------------------------------------
    # Transition-list interface (Minecraft / simple envs)
    # ------------------------------------------------------------------

    @tf.function
    def _train_step(self, s, a, r, s_prime, done):
        with tf.GradientTape() as tape:
            q_all = self.q_net(s, training=True)
            indices = tf.stack([tf.range(tf.shape(a)[0]), a], axis=1)
            q_values = tf.gather_nd(q_all, indices)

            if self.double_dqn:
                best_a = tf.argmax(self.q_net(s_prime, training=False),
                                   axis=1, output_type=tf.int32)
                tgt_q_all = self.target_net(s_prime, training=False)
                tgt_idx = tf.stack([tf.range(tf.shape(best_a)[0]), best_a], axis=1)
                next_q = tf.gather_nd(tgt_q_all, tgt_idx)
            else:
                next_q = tf.reduce_max(self.q_net(s_prime, training=False), axis=1)

            targets = r + self.gamma * next_q * (1.0 - done)
            loss = self.huber(targets, q_values)

        grads = tape.gradient(loss, self.q_net.trainable_variables)
        clipped = [tf.clip_by_norm(g, 10.0) for g in grads]
        self.optimizer.apply_gradients(zip(clipped, self.q_net.trainable_variables))
        return loss

    def update(
        self,
        batch: list[Transition],
        my_dfa_state: int = 0,
        agents_dict: Optional[dict] = None,
    ) -> float:
        s = np.array([t.s for t in batch], dtype=np.float32)
        s_prime = np.array([t.s_prime for t in batch], dtype=np.float32)
        r = np.array([t.r for t in batch], dtype=np.float32)
        a = np.array([t.a for t in batch], dtype=np.int32)
        done = np.array([t.done for t in batch], dtype=np.float32)

        if agents_dict is not None:
            dfa_states = np.array([t.dfa_state for t in batch], dtype=np.int32)
            loss, _ = self.update_from_arrays(
                s, a, r, s_prime, done,
                dfa_states=dfa_states,
                my_dfa_state=my_dfa_state,
                agents_dict=agents_dict,
            )
        else:
            loss = float(self._train_step(s, a, r, s_prime, done).numpy())
        return loss

    # ------------------------------------------------------------------
    # Array interface (Atari / original replay buffer)
    # ------------------------------------------------------------------

    def update_from_arrays(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        new_states: np.ndarray,
        terminal_flags: np.ndarray,
        dfa_states: Optional[np.ndarray] = None,
        my_dfa_state: int = 0,
        agents_dict: Optional[dict] = None,
        importance_weights: Optional[np.ndarray] = None,
    ) -> tuple[float, np.ndarray]:
        """
        Train on pre-batched numpy arrays with optional cross-module
        bootstrapping and PER importance weighting.

        Returns:
            (loss, per_sample_td_errors)
        """
        batch_size = states.shape[0]

        # ---------- compute next_q with cross-module routing ----------
        next_q_np = np.zeros(batch_size, dtype=np.float32)

        if agents_dict is not None and dfa_states is not None:
            module_groups: dict[int, list[int]] = {}
            for i in range(batch_size):
                mod = int(dfa_states[i])
                module_groups.setdefault(mod, []).append(i)

            for mod_id, sample_indices in module_groups.items():
                idx = np.array(sample_indices)
                sp = new_states[idx]
                agent = agents_dict.get(mod_id, self)

                if self.double_dqn:
                    best_a = tf.argmax(agent.q_net(sp, training=False),
                                       axis=1, output_type=tf.int32).numpy()
                    tgt_q = agent.target_net(sp, training=False).numpy()
                    next_q_np[idx] = tgt_q[np.arange(len(idx)), best_a]
                else:
                    next_q_np[idx] = agent.q_net(sp, training=False).numpy().max(axis=1)
        else:
            if self.double_dqn:
                best_a = tf.argmax(self.q_net(new_states, training=False),
                                   axis=1, output_type=tf.int32).numpy()
                tgt_q = self.target_net(new_states, training=False).numpy()
                next_q_np = tgt_q[np.arange(batch_size), best_a]
            else:
                next_q_np = self.q_net(new_states, training=False).numpy().max(axis=1)

        target_q = rewards + self.gamma * next_q_np * (1.0 - terminal_flags)

        # ---------- gradient step ----------
        states_tf = tf.constant(states, dtype=tf.float32)
        actions_tf = tf.constant(actions, dtype=tf.int32)
        target_q_tf = tf.constant(target_q, dtype=tf.float32)

        with tf.GradientTape() as tape:
            q_all = self.q_net(states_tf, training=True)
            indices = tf.stack([tf.range(batch_size, dtype=tf.int32), actions_tf], axis=1)
            q_values = tf.gather_nd(q_all, indices)

            error = q_values - target_q_tf
            per_sample_loss = self.huber_unreduced(target_q_tf, q_values)

            if importance_weights is not None:
                iw = tf.constant(importance_weights, dtype=tf.float32)
                loss = tf.reduce_mean(per_sample_loss * iw)
            else:
                loss = tf.reduce_mean(per_sample_loss)

        grads = tape.gradient(loss, self.q_net.trainable_variables)
        clipped = [tf.clip_by_norm(g, 10.0) for g in grads]
        self.optimizer.apply_gradients(zip(clipped, self.q_net.trainable_variables))

        return float(loss.numpy()), error.numpy()

    # ------------------------------------------------------------------
    # Target network updates
    # ------------------------------------------------------------------

    def hard_update_target(self):
        if self.double_dqn:
            self.target_net.set_weights(self.q_net.get_weights())

    def soft_update_target(self, tau: float = 0.005):
        if self.double_dqn:
            self.target_net.set_weights([
                tau * w + (1.0 - tau) * tw
                for w, tw in zip(self.q_net.get_weights(),
                                 self.target_net.get_weights())
            ])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, folder_name: str):
        import os
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)
        self.q_net.save(folder_name + '/dqn.h5')
        if self.double_dqn:
            self.target_net.save(folder_name + '/target_dqn.h5')


# ======================================================================
# Replay Buffer (simple, for Minecraft / vector-state)
# ======================================================================

class QLearningBuffer:
    def __init__(self, size: int):
        self._storage: list[Transition] = []
        self._max_size = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def __getitem__(self, key):
        return self._storage[key]

    @property
    def count(self):
        return len(self._storage)

    def add(self, data: Transition):
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._max_size

    def sample(self, batch_size: int) -> list[Transition]:
        idxs = np.random.choice(len(self._storage), batch_size).tolist()
        return [self._storage[i] for i in idxs]

    def sample_for_module(self, batch_size: int, dfa_state: int) -> list[Transition]:
        eligible = [t for t in self._storage if t.dfa_state == dfa_state]
        if len(eligible) < batch_size:
            return eligible
        idxs = np.random.choice(len(eligible), batch_size, replace=False).tolist()
        return [eligible[i] for i in idxs]

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