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
# ReplayBuffer
# ======================================================================

class ReplayBuffer:
    def __init__(self, size=1000000, input_shape=(84, 84),
                 history_length=4, use_per=True):
        self.size = size
        self.input_shape = input_shape
        self.history_length = history_length
        self.count = 0
        self.current = 0

        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.dfa_states = np.empty(self.size, dtype=np.int_)
        self.frames = np.empty(
            (self.size, self.input_shape[0], self.input_shape[1]),
            dtype=np.uint8,
        )
        self.terminal_flags = np.empty(self.size, dtype=np.bool_)
        self.priorities = np.zeros(self.size, dtype=np.float32)
        self.use_per = use_per

    def add_experience(self, action, frame, reward, terminal,
                       dfa_state, clip_reward=True):
        if frame.shape != self.input_shape:
            raise ValueError('Dimension of frame is wrong!')
        if clip_reward:
            reward = np.sign(reward)
        self.actions[self.current] = action
        self.frames[self.current, ...] = frame
        self.rewards[self.current] = reward
        self.dfa_states[self.current] = dfa_state
        self.terminal_flags[self.current] = terminal
        self.priorities[self.current] = max(self.priorities.max(), 1)
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.size

    def get_minibatch(self, batch_size=32, priority_scale=0.0):
        if self.count < self.history_length:
            raise ValueError('Not enough memories to get a minibatch')

        if self.use_per:
            scaled_priorities = (
                self.priorities[self.history_length:self.count - 1] ** priority_scale
            )
            sample_probabilities = scaled_priorities / sum(scaled_priorities)

        indices = []
        for _ in range(batch_size):
            while True:
                if self.use_per:
                    index = np.random.choice(
                        np.arange(self.history_length, self.count - 1),
                        p=sample_probabilities,
                    )
                else:
                    index = random.randint(self.history_length, self.count - 1)
                if (index >= self.current
                        and index - self.history_length <= self.current):
                    continue
                if self.terminal_flags[index - self.history_length:index].any():
                    continue
                break
            indices.append(index)

        states, new_states, dfa_states_attached = [], [], []
        for idx in indices:
            states.append(self.frames[idx - self.history_length:idx, ...])
            new_states.append(self.frames[idx - self.history_length + 1:idx + 1, ...])
            dfa_states_attached.append(
                self.dfa_states[idx - self.history_length + 1:idx + 1, ...]
            )

        states = np.transpose(np.asarray(states), axes=(0, 2, 3, 1))
        new_states = np.transpose(np.asarray(new_states), axes=(0, 2, 3, 1))

        if self.use_per:
            importance = (
                1 / self.count
                * 1 / sample_probabilities[[index - 4 for index in indices]]
            )
            importance = importance / importance.max()
            return (
                (states, self.actions[indices], self.rewards[indices],
                 new_states, self.terminal_flags[indices]),
                importance, indices, dfa_states_attached,
            )
        else:
            return (
                states, self.actions[indices], self.rewards[indices],
                new_states, self.terminal_flags[indices],
                dfa_states_attached,
            )

    def set_priorities(self, indices, errors, offset=0.1):
        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e) + offset

    def save(self, folder_name):
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)
        np.save(folder_name + '/actions.npy', self.actions)
        if SAVE_FRAMES:
            np.save(folder_name + '/frames.npy', self.frames)
        np.save(folder_name + '/dfa_states.npy', self.dfa_states)
        np.save(folder_name + '/rewards.npy', self.rewards)
        np.save(folder_name + '/terminal_flags.npy', self.terminal_flags)

    def load(self, folder_name):
        self.actions = np.load(folder_name + '/actions.npy')
        self.frames = np.load(folder_name + '/frames.npy')
        self.rewards = np.load(folder_name + '/rewards.npy')
        self.dfa_states = np.load(folder_name + '/dfa_states.npy')
        self.terminal_flags = np.load(folder_name + '/terminal_flags.npy')