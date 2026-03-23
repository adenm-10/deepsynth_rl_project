"""
DeepSynth learner for Montezuma's Revenge.

Mirrors the original learner.py structure but replaces the Agent class
and build_q_network with the custom DQN class from dqn.py.  Everything
outside the DQN implementation — GameWrapper, ReplayBuffer, image
processing, DFA integration, intrinsic reward, main loop — is kept as
close to the original as possible.
"""

from montezuma.config import (
    BATCH_SIZE, CLIP_REWARD, DISCOUNT_FACTOR, ENV_NAME,
    EVAL_LENGTH, FRAMES_BETWEEN_EVAL, INPUT_SHAPE,
    LEARNING_RATE, SAVE_FRAMES,
    MAX_NOOP_STEPS, MEM_SIZE, MIN_DFA_FRAMES,
    MIN_REPLAY_BUFFER_SIZE, PRIORITY_SCALE, SAVE_PATH,
    TARGET_UPDATE_FREQ, DFA_UPDATE_FREQ, TENSORBOARD_DIR, TOTAL_FRAMES,
    UPDATE_FREQ, USE_PER, WRITE_TENSORBOARD, MU,
)

import numpy as np
import cv2
import dill
import random
import os
import json
import time

# NumPy 2.0 removed aliases that old Gym depends on
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
if not hasattr(np, 'int0'):
    np.int0 = np.intp

import gym
import tensorflow as tf

# SYNTH imports
from synth.synth_wrapper import dfa_init
from synth.synth_wrapper import dfa_update
from synth.synth_wrapper import get_next_state

# Custom DQN
from dqn import DQN, build_conv_dueling_q_net


# ======================================================================
# Frame preprocessing  (identical to original)
# ======================================================================

def process_frame(frame, shape=(84, 84)):
    frame = frame.astype(np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = frame[34:34 + 160, :160]
    frame = cv2.resize(frame, shape, interpolation=cv2.INTER_NEAREST)
    frame = frame.reshape((*shape, 1))
    return frame


# ======================================================================
# GameWrapper  (identical to original)
# ======================================================================

class GameWrapper:
    def __init__(self, env_name, no_op_steps=10, history_length=4):
        self.env = gym.make(env_name)
        self.no_op_steps = no_op_steps
        self.history_length = 4
        self.state = None
        self.last_lives = 0

    def _unpack_reset(self, result):
        """Handle both old gym (obs) and new gym (obs, info) reset API."""
        if isinstance(result, tuple):
            return result[0]
        return result

    def _unpack_step(self, result):
        """Handle both old gym (obs,r,done,info) and new gym (obs,r,term,trunc,info)."""
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            return obs, reward, terminated or truncated, info
        return result  # (obs, reward, done, info)

    def _get_lives(self, info):
        """Handle both 'ale.lives' and 'lives' info keys."""
        if 'ale.lives' in info:
            return info['ale.lives']
        if 'lives' in info:
            return info['lives']
        return 0

    def reset(self, evaluation=False):
        self.frame = self._unpack_reset(self.env.reset())
        self.last_lives = 0
        if evaluation:
            for _ in range(random.randint(0, self.no_op_steps)):
                self._unpack_step(self.env.step(1))
        self.state = np.repeat(process_frame(self.frame), self.history_length, axis=2)

    def step(self, action, render_mode=None):
        new_frame, reward, terminal, info = self._unpack_step(self.env.step(action))
        raw_frame = new_frame.copy()

        lives = self._get_lives(info)
        if lives < self.last_lives:
            life_lost = True
        else:
            life_lost = terminal
        self.last_lives = lives

        processed_frame = process_frame(new_frame)
        self.state = np.append(self.state[:, :, 1:], processed_frame, axis=2)

        if render_mode == 'rgb_array':
            return processed_frame, reward, terminal, life_lost, self.env.render(render_mode)
        elif render_mode == 'human':
            self.env.render()
        return processed_frame, reward, terminal, life_lost, raw_frame


# ======================================================================
# ReplayBuffer  (identical to original — kept for memory-efficient
#                Atari frame storage)
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


# ======================================================================
# Epsilon schedule  (extracted from the original Agent class)
# ======================================================================

EPS_INITIAL = 1.0
EPS_FINAL = 0.2
EPS_FINAL_FRAME = 0.1
EPS_ANNEALING_FRAMES = 150000


def calc_epsilon(frame_number, evaluation=False):
    if evaluation:
        return 0.0
    if frame_number < MIN_REPLAY_BUFFER_SIZE:
        return EPS_INITIAL

    slope = -(EPS_INITIAL - EPS_FINAL) / EPS_ANNEALING_FRAMES
    intercept = EPS_INITIAL - slope * MIN_REPLAY_BUFFER_SIZE

    slope2 = -(EPS_FINAL - EPS_FINAL_FRAME) / (
        TOTAL_FRAMES - EPS_ANNEALING_FRAMES - MIN_REPLAY_BUFFER_SIZE
    )
    intercept2 = EPS_FINAL_FRAME - slope2 * TOTAL_FRAMES

    if frame_number < MIN_REPLAY_BUFFER_SIZE + EPS_ANNEALING_FRAMES:
        return slope * frame_number + intercept
    else:
        return slope2 * frame_number + intercept2


# ======================================================================
# Image-processing helpers  (identical to original)
# ======================================================================

def check(a, b, upper_left):
    ul_row, ul_col = upper_left
    b_rows, b_cols = b.shape
    a_slice = a[ul_row:ul_row + b_rows, :][:, ul_col:ul_col + b_cols]
    if a_slice.shape != b.shape:
        return False
    return (a_slice == b).all()


def subarray_detector(big_array, small_array):
    upper_left = np.argwhere(big_array == small_array[0, 0])
    for ul in upper_left:
        if check(big_array, small_array, ul):
            return True
    return False


def intrinsic_reward(new_obj_set_in, old_obj_set_in):
    if list(set(new_obj_set_in) - set(old_obj_set_in)):
        return 1
    return 0


# ======================================================================
# Helper: determine which DFA module should bootstrap a given sample
# (replicates the cross-module logic from the original Agent.learn)
# ======================================================================

def target_module_for_sample(dfa_window, current_agent_id):
    """
    Given a history window of DFA states and the current module's id,
    return which module's Q-network should provide the bootstrap target.
    """
    unique = np.unique(dfa_window)
    if len(unique) == 1:
        return int(unique[0])
    others = unique[unique != current_agent_id]
    return int(others[0]) if len(others) > 0 else current_agent_id


# ======================================================================
# DQN module factory
# ======================================================================

def make_dqn_module(n_actions):
    """Create a DQN instance with conv networks matching the paper."""
    return DQN(
        gamma=DISCOUNT_FACTOR,
        eps=EPS_INITIAL,
        lr=LEARNING_RATE,
        double_dqn=True,
        network_builder=build_conv_dueling_q_net,
        builder_kwargs=dict(
            n_actions=n_actions,
            learning_rate=LEARNING_RATE,
            input_shape=INPUT_SHAPE,
            history_length=4,
        ),
        action_dim=n_actions,
    )


# ======================================================================
# Save helper
# ======================================================================

def save_all(agent_dict, replay_buffer_dict, save_dir,
             frame_number, rewards, loss_list):
    for key in agent_dict:
        agent_path = f'{save_dir}/save-{str(frame_number).zfill(8)}/agent_{key}'
        agent_dict[key].save(agent_path)
        replay_buffer_dict[key].save(agent_path + '/replay-buffer')

    meta_path = f'{save_dir}/save-{str(frame_number).zfill(8)}'
    if not os.path.isdir(meta_path):
        os.makedirs(meta_path)
    with open(meta_path + '/meta.json', 'w+') as f:
        f.write(json.dumps({
            'frame_number': frame_number,
            'rewards': rewards[-200:],
            'loss_tail': loss_list[-200:],
            'dfa_states': list(agent_dict.keys()),
        }))


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    agent_unique = [
        [478, 478, 478], [478, 478, 478],
        [344, 344, 344], [478, 478, 478],
    ]

    game_wrapper = GameWrapper(ENV_NAME, MAX_NOOP_STEPS)
    n_actions = game_wrapper.env.action_space.n
    print(f"The environment has the following {n_actions} actions: "
          f"{game_wrapper.env.unwrapped.get_action_meanings()}")

    # TensorBoard
    writer = tf.summary.create_file_writer(TENSORBOARD_DIR)

    # --- DQN modules (one per DFA state) ---
    agent_dict = {1: make_dqn_module(n_actions)}

    # --- Replay buffers (one per DFA state, identical to original) ---
    replay_buffer_dict = {
        1: ReplayBuffer(size=MEM_SIZE, input_shape=INPUT_SHAPE, use_per=USE_PER),
    }

    # --- Per-module frame counters (for epsilon scheduling) ---
    frame_number_dict = {1: 0}

    # --- DFA state ---
    current_dfa_state = 1
    dfa_states = [0, 1]
    active_agent_id = 1
    frame_number = 0

    rewards = []
    set_of_episode_traces = []
    loss_list = []
    model_gen = []
    nfa_model = []
    dfa_model = []
    num_states, var, input_dict, hyperparams = dfa_init()
    print("initialise")
    synth_iter_num = 0

    # ======================= MAIN LOOP =======================
    try:
        with writer.as_default():
            while frame_number < TOTAL_FRAMES:
                epoch_frame = 0

                while epoch_frame < FRAMES_BETWEEN_EVAL:
                    start_time = time.time()
                    game_wrapper.reset()
                    life_lost = False
                    terminal = False
                    episode_trace = ['start']
                    episode_detected_objects = []
                    active_agent_id = 1
                    current_dfa_state = 1
                    episode_reward_sum = 0

                    while not terminal:
                        # --- Action selection ---
                        agent_dict[active_agent_id].eps = calc_epsilon(
                            frame_number_dict[active_agent_id]
                        )
                        action = agent_dict[active_agent_id].e_greedy(
                            game_wrapper.state
                        )

                        # --- Environment step ---
                        processed_frame, reward, terminal, life_lost, new_obs = \
                            game_wrapper.step(action)
                        frame_number += 1
                        if frame_number > MIN_REPLAY_BUFFER_SIZE + MIN_DFA_FRAMES:
                            frame_number_dict[active_agent_id] += 1
                            epoch_frame += 1

                        # --- Object detection (identical to original) ---
                        old_obj_set = episode_detected_objects.copy()
                        observation = np.sum(new_obs, axis=2)

                        if subarray_detector(
                            observation[93:134, 76:83], np.array(agent_unique)
                        ):
                            episode_detected_objects.append('middle_ladder')
                            episode_trace.append('middle_ladder')
                        elif subarray_detector(
                            observation[96:134, 110:115], np.array(agent_unique)
                        ):
                            episode_detected_objects.append('rope')
                            episode_trace.append('rope')
                        elif subarray_detector(
                            observation[136:179, 132:139], np.array(agent_unique)
                        ):
                            episode_detected_objects.append('right_ladder')
                            episode_trace.append('right_ladder')
                        elif subarray_detector(
                            observation[136:179, 20:27], np.array(agent_unique)
                        ):
                            episode_detected_objects.append('left_ladder')
                            episode_trace.append('left_ladder')
                        elif subarray_detector(
                            observation[99:106, 13:19], np.array(agent_unique)
                        ):
                            episode_detected_objects.append('key')
                            episode_trace.append('key')
                        elif subarray_detector(
                            observation[50:92, 20:24], np.array(agent_unique)
                        ):
                            episode_detected_objects.append('door')
                            episode_trace.append('door')
                        elif subarray_detector(
                            observation[50:92, 136:140], np.array(agent_unique)
                        ):
                            episode_detected_objects.append('door')
                            episode_trace.append('door')

                        new_obj_set = np.unique(episode_detected_objects).tolist()

                        # ============ SYNTH ============
                        old_dfa_states = dfa_states.copy()

                        if frame_number >= MIN_DFA_FRAMES:

                            # --- DFA update ---
                            if ((frame_number % DFA_UPDATE_FREQ == 0)
                                or (get_next_state(episode_trace,
                                        input_dict['event_uniq'],
                                        processed_dfa) == -1)
                                or (get_next_state(episode_trace,
                                        input_dict['event_uniq'],
                                        processed_dfa) == [])):
                                trace = []
                                set_of_episode_traces.append(episode_trace)
                                for x in set_of_episode_traces:
                                    trace = trace + x
                                trace = trace + ['start']
                                (num_states, processed_dfa, dfa_model,
                                 nfa_model, model_gen, var, input_dict) = \
                                    dfa_update(
                                        trace, num_states, dfa_model,
                                        nfa_model, model_gen, var,
                                        input_dict, hyperparams,
                                        start_time, synth_iter_num,
                                    )
                                dfa_states = list(set(
                                    [t[0] for t in processed_dfa]
                                    + [t[2] for t in processed_dfa]
                                ))
                                synth_iter_num += 1
                                set_of_episode_traces = [episode_trace]

                            # --- Spawn new DQN modules if DFA grew ---
                            new_dfa_states = list(
                                set(dfa_states) - set(old_dfa_states)
                            )
                            if new_dfa_states:
                                for i in new_dfa_states:
                                    agent_dict[i] = make_dqn_module(n_actions)
                                    replay_buffer_dict[i] = ReplayBuffer(
                                        size=MEM_SIZE,
                                        input_shape=INPUT_SHAPE,
                                        use_per=USE_PER,
                                    )
                                    frame_number_dict[i] = 0

                            # --- DFA transition ---
                            next_dfa_state = get_next_state(
                                episode_trace,
                                input_dict['event_uniq'],
                                processed_dfa,
                            )
                            reward = reward + MU * intrinsic_reward(
                                new_obj_set, old_obj_set
                            )
                            episode_detected_objects = new_obj_set

                            # Logging
                            avg_r = (round(np.mean(rewards[-101:-1]), 1)
                                     if len(rewards) > 102 else 0)
                            print(f"{frame_number}: r={reward} // avg_r={avg_r}"
                                  f" // objects:{new_obj_set}"
                                  f" // DFA state:{next_dfa_state}"
                                  f" // lives:{game_wrapper.last_lives}")
                            episode_reward_sum += reward

                            # --- Store experience ---
                            replay_buffer_dict[active_agent_id].add_experience(
                                action=action,
                                frame=processed_frame[:, :, 0],
                                reward=reward,
                                clip_reward=CLIP_REWARD,
                                dfa_state=next_dfa_state,
                                terminal=life_lost,
                            )

                            # --- Update all modules ---
                            for ag in list(agent_dict.keys()):
                                buf = replay_buffer_dict[ag]

                                if (frame_number % UPDATE_FREQ == 0
                                        and buf.count > MIN_REPLAY_BUFFER_SIZE):

                                    # Sample minibatch
                                    if USE_PER:
                                        ((states, actions, rews, new_states,
                                          term_flags),
                                         importance, indices,
                                         dfa_states_attached) = \
                                            buf.get_minibatch(
                                                batch_size=BATCH_SIZE,
                                                priority_scale=PRIORITY_SCALE,
                                            )
                                        eps_now = calc_epsilon(
                                            frame_number_dict[ag]
                                        )
                                        importance = importance ** (1 - eps_now)
                                    else:
                                        (states, actions, rews, new_states,
                                         term_flags,
                                         dfa_states_attached) = \
                                            buf.get_minibatch(
                                                batch_size=BATCH_SIZE,
                                                priority_scale=PRIORITY_SCALE,
                                            )
                                        importance = None

                                    # Per-sample target module
                                    target_dfa = np.array([
                                        target_module_for_sample(dw, ag)
                                        for dw in dfa_states_attached
                                    ], dtype=np.int32)

                                    loss, errors = \
                                        agent_dict[ag].update_from_arrays(
                                            states.astype(np.float32),
                                            actions,
                                            rews,
                                            new_states.astype(np.float32),
                                            term_flags.astype(np.float32),
                                            dfa_states=target_dfa,
                                            my_dfa_state=ag,
                                            agents_dict=agent_dict,
                                            importance_weights=importance,
                                        )
                                    loss_list.append(loss)

                                    if USE_PER:
                                        buf.set_priorities(indices, errors)

                                # --- Hard target update ---
                                if (frame_number % TARGET_UPDATE_FREQ == 0
                                    and frame_number_dict[ag]
                                        > MIN_REPLAY_BUFFER_SIZE):
                                    agent_dict[ag].hard_update_target()

                            # --- Switch active module ---
                            active_agent_id = next_dfa_state
                            current_dfa_state = next_dfa_state

                        else:
                            # Before MIN_DFA_FRAMES: just log
                            print(f"{frame_number}: r=0 // avg_r=0"
                                  f" // objects:{new_obj_set}"
                                  f" // DFA state:1"
                                  f" // lives:{game_wrapper.last_lives}")

                        # --- Life lost: reset episode DFA state ---
                        if life_lost:
                            episode_detected_objects = []
                            active_agent_id = 1
                            current_dfa_state = 1
                            set_of_episode_traces.append(episode_trace)
                            episode_trace = ['start']

                    rewards.append(episode_reward_sum)

                    if len(rewards) % 10 == 0:
                        if WRITE_TENSORBOARD:
                            tf.summary.scalar(
                                'Reward', np.mean(rewards[-10:]), frame_number
                            )
                            tf.summary.scalar(
                                'Loss', np.mean(loss_list[-100:]), frame_number
                            )
                            writer.flush()
                        print(
                            f'Game number: {str(len(rewards)).zfill(6)}  '
                            f'Frame number: {str(frame_number).zfill(8)}  '
                            f'Average reward: {np.mean(rewards[-10:]):0.1f}  '
                            f'Time taken: {(time.time() - start_time):.1f}s'
                        )

                # ============ EVALUATION ============
                terminal = True
                eval_rewards = []
                evaluate_frame_number = 0
                active_agent_id = 1
                current_dfa_state = 1
                episode_detected_objects = []
                episode_trace = ['start']

                for _ in range(EVAL_LENGTH):
                    if terminal:
                        game_wrapper.reset(evaluation=True)
                        life_lost = False
                        episode_reward_sum = 0
                        terminal = False

                    agent_dict[active_agent_id].eps = 0.0
                    action = agent_dict[active_agent_id].e_greedy(
                        game_wrapper.state
                    )

                    _, reward, terminal, life_lost, new_obs = \
                        game_wrapper.step(action)
                    evaluate_frame_number += 1

                    # Object detection (identical to original)
                    old_obj_set = episode_detected_objects.copy()
                    observation = np.sum(new_obs, axis=2)
                    if subarray_detector(
                        observation[93:134, 76:83], np.array(agent_unique)
                    ):
                        episode_detected_objects.append('middle_ladder')
                        episode_trace.append('middle_ladder')
                    elif subarray_detector(
                        observation[96:134, 110:115], np.array(agent_unique)
                    ):
                        episode_detected_objects.append('rope')
                        episode_trace.append('rope')
                    elif subarray_detector(
                        observation[136:179, 132:139], np.array(agent_unique)
                    ):
                        episode_detected_objects.append('right_ladder')
                        episode_trace.append('right_ladder')
                    elif subarray_detector(
                        observation[136:179, 20:27], np.array(agent_unique)
                    ):
                        episode_detected_objects.append('left_ladder')
                        episode_trace.append('left_ladder')
                    elif subarray_detector(
                        observation[99:106, 13:19], np.array(agent_unique)
                    ):
                        episode_detected_objects.append('key')
                        episode_trace.append('key')
                    elif subarray_detector(
                        observation[50:92, 20:24], np.array(agent_unique)
                    ):
                        episode_detected_objects.append('door')
                        episode_trace.append('door')
                    elif subarray_detector(
                        observation[50:92, 136:140], np.array(agent_unique)
                    ):
                        episode_detected_objects.append('door')
                        episode_trace.append('door')

                    new_obj_set = np.unique(episode_detected_objects).tolist()

                    if frame_number > MIN_REPLAY_BUFFER_SIZE:
                        next_dfa_state = get_next_state(
                            episode_trace,
                            input_dict['event_uniq'],
                            processed_dfa,
                        )
                    else:
                        next_dfa_state = current_dfa_state

                    reward = reward + MU * intrinsic_reward(
                        new_obj_set, old_obj_set
                    )
                    episode_detected_objects = new_obj_set
                    episode_reward_sum += reward

                    if next_dfa_state and next_dfa_state in agent_dict:
                        active_agent_id = next_dfa_state

                    if terminal:
                        eval_rewards.append(episode_reward_sum)

                if len(eval_rewards) > 0:
                    final_score = np.mean(eval_rewards)
                else:
                    final_score = episode_reward_sum

                print('Evaluation score:', final_score)
                if WRITE_TENSORBOARD:
                    tf.summary.scalar(
                        'Evaluation score', final_score, frame_number
                    )
                    writer.flush()

                # Save
                if len(rewards) > 300 and SAVE_PATH is not None:
                    save_all(agent_dict, replay_buffer_dict, SAVE_PATH,
                             frame_number, rewards, loss_list)

    except KeyboardInterrupt:
        print('\nTraining exited early.')
        writer.close()

        if SAVE_PATH is None:
            try:
                SAVE_PATH = input(
                    'Would you like to save the trained model? If so, '
                    'type in a save path, otherwise, interrupt with ctrl+c. '
                )
            except KeyboardInterrupt:
                print('\nExiting...')

        if SAVE_PATH is not None:
            print('Saving...')
            save_all(agent_dict, replay_buffer_dict, SAVE_PATH,
                     frame_number, rewards, loss_list)
            print('Saved.')