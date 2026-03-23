"""
DeepSynth learner for Montezuma's Revenge.

Uses the custom DQN class from dqn.py and delegates all automaton
synthesis to DFAManager from dfa_manager.py.
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

from dqn import DQN, ReplayBuffer, build_conv_dueling_q_net
from dfa_manager import DFAManager


# ======================================================================
# Frame preprocessing
# ======================================================================

def process_frame(frame, shape=(84, 84)):
    frame = frame.astype(np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = frame[34:34 + 160, :160]
    frame = cv2.resize(frame, shape, interpolation=cv2.INTER_NEAREST)
    frame = frame.reshape((*shape, 1))
    return frame


# ======================================================================
# GameWrapper
# ======================================================================

class GameWrapper:
    def __init__(self, env_name, no_op_steps=10, history_length=4):
        self.env = gym.make(env_name)
        self.no_op_steps = no_op_steps
        self.history_length = 4
        self.state = None
        self.last_lives = 0

    def _unpack_reset(self, result):
        if isinstance(result, tuple):
            return result[0]
        return result

    def _unpack_step(self, result):
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            return obs, reward, terminated or truncated, info
        return result

    def _get_lives(self, info):
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
# Epsilon schedule
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
# Image-processing helpers
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


def detect_objects(observation, agent_unique, episode_detected_objects, episode_trace):
    """
    Run the pixel-coordinate object detector on a raw frame.
    Mutates episode_detected_objects and episode_trace in place.
    """
    pat = np.array(agent_unique)
    if subarray_detector(observation[93:134, 76:83], pat):
        episode_detected_objects.append('middle_ladder')
        episode_trace.append('middle_ladder')
    elif subarray_detector(observation[96:134, 110:115], pat):
        episode_detected_objects.append('rope')
        episode_trace.append('rope')
    elif subarray_detector(observation[136:179, 132:139], pat):
        episode_detected_objects.append('right_ladder')
        episode_trace.append('right_ladder')
    elif subarray_detector(observation[136:179, 20:27], pat):
        episode_detected_objects.append('left_ladder')
        episode_trace.append('left_ladder')
    elif subarray_detector(observation[99:106, 13:19], pat):
        episode_detected_objects.append('key')
        episode_trace.append('key')
    elif subarray_detector(observation[50:92, 20:24], pat):
        episode_detected_objects.append('door')
        episode_trace.append('door')
    elif subarray_detector(observation[50:92, 136:140], pat):
        episode_detected_objects.append('door')
        episode_trace.append('door')


# ======================================================================
# Cross-module bootstrap helper
# ======================================================================

def target_module_for_sample(dfa_window, current_agent_id):
    unique = np.unique(dfa_window)
    if len(unique) == 1:
        return int(unique[0])
    others = unique[unique != current_agent_id]
    return int(others[0]) if len(others) > 0 else current_agent_id


# ======================================================================
# DQN module factory
# ======================================================================

def make_dqn_module(n_actions):
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
# Update all DQN modules
# ======================================================================

def update_all_modules(agent_dict, replay_buffer_dict, frame_number_dict,
                       frame_number, loss_list):
    """Run one training tick for every DQN module whose buffer is ready."""
    for ag in list(agent_dict.keys()):
        buf = replay_buffer_dict[ag]

        if (frame_number % UPDATE_FREQ == 0
                and buf.count > MIN_REPLAY_BUFFER_SIZE):

            if USE_PER:
                ((states, actions, rews, new_states, term_flags),
                 importance, indices, dfa_states_attached) = \
                    buf.get_minibatch(
                        batch_size=BATCH_SIZE,
                        priority_scale=PRIORITY_SCALE,
                    )
                eps_now = calc_epsilon(frame_number_dict[ag])
                importance = importance ** (1 - eps_now)
            else:
                (states, actions, rews, new_states,
                 term_flags, dfa_states_attached) = \
                    buf.get_minibatch(
                        batch_size=BATCH_SIZE,
                        priority_scale=PRIORITY_SCALE,
                    )
                importance = None

            target_dfa = np.array([
                target_module_for_sample(dw, ag)
                for dw in dfa_states_attached
            ], dtype=np.int32)

            loss, errors = agent_dict[ag].update_from_arrays(
                states.astype(np.float32),
                actions, rews,
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

        if (frame_number % TARGET_UPDATE_FREQ == 0
                and frame_number_dict[ag] > MIN_REPLAY_BUFFER_SIZE):
            agent_dict[ag].hard_update_target()


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

    writer = tf.summary.create_file_writer(TENSORBOARD_DIR)

    # --- DQN modules ---
    agent_dict: dict[int, DQN] = {1: make_dqn_module(n_actions)}
    replay_buffer_dict = {
        1: ReplayBuffer(size=MEM_SIZE, input_shape=INPUT_SHAPE, use_per=USE_PER),
    }
    frame_number_dict: dict[int, int] = {1: 0}

    # --- DFA manager ---
    dfa_mgr = DFAManager(
        update_freq=DFA_UPDATE_FREQ,
        min_frames=MIN_DFA_FRAMES,
        module_factory=lambda: make_dqn_module(n_actions),
    )

    active_agent_id = 1
    frame_number = 0
    rewards: list[float] = []
    loss_list: list[float] = []

    # ======================= MAIN LOOP =======================
    try:
        with writer.as_default():
            while frame_number < TOTAL_FRAMES:
                epoch_frame = 0

                while epoch_frame < FRAMES_BETWEEN_EVAL:
                    start_time = time.time()
                    game_wrapper.reset()
                    terminal = False
                    episode_detected_objects: list[str] = []
                    episode_reward_sum = 0.0

                    active_agent_id = 1
                    dfa_mgr.reset_episode()

                    while not terminal:
                        # --- Action ---
                        agent_dict[active_agent_id].eps = calc_epsilon(
                            frame_number_dict[active_agent_id]
                        )
                        action = agent_dict[active_agent_id].e_greedy(
                            game_wrapper.state
                        )

                        # --- Step ---
                        processed_frame, reward, terminal, life_lost, new_obs = \
                            game_wrapper.step(action)
                        frame_number += 1
                        if frame_number > MIN_REPLAY_BUFFER_SIZE + MIN_DFA_FRAMES:
                            frame_number_dict[active_agent_id] += 1
                            epoch_frame += 1

                        # --- Object detection ---
                        old_obj_set = episode_detected_objects.copy()
                        observation = np.sum(new_obs, axis=2)
                        detect_objects(
                            observation, agent_unique,
                            episode_detected_objects,
                            dfa_mgr.episode_trace,
                        )
                        new_obj_set = np.unique(episode_detected_objects).tolist()

                        # --- DFA synthesis + transition ---
                        if frame_number >= MIN_DFA_FRAMES:
                            dfa_mgr.step(frame_number, start_time)

                            # Absorb any freshly spawned modules
                            for sid, new_agent in dfa_mgr.pop_new_agents().items():
                                agent_dict[sid] = new_agent
                                replay_buffer_dict[sid] = ReplayBuffer(
                                    size=MEM_SIZE,
                                    input_shape=INPUT_SHAPE,
                                    use_per=USE_PER,
                                )
                                frame_number_dict[sid] = 0

                            next_dfa_state = dfa_mgr.current_state
                            reward = reward + MU * intrinsic_reward(
                                new_obj_set, old_obj_set
                            )
                            episode_detected_objects = new_obj_set

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

                            # --- Train ---
                            update_all_modules(
                                agent_dict, replay_buffer_dict,
                                frame_number_dict, frame_number,
                                loss_list,
                            )

                            # --- Switch active module ---
                            active_agent_id = next_dfa_state

                        else:
                            print(f"{frame_number}: r=0 // avg_r=0"
                                  f" // objects:{new_obj_set}"
                                  f" // DFA state:1"
                                  f" // lives:{game_wrapper.last_lives}")

                        # --- Life lost ---
                        if life_lost:
                            episode_detected_objects = []
                            active_agent_id = 1
                            dfa_mgr.on_life_lost()

                    dfa_mgr.on_episode_end()
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
                eval_rewards: list[float] = []
                active_agent_id = 1
                episode_detected_objects = []
                eval_dfa = DFAManager(
                    update_freq=DFA_UPDATE_FREQ,
                    min_frames=0,
                )
                # Copy synth state so eval doesn't mutate training DFA
                eval_dfa._processed_dfa = dfa_mgr._processed_dfa
                eval_dfa._input_dict = dfa_mgr._input_dict
                eval_dfa._dfa_states = dfa_mgr._dfa_states

                for _ in range(EVAL_LENGTH):
                    if terminal:
                        game_wrapper.reset(evaluation=True)
                        episode_reward_sum = 0.0
                        terminal = False
                        episode_detected_objects = []
                        eval_dfa.reset_episode()
                        active_agent_id = 1

                    agent_dict[active_agent_id].eps = 0.0
                    action = agent_dict[active_agent_id].e_greedy(
                        game_wrapper.state
                    )

                    _, reward, terminal, life_lost, new_obs = \
                        game_wrapper.step(action)

                    old_obj_set = episode_detected_objects.copy()
                    observation = np.sum(new_obs, axis=2)
                    detect_objects(
                        observation, agent_unique,
                        episode_detected_objects,
                        eval_dfa.episode_trace,
                    )
                    new_obj_set = np.unique(episode_detected_objects).tolist()

                    if dfa_mgr.is_ready:
                        ns = eval_dfa.current_state
                        # Manually advance eval DFA state using training DFA
                        from synth.synth_wrapper import get_next_state as _gns
                        _ns = _gns(
                            eval_dfa.episode_trace,
                            dfa_mgr._input_dict['event_uniq'],
                            dfa_mgr._processed_dfa,
                        )
                        if _ns is not None and _ns != -1 and _ns != []:
                            eval_dfa._current_state = _ns

                    reward = reward + MU * intrinsic_reward(
                        new_obj_set, old_obj_set
                    )
                    episode_detected_objects = new_obj_set
                    episode_reward_sum += reward

                    next_eval_state = eval_dfa.current_state
                    if next_eval_state in agent_dict:
                        active_agent_id = next_eval_state

                    if terminal:
                        eval_rewards.append(episode_reward_sum)

                final_score = (np.mean(eval_rewards) if eval_rewards
                               else episode_reward_sum)
                print('Evaluation score:', final_score)
                if WRITE_TENSORBOARD:
                    tf.summary.scalar(
                        'Evaluation score', final_score, frame_number
                    )
                    writer.flush()

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