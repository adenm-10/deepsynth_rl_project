from minecraft.mine_craft import MineCraft
import tensorflow as tf
import pickle as pkl
import numpy as np
import random
from collections import defaultdict as ddict
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm

# SYNTH imports — original workflow
from synth.synth_wrapper import dfa_init
from synth.synth_wrapper import dfa_update
from synth.synth_wrapper import get_next_state

# DQN + PER imports
from dqn import DQN, Transition, PrioritizedReplayBuffer

###############################################################
# Hyper-parameters
###############################################################
task_num = 2
DISCOUNT_FACTOR = 0.95
DFA_UPDATE_FREQ_BASE = 5
DFA_UPDATE_FREQ_MAX = 500
DFA_ANNEAL_GROWTH = 1.5
DFA_ANNEAL_EVERY = 10
DFA_ANNEAL_WARMUP_FRAMES = 5000

# DQN architecture
STATE_DIM = 2
ACTION_DIM = 4
HIDDEN_DIM = 32
LEARNING_RATE = 1e-4

# Training
TOTAL_FRAMES = 200_000
MAX_IT_NUMBER = 250
BATCH_SIZE = 64
MEM_SIZE = 25_000
MIN_REPLAY_BUFFER_SIZE = 50
MIN_DFA_FRAMES = 2000
UPDATE_FREQ = 2
TARGET_UPDATE_FREQ = 200
PRIORITY_SCALE = 0.6
USE_PER = True
MU = 0.5
STEP_PENALTY = -0.01
CLIP_REWARD = False

# Bootstrap target clipping
TARGET_CLIP_LOW = -10.0
TARGET_CLIP_HIGH = 20.0

# Epsilon schedule
EPS_INITIAL = 1.0
EPS_FINAL = 0.1
EPS_FINAL_FRAME = 0.05
EPS_ANNEALING_STEPS = 100_000

# Eval
FRAMES_BETWEEN_EVAL = 5000
EVAL_EPISODES = 10

# NFA synthesis budget floor
MIN_NUM_STATES = 5

# Environment
ONLY_NEEDED_OBJECTS = True

SAVE_PATH = None

# Grid dims for normalization
GRID_X = 10
GRID_Y = 10

# Logging
VERBOSE = True

DFA_FREEZE_BEFORE_EVAL = 2000  # frames before eval where DFA is frozen
EVAL_LOG_ROUTING = True  # per-episode DFA routing diagnostics


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def normalize_state(xy):
    return np.array([xy[0] / (GRID_X - 1), xy[1] / (GRID_Y - 1)], dtype=np.float32)


def make_dqn():
    dqn = DQN(
        gamma=DISCOUNT_FACTOR, eps=0.0, tau=0.005, lr=LEARNING_RATE,
        double_dqn=True, dueling_dqn=True,
        state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim=HIDDEN_DIM,
    )
    dqn.hard_update_target()
    return dqn


def intrinsic_reward(new_obj_set_in, old_obj_set_in):
    new_detected_obj = list(set(new_obj_set_in) - set(old_obj_set_in))
    return 1 if new_detected_obj else 0


def dfa_result_invalid(result):
    if result is None or result == -1:
        return True
    if isinstance(result, (list, np.ndarray)) and len(result) == 0:
        return True
    return False


def calc_epsilon(frame_number, evaluation=False):
    if evaluation:
        return 0.0
    elif frame_number < MIN_REPLAY_BUFFER_SIZE:
        return EPS_INITIAL
    elif frame_number < MIN_REPLAY_BUFFER_SIZE + EPS_ANNEALING_STEPS:
        slope = -(EPS_INITIAL - EPS_FINAL) / EPS_ANNEALING_STEPS
        intercept = EPS_INITIAL - slope * MIN_REPLAY_BUFFER_SIZE
        return slope * frame_number + intercept
    else:
        slope = -(EPS_FINAL - EPS_FINAL_FRAME) / (
            TOTAL_FRAMES - EPS_ANNEALING_STEPS - MIN_REPLAY_BUFFER_SIZE)
        intercept = EPS_FINAL_FRAME - slope * TOTAL_FRAMES
        return slope * frame_number + intercept


def get_action(dqn, frame_number, state, n_actions, evaluation=False):
    eps = calc_epsilon(frame_number, evaluation)
    if np.random.rand() < eps:
        return np.random.randint(0, n_actions)
    q_vals = dqn.q_net(state.reshape(1, -1), training=False).numpy()[0]
    return int(np.argmax(q_vals))


def learn(dfa_key, dqn_dict, buffer_dict, frame_number, n_actions,
          batch_size=BATCH_SIZE, priority_scale=PRIORITY_SCALE):
    buf = buffer_dict[dfa_key]
    batch_sz = min(batch_size, len(buf))
    batch, indices, weights = buf.sample(batch_sz)

    eps = calc_epsilon(frame_number)
    weights = weights ** (1 - eps)

    rewards_b = np.array([t.r for t in batch], dtype=np.float32)
    targets = rewards_b.copy()

    dfa_groups = ddict(list)
    for k, t in enumerate(batch):
        if not t.done:
            dfa_groups[t.next_dfa_state].append(k)

    for next_dfa, group_idxs in dfa_groups.items():
        if next_dfa not in dqn_dict:
            continue
        next_dqn = dqn_dict[next_dfa]
        s_primes = np.array([batch[k].s_prime for k in group_idxs], dtype=np.float32)

        q_online = next_dqn.q_net(s_primes, training=False)
        best_actions = tf.argmax(q_online, axis=1, output_type=tf.int32)
        q_target = next_dqn.target_net(s_primes, training=False)
        idx_pairs = tf.stack([tf.range(len(group_idxs)), best_actions], axis=1)
        next_q = tf.gather_nd(q_target, idx_pairs).numpy()

        for j, k in enumerate(group_idxs):
            targets[k] = rewards_b[k] + DISCOUNT_FACTOR * next_q[j]

    targets = np.clip(targets, TARGET_CLIP_LOW, TARGET_CLIP_HIGH)
    loss, td_errors = dqn_dict[dfa_key].update_with_targets(batch, targets, weights)
    buf.update_priorities(indices, td_errors)
    return float(loss), td_errors


# ──────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────
def evaluate(env, DQN_dict, processed_dfa, input_dict,
             n_episodes=EVAL_EPISODES, max_steps=MAX_IT_NUMBER):
    episode_rewards = []
    episode_lengths = []
    completions = 0
    has_dfa = len(processed_dfa) > 0

    # Part 3: diagnostic accumulators
    all_routing_logs = []

    for ep_idx in range(n_episodes):
        current_pos = list(env.initialiser())
        current_layout = env.layout(env._world).copy()
        episode_trace = ['start']
        episode_detected_objects = []
        current_dfa_state = 1
        episode_reward_sum = 0.0
        terminal = False
        steps = 0

        # Part 3: per-episode routing log
        routing_log = []
        fallback_count = 0

        while not terminal and steps < max_steps:
            s = normalize_state(current_pos)

            # ── Part 1: hardened module selection ──
            # If current_dfa_state doesn't have a trained module, fall back to 1
            if current_dfa_state in DQN_dict:
                active_module = current_dfa_state
            else:
                active_module = 1
                fallback_count += 1

            q_vals = DQN_dict[active_module].q_net(
                s.reshape(1, -1), training=False).numpy()[0]
            action = int(np.argmax(q_vals))

            next_pos = list(env.take_action(current_pos, action))
            next_label = current_layout[next_pos[0]][next_pos[1]]
            steps += 1

            old_obj_set = episode_detected_objects.copy()
            if next_label not in (env.neutral, env.obstacles):
                episode_detected_objects.append(int(next_label))
                episode_trace.append(int(next_label))
            new_obj_set = list(set(episode_detected_objects))

            env_reward = env.reward(task_num, np.array(episode_trace[1:]))
            reward = env_reward + MU * intrinsic_reward(new_obj_set, old_obj_set) + STEP_PENALTY
            episode_detected_objects = new_obj_set
            episode_reward_sum += reward

            terminal = (env_reward > 9) or (steps >= max_steps)
            if env_reward > 9:
                completions += 1

            # ── Part 1: hardened DFA state transition ──
            prev_dfa_state = current_dfa_state
            if has_dfa:
                next_dfa_state = get_next_state(
                    episode_trace, input_dict['event_uniq'], processed_dfa)

                if dfa_result_invalid(next_dfa_state):
                    # Invalid DFA output → fall back to module 1
                    current_dfa_state = 1
                elif next_dfa_state in DQN_dict:
                    # Valid and has a trained module → use it
                    current_dfa_state = next_dfa_state
                else:
                    # Valid DFA state but no module → fall back to module 1
                    current_dfa_state = 1
                    fallback_count += 1

            # Part 3: log this step's routing
            if EVAL_LOG_ROUTING:
                routing_log.append({
                    'step': steps,
                    'requested_dfa': next_dfa_state if has_dfa else None,
                    'active_module': active_module,
                    'resolved_dfa': current_dfa_state,
                    'fell_back': (active_module != (prev_dfa_state if prev_dfa_state in DQN_dict else 1)),
                })

            if env._vanishing == 1 and \
                    current_layout[current_pos[0]][current_pos[1]] != env.workbench and \
                    current_layout[current_pos[0]][current_pos[1]] != env.toolshed:
                current_layout[current_pos[0]][current_pos[1]] = env.neutral

            current_pos = next_pos

        episode_rewards.append(episode_reward_sum)
        episode_lengths.append(steps)

        # Part 3: summarize this episode's routing
        if EVAL_LOG_ROUTING:
            modules_used = set(r['active_module'] for r in routing_log)
            invalid_requests = sum(
                1 for r in routing_log
                if r['requested_dfa'] is not None and dfa_result_invalid(r['requested_dfa'])
            )
            unmapped_requests = sum(
                1 for r in routing_log
                if r['requested_dfa'] is not None
                and not dfa_result_invalid(r['requested_dfa'])
                and r['requested_dfa'] not in DQN_dict
            )
            ep_summary = {
                'episode': ep_idx,
                'reward': episode_reward_sum,
                'steps': steps,
                'completed': env_reward > 9,
                'modules_used': sorted(modules_used),
                'fallback_count': fallback_count,
                'invalid_dfa_requests': invalid_requests,
                'unmapped_dfa_requests': unmapped_requests,
            }
            all_routing_logs.append(ep_summary)

            if VERBOSE and (fallback_count > 0 or unmapped_requests > 0):
                tqdm.write(
                    f'  [EVAL ep {ep_idx}] '
                    f'modules={sorted(modules_used)} '
                    f'fallbacks={fallback_count} '
                    f'invalid={invalid_requests} '
                    f'unmapped={unmapped_requests} '
                    f'reward={episode_reward_sum:.2f} '
                    f'steps={steps}')

    # Part 3: aggregate routing diagnostics
    routing_summary = {}
    if EVAL_LOG_ROUTING and all_routing_logs:
        total_fallbacks = sum(e['fallback_count'] for e in all_routing_logs)
        total_invalid = sum(e['invalid_dfa_requests'] for e in all_routing_logs)
        total_unmapped = sum(e['unmapped_dfa_requests'] for e in all_routing_logs)
        routing_summary = {
            'total_fallbacks': total_fallbacks,
            'total_invalid_dfa': total_invalid,
            'total_unmapped_dfa': total_unmapped,
            'per_episode': all_routing_logs,
        }
        if VERBOSE and (total_fallbacks > 0 or total_unmapped > 0):
            tqdm.write(
                f'  [EVAL ROUTING SUMMARY] '
                f'fallbacks={total_fallbacks} '
                f'invalid={total_invalid} '
                f'unmapped={total_unmapped} '
                f'across {n_episodes} episodes')

    return {
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths)),
        'completion_rate': completions / n_episodes,
        'per_episode_rewards': episode_rewards,
        'per_episode_lengths': episode_lengths,
        'routing': routing_summary,
    }


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':

    mine_craft_env = MineCraft(task_num=task_num,
                               only_needed_objects=ONLY_NEEDED_OBJECTS)
    n_actions = mine_craft_env.num_actions

    DQN_dict = {1: make_dqn()}
    buffer_dict = {1: PrioritizedReplayBuffer(size=MEM_SIZE, alpha=PRIORITY_SCALE)}
    frame_number_dict = {1: 0}

    current_dfa_state = 1
    dfa_states = [0, 1]
    frame_number = 0
    rewards = []
    episode_lengths = []
    set_of_episode_traces = []
    loss_list = []
    model_gen = []
    nfa_model = []
    dfa_model = []
    processed_dfa = []
    eval_history = []
    dfa_change_log = []
    num_states, var, input_dict, hyperparams = dfa_init()

    num_states = max(num_states, MIN_NUM_STATES)

    if VERBOSE:
        print(f"Task {task_num}, MIN_NUM_STATES={MIN_NUM_STATES}, "
              f"only_needed={ONLY_NEEDED_OBJECTS}")
    synth_iter_num = 0
    dfa_update_freq = DFA_UPDATE_FREQ_BASE

    next_eval_frame = FRAMES_BETWEEN_EVAL

    pbar = tqdm(total=TOTAL_FRAMES, desc='Training', unit='frame',
                dynamic_ncols=True, smoothing=0.05)

    try:
        while frame_number < TOTAL_FRAMES:
            start_time = time.time()

            current_pos = list(mine_craft_env.initialiser())
            current_layout = mine_craft_env.layout(mine_craft_env._world).copy()
            terminal = False
            episode_trace = ['start']
            episode_detected_objects = []
            current_dfa_state = 1
            episode_reward_sum = 0
            episode_completed = False
            iter_number = 0

            while not terminal:
                prev_frame = frame_number

                s = normalize_state(current_pos)
                action = get_action(DQN_dict[current_dfa_state],
                                    frame_number_dict[current_dfa_state],
                                    s, n_actions)

                next_pos = list(mine_craft_env.take_action(current_pos, action))
                next_label = current_layout[next_pos[0]][next_pos[1]]

                frame_number += 1
                iter_number += 1
                if frame_number > MIN_REPLAY_BUFFER_SIZE + MIN_DFA_FRAMES:
                    frame_number_dict[current_dfa_state] = frame_number_dict.get(current_dfa_state, 0) + 1

                old_obj_set = episode_detected_objects.copy()
                if next_label not in (mine_craft_env.neutral, mine_craft_env.obstacles):
                    episode_detected_objects.append(int(next_label))
                    episode_trace.append(int(next_label))
                new_obj_set = list(set(episode_detected_objects))

                env_reward = mine_craft_env.reward(task_num, np.array(episode_trace[1:]))
                terminal = (env_reward > 9) or (iter_number >= MAX_IT_NUMBER)
                if env_reward > 9:
                    episode_completed = True

                # ### SYNTH — original workflow ###
                old_dfa_states = dfa_states.copy()
                if frame_number >= MIN_DFA_FRAMES:
                    if (frame_number % dfa_update_freq == 0) or \
                       (get_next_state(episode_trace, input_dict['event_uniq'], processed_dfa) == -1) or \
                       (get_next_state(episode_trace, input_dict['event_uniq'], processed_dfa) == -2):
                            trace = []
                            set_of_episode_traces.append(episode_trace.copy())
                            for x in set_of_episode_traces:
                                trace = trace + x
                            trace = trace + ['start']

                            num_states, processed_dfa, dfa_model, nfa_model, model_gen, var, input_dict = dfa_update(
                                trace, num_states,
                                dfa_model,
                                nfa_model,
                                model_gen, var,
                                input_dict,
                                hyperparams,
                                start_time,
                                synth_iter_num)

                            # if VERBOSE:
                            #     tqdm.write(f"[DFA UPDATE] transitions={len(processed_dfa)}")
                            #     tqdm.write(f"  event_uniq: {input_dict['event_uniq']}")
                            #     tqdm.write(f"  traces fed: {len(set_of_episode_traces)}")

                            dfa_states = list(set([dfa_transitions[0] for dfa_transitions in processed_dfa] +
                                                  [dfa_transitions[2] for dfa_transitions in processed_dfa]))

                            dfa_change_log.append((
                                frame_number, len(DQN_dict), len(processed_dfa), True))

                            synth_iter_num = synth_iter_num + 1

                            # Anneal DFA update frequency after warmup
                            if frame_number >= DFA_ANNEAL_WARMUP_FRAMES and \
                                    synth_iter_num % DFA_ANNEAL_EVERY == 0:
                                dfa_update_freq = min(
                                    int(dfa_update_freq * DFA_ANNEAL_GROWTH),
                                    DFA_UPDATE_FREQ_MAX)
                                if VERBOSE:
                                    tqdm.write(f"DFA update freq annealed to {dfa_update_freq}")

                            set_of_episode_traces = [episode_trace.copy()]

                            num_states = max(num_states, MIN_NUM_STATES)

                    # Create DQN modules if necessary
                    new_dfa_states = list(set(dfa_states) - set(old_dfa_states))
                    if new_dfa_states:
                        if VERBOSE:
                            tqdm.write(f"[NEW MODULES] Creating DQNs for states: {new_dfa_states}")
                        for i in new_dfa_states:
                            DQN_dict[i] = make_dqn()
                            buffer_dict[i] = PrioritizedReplayBuffer(size=MEM_SIZE, alpha=PRIORITY_SCALE)
                            frame_number_dict[i] = 0

                    next_dfa_state = get_next_state(episode_trace, input_dict['event_uniq'], processed_dfa)

                    reward = env_reward + MU * intrinsic_reward(new_obj_set, old_obj_set) + STEP_PENALTY
                    episode_detected_objects = new_obj_set

                    if VERBOSE and frame_number % 10000 == 0:
                        avg_r = round(np.mean(rewards[-10:]), 1) if len(rewards) > 10 else 0
                        tqdm.write(
                            f'{frame_number}: r={round(reward, 2)} // avg_r={avg_r}'
                            f' // objects:{new_obj_set}'
                            f' // DFA state:{next_dfa_state}'
                            f' // modules:{len(DQN_dict)}')
                    episode_reward_sum += reward

                    # Vanishing objects
                    if mine_craft_env._vanishing == 1 and \
                            current_layout[current_pos[0]][current_pos[1]] != mine_craft_env.workbench and \
                            current_layout[current_pos[0]][current_pos[1]] != mine_craft_env.toolshed:
                        current_layout[current_pos[0]][current_pos[1]] = mine_craft_env.neutral

                    # Add experience
                    buffer_dict[current_dfa_state].add(Transition(
                        s=s,
                        a=action,
                        r=float(reward),
                        s_prime=normalize_state(next_pos),
                        done=(env_reward > 9),
                        next_dfa_state=next_dfa_state,
                    ))

                    # Update ALL agents
                    for ag in list(DQN_dict.keys()):
                        if frame_number % UPDATE_FREQ == 0 and \
                                len(buffer_dict.get(ag, [])) > MIN_REPLAY_BUFFER_SIZE:
                            loss, _ = learn(ag, DQN_dict, buffer_dict,
                                            frame_number_dict.get(ag, 0),
                                            n_actions)
                            loss_list.append(loss)
                            DQN_dict[ag].soft_update_target()

                    current_dfa_state = next_dfa_state
                else:
                    # Before MIN_DFA_FRAMES
                    reward = env_reward + MU * intrinsic_reward(new_obj_set, old_obj_set) + STEP_PENALTY
                    episode_detected_objects = new_obj_set
                    episode_reward_sum += reward

                    if mine_craft_env._vanishing == 1 and \
                            current_layout[current_pos[0]][current_pos[1]] != mine_craft_env.workbench and \
                            current_layout[current_pos[0]][current_pos[1]] != mine_craft_env.toolshed:
                        current_layout[current_pos[0]][current_pos[1]] = mine_craft_env.neutral

                current_pos = next_pos

                # Advance progress bar
                frames_advanced = frame_number - prev_frame
                pbar.update(frames_advanced)

                # Periodic evaluation
                if frame_number >= next_eval_frame:
                    eval_results = evaluate(
                        mine_craft_env, DQN_dict, processed_dfa,
                        input_dict, n_episodes=EVAL_EPISODES,
                        max_steps=MAX_IT_NUMBER)
                    eval_history.append((frame_number, eval_results))
                    next_eval_frame += FRAMES_BETWEEN_EVAL

                    tqdm.write(
                        f'[EVAL @ frame {frame_number}]  '
                        f'mean_reward={eval_results["mean_reward"]:.2f} '
                        f'(±{eval_results["std_reward"]:.2f})  '
                        f'completion={eval_results["completion_rate"]:.0%}  '
                        f'mean_len={eval_results["mean_length"]:.1f}  '
                        f'modules={len(DQN_dict)}')

                # On episode end
                if terminal:
                    episode_lengths.append(iter_number)
                    episode_detected_objects = []
                    current_dfa_state = 1
                    if episode_completed:
                        set_of_episode_traces.append(episode_trace.copy())
                        if VERBOSE:
                            tqdm.write(
                                f'[TRACE] completed: {episode_trace[:15]}'
                                f'{"..." if len(episode_trace) > 15 else ""}')
                    episode_trace = ['start']

            rewards.append(episode_reward_sum)

            # Update progress bar postfix
            postfix = {
                'ep': len(rewards),
                'modules': len(DQN_dict),
                'eps': f'{calc_epsilon(frame_number):.3f}',
                'dfa_freq': dfa_update_freq,
            }
            if len(rewards) >= 10:
                postfix['avg_r'] = f'{np.mean(rewards[-10:]):.1f}'
            if loss_list:
                postfix['loss'] = f'{np.mean(loss_list[-100:]):.4f}'
            if eval_history:
                latest = eval_history[-1][1]
                postfix['eval_r'] = f'{latest["mean_reward"]:.1f}'
                postfix['compl'] = f'{latest["completion_rate"]:.0%}'
            pbar.set_postfix(postfix)

    except KeyboardInterrupt:
        tqdm.write('\nTraining exited early.')

    pbar.close()

    # ──────────────────────────────────────────────────────────
    # Save
    # ──────────────────────────────────────────────────────────
    file_path = os.path.dirname(os.path.abspath(__file__))
    history_path = os.path.join(file_path, 'history_online')
    if not os.path.exists(history_path):
        os.mkdir(history_path)

    pkl.dump(rewards, open(os.path.join(history_path, 'episode_rewards.p'), 'wb'))
    pkl.dump(episode_lengths, open(os.path.join(history_path, 'episode_lengths.p'), 'wb'))
    pkl.dump(loss_list, open(os.path.join(history_path, 'loss_list.p'), 'wb'))
    pkl.dump(eval_history, open(os.path.join(history_path, 'eval_history.p'), 'wb'))
    pkl.dump(dfa_change_log, open(os.path.join(history_path, 'dfa_change_log.p'), 'wb'))

    for dfa_state in sorted(DQN_dict.keys()):
        DQN_dict[dfa_state].q_net.save(os.path.join(history_path, f'dqn_{dfa_state}.h5'))
        DQN_dict[dfa_state].target_net.save(os.path.join(history_path, f'target_dqn_{dfa_state}.h5'))
        if dfa_state in buffer_dict:
            buf_state = buffer_dict[dfa_state].getSaveState()
            pkl.dump(buf_state, open(os.path.join(history_path, f'buffer_{dfa_state}.p'), 'wb'))

    # ──────────────────────────────────────────────────────────
    # Plots — 2 rows × 3 columns
    # ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle(f'Task {task_num} — Original DFA Workflow', fontsize=14)

    axes[0, 0].plot(rewards, alpha=0.3, label='per episode')
    if len(rewards) >= 20:
        rolling = [np.mean(rewards[max(0, i - 19):i + 1]) for i in range(len(rewards))]
        axes[0, 0].plot(rolling, color='orange', label='rolling mean (20)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Training Episode Rewards')
    axes[0, 0].legend()

    axes[0, 1].plot(episode_lengths, alpha=0.3, label='per episode')
    if len(episode_lengths) >= 20:
        rolling_len = [np.mean(episode_lengths[max(0, i - 19):i + 1]) for i in range(len(episode_lengths))]
        axes[0, 1].plot(rolling_len, color='orange', label='rolling mean (20)')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].set_title('Training Episode Length')
    axes[0, 1].axhline(y=MAX_IT_NUMBER, color='red', linestyle='--', alpha=0.5, label=f'max ({MAX_IT_NUMBER})')
    axes[0, 1].legend()

    if loss_list:
        axes[0, 2].plot(loss_list, alpha=0.3, label='per step')
        if len(loss_list) >= 100:
            rolling_loss = [np.mean(loss_list[max(0, i - 99):i + 1]) for i in range(len(loss_list))]
            axes[0, 2].plot(rolling_loss, color='orange', label='rolling mean (100)')
        axes[0, 2].set_xlabel('Training Step')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].set_title('Training Loss')
        axes[0, 2].legend()
    else:
        axes[0, 2].set_title('Training Loss (no data)')

    if eval_history:
        eval_frames = [e[0] for e in eval_history]
        eval_means = [e[1]['mean_reward'] for e in eval_history]
        eval_stds = [e[1]['std_reward'] for e in eval_history]
        eval_comp = [e[1]['completion_rate'] for e in eval_history]
        eval_lens = [e[1]['mean_length'] for e in eval_history]

        def draw_dfa_markers(ax):
            labeled = set()
            for f, nm, nt, accepted in dfa_change_log:
                color = 'green' if accepted else 'red'
                ls = '-' if accepted else ':'
                key = 'accepted' if accepted else 'rejected'
                label = f'DFA {key}' if key not in labeled else None
                ax.axvline(x=f, color=color, linestyle=ls, alpha=0.5,
                           linewidth=1.5, label=label)
                labeled.add(key)

        axes[1, 0].plot(eval_frames, eval_means, 'o-', color='tab:blue', label='mean reward')
        axes[1, 0].fill_between(
            eval_frames,
            np.array(eval_means) - np.array(eval_stds),
            np.array(eval_means) + np.array(eval_stds),
            alpha=0.2, color='tab:blue', label='±1 std')
        axes[1, 0].set_xlabel('Frame')
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].set_title('Eval Reward vs Frame')
        draw_dfa_markers(axes[1, 0])
        axes[1, 0].legend(fontsize=7)

        axes[1, 1].plot(eval_frames, eval_comp, 's-', color='tab:green')
        axes[1, 1].set_xlabel('Frame')
        axes[1, 1].set_ylabel('Completion Rate')
        axes[1, 1].set_title('Eval Completion Rate vs Frame')
        axes[1, 1].set_ylim(-0.05, 1.05)
        draw_dfa_markers(axes[1, 1])
        axes[1, 1].legend(fontsize=7)

        axes[1, 2].plot(eval_frames, eval_lens, '^-', color='tab:purple')
        axes[1, 2].set_xlabel('Frame')
        axes[1, 2].set_ylabel('Mean Episode Length')
        axes[1, 2].set_title('Eval Episode Length vs Frame')
        axes[1, 2].axhline(y=MAX_IT_NUMBER, color='red', linestyle='--', alpha=0.5, label=f'max ({MAX_IT_NUMBER})')
        draw_dfa_markers(axes[1, 2])
        axes[1, 2].legend(fontsize=7)
    else:
        for j in range(3):
            axes[1, j].set_title(f'Eval (no data)')

    plt.tight_layout()
    plt.savefig(os.path.join(history_path, 'training_curves.png'), dpi=150)
    plt.show()