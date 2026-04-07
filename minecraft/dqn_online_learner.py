"""
DeepSynth DQN Online Learner — Hydra-managed experiments.

Usage:
    # Run with defaults (task2, default env, baseline algo, default dfa):
    python dqn_online_learner.py

    # Run a named experiment preset:
    python dqn_online_learner.py +experiment=task2_baseline

    # Override individual groups:
    python dqn_online_learner.py task=task3 env=vanishing algo=stable

    # Override a single param:
    python dqn_online_learner.py algo.learning_rate=0.0005

    # Multirun ablation:
    python dqn_online_learner.py --multirun env=default,vanishing algo=baseline,stable
"""

from minecraft.mine_craft import MineCraft
import tensorflow as tf
import pickle as pkl
import numpy as np
import random
from collections import defaultdict as ddict
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for experiment runs
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf

# SYNTH imports
from synth.synth_wrapper import dfa_init, dfa_update, get_next_state

# DQN + PER imports
from dqn import DQN, Transition, PrioritizedReplayBuffer

# Plotting
from experiment_utils import save_plots  # lives alongside this file in minecraft/


# ──────────────────────────────────────────────────────────────
# Helpers — all take cfg (or cfg sub-groups) instead of globals
# ──────────────────────────────────────────────────────────────

def normalize_state(xy, cfg):
    gx = cfg.task.grid_x
    gy = cfg.task.grid_y
    return np.array([xy[0] / (gx - 1), xy[1] / (gy - 1)], dtype=np.float32)


def make_dqn(cfg, source_dqn=None):
    algo = cfg.algo
    dqn = DQN(
        gamma=algo.discount_factor, eps=0.0, tau=algo.tau, lr=algo.learning_rate,
        double_dqn=True, dueling_dqn=True,
        state_dim=cfg.task.state_dim, action_dim=cfg.task.action_dim,
        hidden_dim=algo.hidden_dim,
    )
    if source_dqn is not None:
        for tv, sv in zip(dqn.q_net.trainable_variables,
                          source_dqn.q_net.trainable_variables):
            tv.assign(sv)
        for tv, sv in zip(dqn.target_net.trainable_variables,
                          source_dqn.target_net.trainable_variables):
            tv.assign(sv)
    dqn.hard_update_target()
    return dqn


def intrinsic_reward(new_obj_set_in, old_obj_set_in):
    return 1 if set(new_obj_set_in) - set(old_obj_set_in) else 0


def dfa_result_invalid(result):
    if result is None or result == -1:
        return True
    if isinstance(result, (list, np.ndarray)) and len(result) == 0:
        return True
    return False


def calc_epsilon(frame_number, cfg, evaluation=False):
    algo = cfg.algo
    if evaluation:
        return 0.0
    elif frame_number < algo.min_replay_buffer_size:
        return algo.eps_initial
    elif frame_number < algo.min_replay_buffer_size + algo.eps_annealing_steps:
        slope = -(algo.eps_initial - algo.eps_final) / algo.eps_annealing_steps
        intercept = algo.eps_initial - slope * algo.min_replay_buffer_size
        return slope * frame_number + intercept
    else:
        slope = -(algo.eps_final - algo.eps_final_frame) / (
            algo.total_frames - algo.eps_annealing_steps - algo.min_replay_buffer_size)
        intercept = algo.eps_final_frame - slope * algo.total_frames
        return slope * frame_number + intercept


def get_action(dqn, frame_number, state, n_actions, cfg, evaluation=False):
    eps = calc_epsilon(frame_number, cfg, evaluation)
    if np.random.rand() < eps:
        return np.random.randint(0, n_actions)
    q_vals = dqn.q_net(state.reshape(1, -1), training=False).numpy()[0]
    return int(np.argmax(q_vals))


def learn(dfa_key, dqn_dict, buffer_dict, frame_number, n_actions, cfg):
    algo = cfg.algo
    buf = buffer_dict[dfa_key]
    batch_sz = min(algo.batch_size, len(buf))
    batch, indices, weights = buf.sample(batch_sz)

    eps = calc_epsilon(frame_number, cfg)
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
            targets[k] = rewards_b[k] + algo.discount_factor * next_q[j]

    targets = np.clip(targets, algo.target_clip_low, algo.target_clip_high)
    loss, td_errors = dqn_dict[dfa_key].update_with_targets(batch, targets, weights)
    buf.update_priorities(indices, td_errors)
    return float(loss), td_errors


# ──────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────

def evaluate(env, DQN_dict, processed_dfa, input_dict, cfg):
    algo = cfg.algo
    task_num = cfg.task.task_num
    n_episodes = algo.eval_episodes
    max_steps = algo.max_it_number

    episode_rewards = []
    episode_lengths = []
    completions = 0
    has_dfa = len(processed_dfa) > 0
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
        routing_log = []
        fallback_count = 0

        while not terminal and steps < max_steps:
            s = normalize_state(current_pos, cfg)

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
            reward = (env_reward
                      + algo.mu * intrinsic_reward(new_obj_set, old_obj_set)
                      + algo.step_penalty)
            episode_detected_objects = new_obj_set
            episode_reward_sum += reward

            terminal = (env_reward > 9) or (steps >= max_steps)
            if env_reward > 9:
                completions += 1

            prev_dfa_state = current_dfa_state
            next_dfa_state = None
            if has_dfa:
                next_dfa_state = get_next_state(
                    episode_trace, input_dict['event_uniq'], processed_dfa)
                if dfa_result_invalid(next_dfa_state):
                    current_dfa_state = 1
                elif next_dfa_state in DQN_dict:
                    current_dfa_state = next_dfa_state
                else:
                    current_dfa_state = 1
                    fallback_count += 1

            if algo.eval_log_routing:
                routing_log.append({
                    'step': steps,
                    'requested_dfa': next_dfa_state,
                    'active_module': active_module,
                    'resolved_dfa': current_dfa_state,
                })

            if env._vanishing == 1 and \
                    current_layout[current_pos[0]][current_pos[1]] != env.workbench and \
                    current_layout[current_pos[0]][current_pos[1]] != env.toolshed:
                current_layout[current_pos[0]][current_pos[1]] = env.neutral

            current_pos = next_pos

        episode_rewards.append(episode_reward_sum)
        episode_lengths.append(steps)

        if algo.eval_log_routing:
            modules_used = set(r['active_module'] for r in routing_log)
            invalid_requests = sum(
                1 for r in routing_log
                if r['requested_dfa'] is not None and dfa_result_invalid(r['requested_dfa']))
            unmapped_requests = sum(
                1 for r in routing_log
                if r['requested_dfa'] is not None
                and not dfa_result_invalid(r['requested_dfa'])
                and r['requested_dfa'] not in DQN_dict)
            all_routing_logs.append({
                'episode': ep_idx,
                'reward': episode_reward_sum,
                'steps': steps,
                'completed': env_reward > 9,
                'modules_used': sorted(modules_used),
                'fallback_count': fallback_count,
                'invalid_dfa_requests': invalid_requests,
                'unmapped_dfa_requests': unmapped_requests,
            })

            if algo.verbose and (fallback_count > 0 or unmapped_requests > 0):
                tqdm.write(
                    f'  [EVAL ep {ep_idx}] '
                    f'modules={sorted(modules_used)} '
                    f'fallbacks={fallback_count} '
                    f'unmapped={unmapped_requests} '
                    f'reward={episode_reward_sum:.2f}')

    routing_summary = {}
    if algo.eval_log_routing and all_routing_logs:
        total_fallbacks = sum(e['fallback_count'] for e in all_routing_logs)
        total_unmapped = sum(e['unmapped_dfa_requests'] for e in all_routing_logs)
        routing_summary = {
            'total_fallbacks': total_fallbacks,
            'total_unmapped_dfa': total_unmapped,
            'per_episode': all_routing_logs,
        }

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
# Main — Hydra entry point
# ──────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Print resolved config
    if cfg.algo.verbose:
        print(OmegaConf.to_yaml(cfg))

    # Hydra auto-creates output dir; we add subdirs
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    history_path = os.path.join(output_dir, 'history')
    plots_path = os.path.join(output_dir, 'plots')
    os.makedirs(history_path, exist_ok=True)
    os.makedirs(plots_path, exist_ok=True)

    # ── Aliases for brevity in training loop ──
    algo = cfg.algo
    dfa_cfg = cfg.dfa
    task_num = cfg.task.task_num

    # ── Environment ──
    mine_craft_env = MineCraft(
        task_num=task_num,
        only_needed_objects=cfg.env.only_needed_objects,
        vanishing=1 if cfg.env.vanishing else 0,
    )
    n_actions = mine_craft_env.num_actions

    # ── Agent state ──
    DQN_dict = {1: make_dqn(cfg)}
    buffer_dict = {1: PrioritizedReplayBuffer(
        size=algo.mem_size, alpha=algo.priority_scale)}
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
    # dfa_init() uses its own argparse internally — shield it from
    # Hydra's argv so it only sees the script name.
    import sys
    _saved_argv = sys.argv
    sys.argv = [sys.argv[0]]
    num_states, var, input_dict, hyperparams = dfa_init()
    sys.argv = _saved_argv

    num_states = max(num_states, dfa_cfg.min_num_states)

    if algo.verbose:
        print(f"[EXPERIMENT] {cfg.experiment_name}")
        print(f"[OUTPUT]     {output_dir}")
        print(f"Task {task_num}, MIN_NUM_STATES={dfa_cfg.min_num_states}, "
              f"only_needed={cfg.env.only_needed_objects}, "
              f"vanishing={cfg.env.vanishing}")

    synth_iter_num = 0
    dfa_update_freq = dfa_cfg.dfa_update_freq_base
    next_eval_frame = algo.frames_between_eval

    pbar = tqdm(total=algo.total_frames, desc='Training', unit='frame',
                dynamic_ncols=True, smoothing=0.05)

    try:
        while frame_number < algo.total_frames:
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

                s = normalize_state(current_pos, cfg)
                action = get_action(DQN_dict[current_dfa_state],
                                    frame_number_dict[current_dfa_state],
                                    s, n_actions, cfg)

                next_pos = list(mine_craft_env.take_action(current_pos, action))
                next_label = current_layout[next_pos[0]][next_pos[1]]

                frame_number += 1
                iter_number += 1
                if frame_number > algo.min_replay_buffer_size + dfa_cfg.min_dfa_frames:
                    frame_number_dict[current_dfa_state] = \
                        frame_number_dict.get(current_dfa_state, 0) + 1

                old_obj_set = episode_detected_objects.copy()
                if next_label not in (mine_craft_env.neutral, mine_craft_env.obstacles):
                    episode_detected_objects.append(int(next_label))
                    episode_trace.append(int(next_label))
                new_obj_set = list(set(episode_detected_objects))

                env_reward = mine_craft_env.reward(task_num, np.array(episode_trace[1:]))
                terminal = (env_reward > 9) or (iter_number >= algo.max_it_number)
                if env_reward > 9:
                    episode_completed = True

                # ── DFA synthesis ──
                old_dfa_states = dfa_states.copy()
                dfa_frozen = (next_eval_frame - frame_number) <= dfa_cfg.dfa_freeze_before_eval

                if frame_number >= dfa_cfg.min_dfa_frames and not dfa_frozen:
                    if (frame_number % dfa_update_freq == 0) or \
                       (get_next_state(episode_trace, input_dict['event_uniq'], processed_dfa) == -1) or \
                       (get_next_state(episode_trace, input_dict['event_uniq'], processed_dfa) == -2):
                            trace = []
                            set_of_episode_traces.append(episode_trace.copy())
                            for x in set_of_episode_traces:
                                trace = trace + x
                            trace = trace + ['start']

                            num_states, processed_dfa, dfa_model, nfa_model, model_gen, var, input_dict = dfa_update(
                                trace, num_states, dfa_model, nfa_model,
                                model_gen, var, input_dict, hyperparams,
                                start_time, synth_iter_num)

                            dfa_states = list(set(
                                [d[0] for d in processed_dfa] +
                                [d[2] for d in processed_dfa]))

                            dfa_change_log.append((
                                frame_number, len(DQN_dict),
                                len(processed_dfa), True))

                            synth_iter_num += 1

                            if frame_number >= dfa_cfg.dfa_anneal_warmup_frames and \
                                    synth_iter_num % dfa_cfg.dfa_anneal_every == 0:
                                dfa_update_freq = min(
                                    int(dfa_update_freq * dfa_cfg.dfa_anneal_growth),
                                    dfa_cfg.dfa_update_freq_max)
                                if algo.verbose:
                                    tqdm.write(f"DFA update freq → {dfa_update_freq}")

                            set_of_episode_traces = [episode_trace.copy()]
                            num_states = max(num_states, dfa_cfg.min_num_states)

                if frame_number >= dfa_cfg.min_dfa_frames:
                    # Create new modules (warm-started)
                    new_dfa_states = list(set(dfa_states) - set(old_dfa_states))
                    if new_dfa_states:
                        root_dqn = DQN_dict.get(1, None) if algo.warm_start_modules else None
                        if algo.verbose:
                            mode = "warm-start" if algo.warm_start_modules else "random-init"
                            tqdm.write(f"[NEW MODULES] {mode}: {new_dfa_states}")
                        for i in new_dfa_states:
                            DQN_dict[i] = make_dqn(cfg, source_dqn=root_dqn)
                            buffer_dict[i] = PrioritizedReplayBuffer(
                                size=algo.mem_size, alpha=algo.priority_scale)
                            frame_number_dict[i] = 0

                    next_dfa_state = get_next_state(
                        episode_trace, input_dict['event_uniq'], processed_dfa)

                    reward = (env_reward
                              + algo.mu * intrinsic_reward(new_obj_set, old_obj_set)
                              + algo.step_penalty)
                    episode_detected_objects = new_obj_set

                    if algo.verbose and frame_number % 10000 == 0:
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

                    # Store transition
                    buffer_dict[current_dfa_state].add(Transition(
                        s=s, a=action, r=float(reward),
                        s_prime=normalize_state(next_pos, cfg),
                        done=(env_reward > 9),
                        next_dfa_state=next_dfa_state,
                    ))

                    # Update ALL agents
                    for ag in list(DQN_dict.keys()):
                        min_buf = (algo.min_replay_buffer_size if ag == 1
                                   else algo.min_replay_buffer_size_child)
                        if frame_number % algo.update_freq == 0 and \
                                len(buffer_dict.get(ag, [])) > min_buf:
                            loss, _ = learn(ag, DQN_dict, buffer_dict,
                                            frame_number_dict.get(ag, 0),
                                            n_actions, cfg)
                            loss_list.append(loss)
                            DQN_dict[ag].soft_update_target()

                    current_dfa_state = next_dfa_state
                else:
                    # Before MIN_DFA_FRAMES
                    reward = (env_reward
                              + algo.mu * intrinsic_reward(new_obj_set, old_obj_set)
                              + algo.step_penalty)
                    episode_detected_objects = new_obj_set
                    episode_reward_sum += reward

                    if mine_craft_env._vanishing == 1 and \
                            current_layout[current_pos[0]][current_pos[1]] != mine_craft_env.workbench and \
                            current_layout[current_pos[0]][current_pos[1]] != mine_craft_env.toolshed:
                        current_layout[current_pos[0]][current_pos[1]] = mine_craft_env.neutral

                current_pos = next_pos
                pbar.update(frame_number - prev_frame)

                # Periodic evaluation
                if frame_number >= next_eval_frame:
                    # Module diagnostics
                    if algo.verbose:
                        parts = []
                        for ag in sorted(DQN_dict.keys()):
                            buf_sz = len(buffer_dict.get(ag, []))
                            fn = frame_number_dict.get(ag, 0)
                            parts.append(f"m{ag}(buf={buf_sz}, steps={fn})")
                        tqdm.write(f"  [MODULE DIAG] {' | '.join(parts)}")

                    eval_results = evaluate(
                        mine_craft_env, DQN_dict, processed_dfa,
                        input_dict, cfg)
                    eval_history.append((frame_number, eval_results))
                    next_eval_frame += algo.frames_between_eval

                    tqdm.write(
                        f'[EVAL @ {frame_number}]  '
                        f'mean_reward={eval_results["mean_reward"]:.2f} '
                        f'(±{eval_results["std_reward"]:.2f})  '
                        f'completion={eval_results["completion_rate"]:.0%}  '
                        f'mean_len={eval_results["mean_length"]:.1f}  '
                        f'modules={len(DQN_dict)}')

                # Episode end
                if terminal:
                    episode_lengths.append(iter_number)
                    episode_detected_objects = []
                    current_dfa_state = 1
                    if episode_completed:
                        set_of_episode_traces.append(episode_trace.copy())
                        if algo.verbose:
                            tqdm.write(
                                f'[TRACE] completed: {episode_trace[:15]}'
                                f'{"..." if len(episode_trace) > 15 else ""}')
                    episode_trace = ['start']

            rewards.append(episode_reward_sum)

            postfix = {
                'ep': len(rewards),
                'modules': len(DQN_dict),
                'eps': f'{calc_epsilon(frame_number, cfg):.3f}',
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

    # ── Save ─────────────────────────────────────────────────
    pkl.dump(rewards, open(os.path.join(history_path, 'episode_rewards.p'), 'wb'))
    pkl.dump(episode_lengths, open(os.path.join(history_path, 'episode_lengths.p'), 'wb'))
    pkl.dump(loss_list, open(os.path.join(history_path, 'loss_list.p'), 'wb'))
    pkl.dump(eval_history, open(os.path.join(history_path, 'eval_history.p'), 'wb'))
    pkl.dump(dfa_change_log, open(os.path.join(history_path, 'dfa_change_log.p'), 'wb'))

    for dfa_state in sorted(DQN_dict.keys()):
        DQN_dict[dfa_state].q_net.save(
            os.path.join(history_path, f'dqn_{dfa_state}.h5'))
        DQN_dict[dfa_state].target_net.save(
            os.path.join(history_path, f'target_dqn_{dfa_state}.h5'))
        if dfa_state in buffer_dict:
            buf_state = buffer_dict[dfa_state].getSaveState()
            pkl.dump(buf_state, open(
                os.path.join(history_path, f'buffer_{dfa_state}.p'), 'wb'))

    # ── Plots ────────────────────────────────────────────────
    save_plots(
        rewards=rewards,
        episode_lengths=episode_lengths,
        loss_list=loss_list,
        eval_history=eval_history,
        dfa_change_log=dfa_change_log,
        cfg=cfg,
        plots_dir=plots_path,
    )

    print(f"[DONE] Results saved to {output_dir}")


if __name__ == '__main__':
    main()