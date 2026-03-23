import argparse
import sys
import numpy as np
import random
import pickle as pkl
import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import defaultdict as ddict

from minecraft.mine_craft import MineCraft
from dqn import DQN, QLearningBuffer, Transition
from synth.synth_wrapper import dfa_init, dfa_update, get_next_state

# ── Shared hyper-parameters ──
task_num = 5
discount_factor = 0.95
hidden_dim = 128
lr = 1e-3

# Exploration (shared by both modes, matches original)
exploration_episode_number = 200
max_it_number = 200
DFA_UPDATE_FREQ = 5

# Offline mode
offline_epochs = 100        # matches original episode_number
exp_size = 1500             # matches original refinery cap

# Online mode
online_episodes = 2000
max_online_steps = 200
online_update_freq = 4
online_batch_size = 64
buffer_size = 50_000
eval_freq = 50
eps_start = 1.0
eps_end = 0.05
eps_decay_episodes = 1000


def parse_args():
    parser = argparse.ArgumentParser(description="DQN learner for MineCraft + DeepSynth")
    parser.add_argument("--mode", choices=["online", "offline"], required=True,
                        help="'offline' = train on exploration data only (like NFQ baseline). "
                             "'online' = train while interacting with env.")
    parser.add_argument("--double", action="store_true", default=False)
    parser.add_argument("--dueling", action="store_true", default=False)
    parser.add_argument("--use-automaton", action="store_true", default=False)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


def make_state(state_2d, automaton_state=None, use_automaton=False):
    if use_automaton:
        return np.array([state_2d[0], state_2d[1], automaton_state], dtype=np.float32)
    return np.array([state_2d[0], state_2d[1]], dtype=np.float32)


def evaluate_policy(dqn, env, task_num, num_episodes=50, max_steps=200,
                    use_automaton=False, input_dict=None, processed_dfa=None):
    successes = 0
    all_steps = []

    for _ in range(num_episodes):
        state_2d = list(env.initialiser())
        layout = env.layout(env._world).copy()
        automaton_state = 1
        trace = []
        episode_trace = ['start']
        steps = max_steps

        for step in range(max_steps):
            s = make_state(state_2d, automaton_state, use_automaton)
            action = dqn.e_greedy(s, pure_greedy=True)
            next_2d = list(env.take_action(state_2d, action))
            label = layout[next_2d[0]][next_2d[1]]

            if label != env.neutral:
                episode_trace.append(label)
                trace.append(label)
                if use_automaton and input_dict is not None and processed_dfa is not None:
                    new_state = get_next_state(
                        episode_trace, input_dict['event_uniq'], processed_dfa)
                    if new_state not in (-1, []):
                        automaton_state = new_state

            if env._vanishing == 1 and \
                    layout[state_2d[0]][state_2d[1]] != env.workbench and \
                    layout[state_2d[0]][state_2d[1]] != env.toolshed:
                layout[state_2d[0]][state_2d[1]] = env.neutral

            if env.reward(task_num, np.array(trace)) > 9:
                successes += 1
                steps = step + 1
                break

            state_2d = next_2d

        all_steps.append(steps)

    success_rate = successes / num_episodes
    avg_steps = sum(all_steps) / len(all_steps)
    success_steps = [s for s in all_steps if s < max_steps]
    avg_success_steps = sum(success_steps) / len(success_steps) if success_steps else float('inf')

    print(f"  Success rate: {success_rate:.2%} ({successes}/{num_episodes})")
    print(f"  Avg steps (all): {avg_steps:.1f}")
    print(f"  Avg steps (successes only): {avg_success_steps:.1f}")

    return success_rate, avg_steps, avg_success_steps


# ══════════════════════════════════════════════════════
# Exploration + DFA Synthesis
#   (shared by both modes, identical to original)
# ══════════════════════════════════════════════════════

def run_exploration(mine_craft_env):
    """Run exploration with DFA synthesis. Returns sar_dict, processed_dfa, input_dict."""
    print("── Exploration + DFA Synthesis ──")

    dfa_states = [0, 1]
    set_of_episode_traces = []
    model_gen, nfa_model, dfa_model = [], [], []
    num_states, var, input_dict, hyperparams = dfa_init()
    synth_iter_num = 0
    processed_dfa = None

    sar_dict = ddict(list)

    for ep_n in range(exploration_episode_number):
        current_state = list(mine_craft_env.initialiser()) + [1]
        current_layout = mine_craft_env.layout(mine_craft_env._world).copy()
        iter_number = 0
        episode_trace = ['start']
        start_time = time.time()

        while mine_craft_env.reward(task_num, np.array(episode_trace[1:])) < 9 \
                and iter_number < max_it_number:
            action = random.randint(0, mine_craft_env.num_actions - 1)
            next_state_2d = list(mine_craft_env.take_action(current_state[0:2], action))
            next_state_label = current_layout[next_state_2d[0]][next_state_2d[1]]

            if next_state_label != mine_craft_env.neutral:
                episode_trace.append(next_state_label)
                old_dfa_states = dfa_states.copy()

                if (len(episode_trace) < 3) or \
                        (iter_number % DFA_UPDATE_FREQ == 0) or \
                        (processed_dfa is not None and
                         get_next_state(episode_trace, input_dict['event_uniq'], processed_dfa) in (-1, [])):
                    trace = []
                    set_of_episode_traces.append(episode_trace)
                    for x in set_of_episode_traces:
                        trace = trace + x
                    trace = trace + ['start']
                    num_states, processed_dfa, dfa_model, nfa_model, model_gen, var, input_dict = dfa_update(
                        trace, num_states,
                        dfa_model, nfa_model,
                        model_gen, var,
                        input_dict, hyperparams,
                        start_time, synth_iter_num)
                    dfa_states = list(set(
                        [t[0] for t in processed_dfa] +
                        [t[2] for t in processed_dfa]))
                    synth_iter_num += 1
                    set_of_episode_traces = [episode_trace]

                next_automaton_state = get_next_state(
                    episode_trace, input_dict['event_uniq'], processed_dfa)
            else:
                next_automaton_state = current_state[-1]

            if mine_craft_env._vanishing == 1 and \
                    current_layout[current_state[0]][current_state[1]] != mine_craft_env.workbench and \
                    current_layout[current_state[0]][current_state[1]] != mine_craft_env.toolshed:
                current_layout[current_state[0]][current_state[1]] = mine_craft_env.neutral

            next_state_2d.append(next_automaton_state)
            sar = current_state + \
                  [action] + \
                  next_state_2d + \
                  [mine_craft_env.reward(task_num, np.array(episode_trace[1:]))]
            sar_dict[current_state[-1]].append(sar)
            current_state = next_state_2d
            set_of_episode_traces.append(episode_trace)
            iter_number += 1

    return sar_dict, processed_dfa, input_dict


def refine_sars(sar_dict):
    """Dedup and downsample, identical to original nfq_learner.py."""
    print("\n── SARS Refinement ──")

    sars = []
    state_keys = list(sar_dict.keys())
    normal_refinery = []
    rew_idx = None

    for i, key in enumerate(state_keys):
        arr = np.unique(np.array(sar_dict[key]), axis=0)
        sars.append(arr)
        if np.any(arr[:, 7] > 9):
            rew_idx = i
        else:
            normal_refinery.append(i)

    if rew_idx is None:
        print("No positive reward transitions found.")
        print("Consider increasing exploration_episode_number or max_it_number.")
        exit(1)

    # Downsample non-reward states
    for i in normal_refinery:
        if len(sars[i]) > exp_size:
            remove_n = len(sars[i]) - exp_size
            sars[i] = np.delete(sars[i], random.sample(range(len(sars[i])), remove_n), axis=0)

    # Reward state: keep all high-reward + up to 10 low-reward (matches original intent)
    reward_column = sars[rew_idx][:, 7]
    high_mask = reward_column > 9
    high_reward_sars = sars[rew_idx][high_mask]
    low_reward_sars = sars[rew_idx][~high_mask]
    n_low_keep = min(10, len(low_reward_sars))
    if len(low_reward_sars) > n_low_keep:
        keep = random.sample(range(len(low_reward_sars)), n_low_keep)
        low_reward_sars = low_reward_sars[keep]
    sars[rew_idx] = np.vstack([low_reward_sars, high_reward_sars])

    for i, s in enumerate(sars):
        n_pos = np.sum(s[:, 7] > 9)
        print(f"  Automaton state {state_keys[i]}: {len(s)} transitions ({n_pos} positive)")
    print(f"  Total: {sum(len(s) for s in sars)} transitions")

    return sars, state_keys


# ══════════════════════════════════════════════════════
# Mode 1: Offline-only
#   (same data, same backward schedule as NFQ baseline)
# ══════════════════════════════════════════════════════

def run_offline(args, mine_craft_env, sars, state_keys, processed_dfa, input_dict):
    print("\n── Offline Training ──")
    state_dim = 3 if args.use_automaton else 2

    # Group transitions by automaton state
    sars_by_key = {}
    for i, key in enumerate(state_keys):
        transitions = []
        for row in sars[i]:
            if args.use_automaton:
                s = row[0:3].astype(np.float32)
                s_prime = row[4:7].astype(np.float32)
            else:
                s = row[0:2].astype(np.float32)
                s_prime = row[4:6].astype(np.float32)
            a = int(row[3])
            r = float(row[7])
            done = row[7] > 9.0
            transitions.append(Transition(s=s, a=a, r=r, s_prime=s_prime, done=done))
        sars_by_key[key] = transitions

    dqn = DQN(
        gamma=discount_factor, eps=0.1, tau=0.0, lr=lr,
        double_dqn=args.double, dueling_dqn=args.dueling,
        state_dim=state_dim, action_dim=4, hidden_dim=hidden_dim,
    )

    if args.use_automaton:
        init_state = np.array([[4.0, 4.0, 1.0]], dtype=np.float32)
    else:
        init_state = np.array([[4.0, 4.0]], dtype=np.float32)

    utility = []
    losses = []
    sorted_keys = sorted(sars_by_key.keys(), reverse=True)

    for epoch in range(offline_epochs):
        q_vals = dqn.q_net(init_state, training=False)
        utility.append(float(tf.reduce_max(q_vals).numpy()))

        epoch_losses = []

        # Backward through DFA states (Algorithm 1)
        for key in sorted_keys:
            transitions = sars_by_key[key]
            if not transitions:
                continue

            # 3 fitting passes (matches epochs=3 in original model.fit)
            for _ in range(3):
                if len(transitions) <= online_batch_size:
                    loss = dqn.update(transitions)
                    epoch_losses.append(loss)
                else:
                    indices = list(range(len(transitions)))
                    random.shuffle(indices)
                    for start in range(0, len(transitions), online_batch_size):
                        end = min(start + online_batch_size, len(transitions))
                        batch = [transitions[indices[idx]] for idx in range(start, end)]
                        loss = dqn.update(batch)
                        epoch_losses.append(loss)

            if args.double:
                dqn.target_net.set_weights(dqn.q_net.get_weights())

        losses.append(np.mean(epoch_losses))

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{offline_epochs}: "
                  f"loss={losses[-1]:.4f}, max_Q={utility[-1]:.4f}")

    # Evaluate
    print("\n── Offline Evaluation ──")
    evaluate_policy(dqn, mine_craft_env, task_num, num_episodes=50, max_steps=200,
                    use_automaton=args.use_automaton,
                    input_dict=input_dict, processed_dfa=processed_dfa)

    return dqn, utility, losses


# ══════════════════════════════════════════════════════
# Mode 2: Online
#   (uses exploration data to seed buffer, then trains
#    by interacting with env)
# ══════════════════════════════════════════════════════

def run_online(args, mine_craft_env, sars, state_keys, processed_dfa, input_dict):
    print("\n── Online Training ──")
    state_dim = 3 if args.use_automaton else 2

    # Seed replay buffer with exploration data
    replay_buffer = QLearningBuffer(buffer_size)
    for i, key in enumerate(state_keys):
        for row in sars[i]:
            if args.use_automaton:
                s = row[0:3].astype(np.float32)
                s_prime = row[4:7].astype(np.float32)
            else:
                s = row[0:2].astype(np.float32)
                s_prime = row[4:6].astype(np.float32)
            a = int(row[3])
            r = float(row[7])
            done = row[7] > 9.0
            replay_buffer.add(Transition(s=s, a=a, r=r, s_prime=s_prime, done=done))

    print(f"  Buffer seeded with {len(replay_buffer)} exploration transitions")

    dqn = DQN(
        gamma=discount_factor, eps=eps_start, tau=0.0, lr=lr,
        double_dqn=args.double, dueling_dqn=args.dueling,
        state_dim=state_dim, action_dim=4, hidden_dim=hidden_dim,
    )

    if args.use_automaton:
        init_state = np.array([[4.0, 4.0, 1.0]], dtype=np.float32)
    else:
        init_state = np.array([[4.0, 4.0]], dtype=np.float32)

    utility = []
    online_returns = []
    online_success = []
    online_steps_per_ep = []
    total_steps = 0

    for ep in range(online_episodes):
        frac = min(ep / eps_decay_episodes, 1.0)
        dqn.eps = eps_start + frac * (eps_end - eps_start)

        # Track Q at init
        q_vals = dqn.q_net(init_state, training=False)
        utility.append(float(tf.reduce_max(q_vals).numpy()))

        state_2d = list(mine_craft_env.initialiser())
        layout = mine_craft_env.layout(mine_craft_env._world).copy()
        automaton_state = 1
        trace = []
        episode_trace = ['start']
        ep_reward = 0.0
        done = False

        for step in range(max_online_steps):
            s = make_state(state_2d, automaton_state, args.use_automaton)
            action = dqn.e_greedy(s, pure_greedy=False)
            next_2d = list(mine_craft_env.take_action(state_2d, action))
            label = layout[next_2d[0]][next_2d[1]]

            next_automaton_state = automaton_state
            if label != mine_craft_env.neutral:
                episode_trace.append(label)
                trace.append(label)
                if args.use_automaton:
                    new_state = get_next_state(
                        episode_trace, input_dict['event_uniq'], processed_dfa)
                    if new_state not in (-1, []):
                        next_automaton_state = new_state

            if mine_craft_env._vanishing == 1 and \
                    layout[state_2d[0]][state_2d[1]] != mine_craft_env.workbench and \
                    layout[state_2d[0]][state_2d[1]] != mine_craft_env.toolshed:
                layout[state_2d[0]][state_2d[1]] = mine_craft_env.neutral

            r = mine_craft_env.reward(task_num, np.array(trace)) if trace else 0.0
            done = r > 9.0
            ep_reward += r

            s_prime = make_state(next_2d, next_automaton_state, args.use_automaton)
            replay_buffer.add(Transition(s=s, a=action, r=r, s_prime=s_prime, done=done))
            total_steps += 1

            if total_steps % online_update_freq == 0 and len(replay_buffer) >= online_batch_size:
                batch = replay_buffer.sample(online_batch_size)
                dqn.update(batch)

            if done:
                break

            state_2d = next_2d
            automaton_state = next_automaton_state

        online_steps_per_ep.append(step + 1)
        online_returns.append(ep_reward)
        online_success.append(1 if done else 0)

        if args.double and (ep + 1) % 200 == 0:
            dqn.target_net.set_weights(dqn.q_net.get_weights())

        if (ep + 1) % eval_freq == 0:
            recent_rate = sum(online_success[-eval_freq:]) / eval_freq
            recent_return = sum(online_returns[-eval_freq:]) / eval_freq
            recent_steps = sum(online_steps_per_ep[-eval_freq:]) / eval_freq
            print(f"  Ep {ep+1}/{online_episodes} (eps={dqn.eps:.3f}): "
                  f"success={recent_rate:.2%}, return={recent_return:.1f}, "
                  f"steps={recent_steps:.1f}")

    # Evaluate
    print("\n── Online Evaluation ──")
    evaluate_policy(dqn, mine_craft_env, task_num, num_episodes=50, max_steps=200,
                    use_automaton=args.use_automaton,
                    input_dict=input_dict, processed_dfa=processed_dfa)

    return dqn, utility, online_returns, online_success, online_steps_per_ep


# ══════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════

def main(args):
    mine_craft_env = MineCraft()

    # Shared exploration phase
    sar_dict, processed_dfa, input_dict = run_exploration(mine_craft_env)
    sars, state_keys = refine_sars(sar_dict)

    # Dispatch to mode
    if args.mode == "offline":
        dqn, utility, losses = run_offline(
            args, mine_craft_env, sars, state_keys, processed_dfa, input_dict)

        # ── Plot ──
        tag = f"offline_{'automaton' if args.use_automaton else 'no_automaton'}"
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(utility)
        axes[0].set_title('max Q at (4,4)')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('max Q')

        axes[1].plot(losses)
        axes[1].set_title('Training loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MSE')

    elif args.mode == "online":
        dqn, utility, online_returns, online_success, online_steps = run_online(
            args, mine_craft_env, sars, state_keys, processed_dfa, input_dict)

        # ── Plot ──
        tag = f"online_{'automaton' if args.use_automaton else 'no_automaton'}"
        window = eval_freq
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        axes[0, 0].plot(utility)
        axes[0, 0].set_title('max Q at (4,4)')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('max Q')

        rolling_success = [
            sum(online_success[max(0, i - window):i]) / min(i, window)
            for i in range(1, len(online_success) + 1)
        ]
        axes[0, 1].plot(rolling_success)
        axes[0, 1].set_title(f'Success rate (rolling {window})')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Success rate')
        axes[0, 1].set_ylim(-0.05, 1.05)

        rolling_return = [
            sum(online_returns[max(0, i - window):i]) / min(i, window)
            for i in range(1, len(online_returns) + 1)
        ]
        axes[1, 0].plot(rolling_return)
        axes[1, 0].set_title(f'Avg return (rolling {window})')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Return')

        rolling_steps = [
            sum(online_steps[max(0, i - window):i]) / min(i, window)
            for i in range(1, len(online_steps) + 1)
        ]
        axes[1, 1].plot(rolling_steps)
        axes[1, 1].set_title(f'Avg steps (rolling {window})')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Steps')

    # ── Save ──
    file_path = os.path.dirname(os.path.abspath(__file__))
    history_path = os.path.join(file_path, 'history', tag)
    os.makedirs(history_path, exist_ok=True)

    dqn.q_net.save(os.path.join(history_path, 'dqn_model.h5'))
    if args.double:
        dqn.target_net.save(os.path.join(history_path, 'dqn_target.h5'))
    pkl.dump(utility, open(os.path.join(history_path, 'utility.p'), 'wb'))

    plt.suptitle(tag, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(history_path, 'training_curves.png'), dpi=150)
    plt.show()
    print(f"\nResults saved to {history_path}")


if __name__ == '__main__':
    args = parse_args()
    print(f"Config: mode={args.mode}, double={args.double}, dueling={args.dueling}, "
          f"use_automaton={args.use_automaton}")
    main(args)