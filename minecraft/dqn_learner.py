from minecraft.mine_craft import MineCraft
import tensorflow as tf
from tensorflow import keras
import pickle as pkl
import numpy as np
import random
from collections import defaultdict as ddict
import matplotlib.pyplot as plt
import time
import os

# SYNTH imports
from synth.synth_wrapper import dfa_init
from synth.synth_wrapper import dfa_update
from synth.synth_wrapper import get_next_state

# DQN + PER imports
from dqn import DQN, Transition, PrioritizedReplayBuffer

# Hyper-parameters
task_num = 1
discount_factor = 0.95
exploration_episode_number = 200
max_it_number = 200
episode_number = 100
DFA_UPDATE_FREQ = 5
BATCH_SIZE = 64
BUFFER_SIZE = 10000
STATE_DIM = 2     # grid (x, y)
ACTION_DIM = 4
HIDDEN_DIM = 128
LR = 1e-3

# Online hyper-parameters
ONLINE_MAX_EPISODES = 500
ONLINE_MAX_IT = 200
EPS_START = 0.3
EPS_END = 0.05
EPS_DECAY_EPISODES = 200
TARGET_UPDATE_FREQ = 50       # steps between hard target updates
TRAIN_EVERY = 1               # gradient steps per env step
MIN_REPLAY_SIZE = 64          # min buffer size before training
CONVERGENCE_WINDOW = 20       # rolling window for convergence check
CONVERGENCE_PATIENCE = 30     # episodes without improvement before stopping
CONVERGENCE_DELTA = 0.01      # minimum improvement threshold


def make_dqn(state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim=HIDDEN_DIM, lr=LR):
    """Create a DQN instance with the shared architecture for this domain."""
    return DQN(
        gamma=discount_factor,
        eps=0.0,            # not used — exploration is random
        tau=0.005,
        lr=lr,
        double_dqn=True,
        dueling_dqn=True,
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
    )


if __name__ == '__main__':

    # one DQN per DFA state (buffers created after refinement)
    initial_dfa_state = 1
    dqn_dict = {initial_dfa_state: make_dqn()}

    dfa_states = [0, 1]
    set_of_episode_traces = []
    model_gen = []
    nfa_model = []
    dfa_model = []
    num_states, var, input_dict, hyperparams = dfa_init()
    synth_iter_num = 0

    #############################################################
    # Exploration
    #############################################################
    mine_craft_env = MineCraft()
    sar_dict = ddict(list)
    set_of_episode_traces = []

    # Iterate over all episodes
    for ep_n in range(exploration_episode_number):
        current_state = list(mine_craft_env.initialiser()) + [initial_dfa_state]
        current_layout = mine_craft_env.layout(mine_craft_env._world).copy()
        iter_number = 0
        episode_trace = ['start']
        start_time = time.time()

        # Iterate over timesteps in episode (while episode not done)
        while (mine_craft_env.reward(task_num, np.array(episode_trace[1:])) < 9) and (iter_number < max_it_number):

            # Take random action and get next env and dfa state
            action = random.randint(0, mine_craft_env.num_actions - 1)
            next_state_2d = list(mine_craft_env.take_action(current_state[0:2], action))
            next_state_label = current_layout[next_state_2d[0]][next_state_2d[1]]

            # if the label of the next state is an object / not default
            if next_state_label != mine_craft_env.neutral:
                episode_trace.append(next_state_label)

                #############################################################
                # Synthesize
                #############################################################

                old_dfa_states = dfa_states.copy()

                if (len(episode_trace) < 3) or \
                   (iter_number % DFA_UPDATE_FREQ == 0) or \
                   (get_next_state(episode_trace, input_dict['event_uniq'], processed_dfa) == -1) or \
                   (get_next_state(episode_trace, input_dict['event_uniq'], processed_dfa) == []):

                    trace = []
                    set_of_episode_traces.append(episode_trace)
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
                    dfa_states = list(set([dfa_transitions[0] for dfa_transitions in processed_dfa] +
                                          [dfa_transitions[2] for dfa_transitions in processed_dfa]))
                    synth_iter_num = synth_iter_num + 1
                    set_of_episode_traces = [episode_trace]

                #############################################################
                # Create DQN modules if necessary
                #############################################################

                new_dfa_states = list(set(dfa_states) - set(old_dfa_states))
                if new_dfa_states:
                    for i in new_dfa_states:
                        dqn_dict[i] = make_dqn()

                # determine next dfa state given updated dfa
                next_automaton_state = get_next_state(episode_trace, input_dict['event_uniq'], processed_dfa)
            else:
                next_automaton_state = current_state[-1]

            # vanishing objects
            if mine_craft_env._vanishing == 1 and \
                    current_layout[current_state[0]][current_state[1]] != mine_craft_env.workbench and \
                    current_layout[current_state[0]][current_state[1]] != mine_craft_env.toolshed:
                current_layout[current_state[0]][current_state[1]] = mine_craft_env.neutral

            reward = mine_craft_env.reward(task_num, np.array(episode_trace[1:]))

            # store transition in original SAR format:
            #   current env x, env y, dfa state, action, next env x, env y, next dfa state, reward
            next_state_2d.append(next_automaton_state)
            sar = current_state + [action] + next_state_2d + [reward]
            sar_dict[current_state[-1]].append(sar)

            current_state = next_state_2d
            set_of_episode_traces.append(episode_trace)

            iter_number += 1

    #############################################################
    # Deduplicate SARS (identical to original)
    #############################################################
    sars = []
    normal_refinary = []
    rew = None

    # for each dfa state
    dfa_keys = list(sar_dict.keys())
    for i in range(len(dfa_keys)):
        # deduplicate transitions
        sars.append(np.unique(np.array(sar_dict[dfa_keys[i]]), axis=0))

        # if dfa state wasn't the final dfa state, add to refinery, else note the index
        if sum(sars[i][:, 7] > 9) == 0:
            normal_refinary.append(i)
        else:
            rew = i

        # ensure a DQN exists for this dfa state
        if dfa_keys[i] not in dqn_dict:
            dqn_dict[dfa_keys[i]] = make_dqn()

    if rew is None:
        print('\n please consider tuning exploration parameters, e.g. increasing exploration_episode_number and '
              'max_it_number \n')

    #############################################################
    # Refine SARS (identical to original)
    #############################################################
    exp_size = 1500

    # for each non-terminal automaton:
    #   if there are more experiences than exp_size, randomly downsample
    for i in normal_refinary:
        if len(sars[i]) > exp_size:
            sars[i] = np.delete(sars[i], random.sample(range(0, len(sars[i])), len(sars[i]) - exp_size), axis=0)

    # rebalance terminal dfa state: keep ~10 low-reward + all high-reward transitions
    reward_column = sars[rew][:, 7]
    indx_rew = np.where([reward_column > 9])
    high_reward_sar_rew = sars[rew][indx_rew[1]]
    sars[rew] = np.delete(sars[rew], random.sample(range(0, len(sars[rew])), len(sars[rew]) - 10 + len(indx_rew[1])),
                          axis=0)
    sars[rew] = np.vstack((sars[rew], high_reward_sar_rew))

    #############################################################
    # Load refined data into PER buffers
    #############################################################
    buffer_dict = {}
    for i in range(len(dfa_keys)):
        buf = PrioritizedReplayBuffer(size=BUFFER_SIZE)
        for row in sars[i]:
            # row: [env_x, env_y, dfa_state, action, next_x, next_y, next_dfa_state, reward]
            next_dfa = int(row[6])
            if next_dfa not in dqn_dict:
                next_dfa = max(dqn_dict.keys())
            buf.add(Transition(
                s=np.array(row[0:2], dtype=np.float32),
                a=int(row[3]),
                r=float(row[7]),
                s_prime=np.array(row[4:6], dtype=np.float32),
                done=row[7] > 9,
                next_dfa_state=next_dfa,
            ))
        buffer_dict[dfa_keys[i]] = buf

    #############################################################
    # Initialize target networks
    #############################################################
    for dfa_state in dqn_dict:
        dqn_dict[dfa_state].hard_update_target()

    #############################################################
    # Train Models on Offline Data
    #############################################################
    history = ddict(list)
    utility = []

    for i in range(episode_number):
        print(f'{int(i / episode_number * 100)}%')

        # estimate utility of initial state (4,4) under initial dfa state's model
        q_vals = dqn_dict[initial_dfa_state].q_net(
            np.array([[4.0, 4.0]], dtype=np.float32), training=False)
        utility.append(float(tf.reduce_max(q_vals).numpy()))

        # for each dfa state in reverse order (terminal -> initial):
        for idx in range(len(dfa_keys) - 1, -1, -1):
            dfa_state = dfa_keys[idx]
            buf = buffer_dict[dfa_state]
            if len(buf) == 0:
                continue

            # ~3 passes over the buffer to match original's epochs=3
            n_steps = max(1, 3 * len(buf) // BATCH_SIZE)
            batch_sz = min(BATCH_SIZE, len(buf))

            epoch_losses = []
            for _ in range(n_steps):
                batch, indices, weights = buf.sample(batch_sz)

                # compute cross-module double-DQN targets
                targets = np.zeros(len(batch), dtype=np.float32)
                for k, t in enumerate(batch):
                    if t.done:
                        targets[k] = t.r
                        continue

                    next_dfa = t.next_dfa_state
                    next_dqn = dqn_dict[next_dfa]
                    s_p = np.array([t.s_prime], dtype=np.float32)

                    # online net of next dfa state selects best action
                    q_online = next_dqn.q_net(s_p, training=False)
                    best_a = int(tf.argmax(q_online, axis=1).numpy()[0])

                    # target net of next dfa state evaluates
                    q_target = next_dqn.target_net(s_p, training=False)
                    next_q = float(q_target[0, best_a].numpy())

                    targets[k] = t.r + discount_factor * next_q

                loss, td_errors = dqn_dict[dfa_state].update_with_targets(batch, targets, weights)
                buf.update_priorities(indices, td_errors)
                epoch_losses.append(loss)

            # hard target update after each training episode
            #   (keeps targets stable during the next episode's gradient steps)
            dqn_dict[dfa_state].hard_update_target()
            history[dfa_state].append(epoch_losses)

    #############################################################
    # Online Training
    #############################################################
    online_rewards = []
    online_utility = []
    global_step = 0
    best_rolling_mean = -np.inf
    patience_counter = 0

    print("Starting online training")

    for ep_n in range(ONLINE_MAX_EPISODES):

        # print(f'Online Episode Progress: {ep_n} / {ONLINE_MAX_EPISODES} | {int(ep_n / ONLINE_MAX_EPISODES * 100)}%')

        # linearly decay epsilon from EPS_START to EPS_END over EPS_DECAY_EPISODES
        eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * ep_n / EPS_DECAY_EPISODES)

        # set epsilon for all DQN modules
        for dfa_state in dqn_dict:
            dqn_dict[dfa_state].eps = eps

        current_state = list(mine_craft_env.initialiser()) + [initial_dfa_state]
        current_layout = mine_craft_env.layout(mine_craft_env._world).copy()
        iter_number = 0
        episode_trace = ['start']
        episode_reward_sum = 0.0
        start_time = time.time()

        # iterate over timesteps in episode (while episode not done)
        while (mine_craft_env.reward(task_num, np.array(episode_trace[1:])) < 9) and (iter_number < ONLINE_MAX_IT):

            # print(f'{int(iter_number / ONLINE_MAX_IT * 100)}%')

            # select action via e-greedy from the current dfa state's DQN
            current_dfa = current_state[-1]
            s = np.array(current_state[0:2], dtype=np.float32)
            action = dqn_dict[current_dfa].e_greedy(s)

            next_state_2d = list(mine_craft_env.take_action(current_state[0:2], action))
            next_state_label = current_layout[next_state_2d[0]][next_state_2d[1]]

            # if the label of the next state is an object / not default
            if next_state_label != mine_craft_env.neutral:
                episode_trace.append(next_state_label)

                #############################################################
                # Synthesize
                #############################################################

                old_dfa_states = dfa_states.copy()

                if (get_next_state(episode_trace, input_dict['event_uniq'], processed_dfa) == -1) or \
                   (get_next_state(episode_trace, input_dict['event_uniq'], processed_dfa) == []):

                    trace = []
                    set_of_episode_traces.append(episode_trace)
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
                    dfa_states = list(set([dfa_transitions[0] for dfa_transitions in processed_dfa] +
                                          [dfa_transitions[2] for dfa_transitions in processed_dfa]))
                    synth_iter_num = synth_iter_num + 1
                    set_of_episode_traces = [episode_trace]

                #############################################################
                # Create DQN modules if necessary
                #############################################################

                # if new DFA state, make new DQN and buffer
                new_dfa_states = list(set(dfa_states) - set(old_dfa_states))
                if new_dfa_states:
                    for i in new_dfa_states:
                        dqn_dict[i] = make_dqn()
                        dqn_dict[i].hard_update_target()
                        buffer_dict[i] = PrioritizedReplayBuffer(size=BUFFER_SIZE)

                # determine next dfa state given updated dfa
                next_automaton_state = get_next_state(episode_trace, input_dict['event_uniq'], processed_dfa)
            else:
                # else, dfa state stays the same
                next_automaton_state = current_state[-1]

            # vanishing objects
            if mine_craft_env._vanishing == 1 and \
                    current_layout[current_state[0]][current_state[1]] != mine_craft_env.workbench and \
                    current_layout[current_state[0]][current_state[1]] != mine_craft_env.toolshed:
                current_layout[current_state[0]][current_state[1]] = mine_craft_env.neutral

            reward = mine_craft_env.reward(task_num, np.array(episode_trace[1:]))
            done = reward > 9
            episode_reward_sum += reward

            # clamp next_dfa_state to a valid key
            if next_automaton_state not in dqn_dict:
                next_dfa_key = max(dqn_dict.keys())
            else:
                next_dfa_key = next_automaton_state

            # ensure buffer and dqn exist for current dfa state
            if current_dfa not in buffer_dict:
                buffer_dict[current_dfa] = PrioritizedReplayBuffer(size=BUFFER_SIZE)
            if current_dfa not in dqn_dict:
                dqn_dict[current_dfa] = make_dqn()
                dqn_dict[current_dfa].hard_update_target()

            # store transition in the current dfa state's buffer
            buffer_dict[current_dfa].add(Transition(
                s=s,
                a=action,
                r=float(reward),
                s_prime=np.array(next_state_2d[0:2], dtype=np.float32),
                done=done,
                next_dfa_state=next_dfa_key,
            ))

            #############################################################
            # Train only the active DQN module (the one that just got new data)
            #############################################################

            if global_step % TRAIN_EVERY == 0:
                buf = buffer_dict.get(current_dfa)
                if buf is not None and len(buf) >= MIN_REPLAY_SIZE:
                    batch_sz = min(BATCH_SIZE, len(buf))
                    batch, indices, weights = buf.sample(batch_sz)

                    # compute cross-module double-DQN targets
                    #   group non-terminal transitions by next_dfa_state
                    #   so we do one batched forward pass per dfa module instead of one per transition
                    rewards = np.array([t.r for t in batch], dtype=np.float32)
                    dones = np.array([t.done for t in batch], dtype=np.float32)
                    targets = rewards.copy()

                    # collect indices of non-terminal transitions grouped by next dfa state
                    dfa_groups = ddict(list)
                    for k, t in enumerate(batch):
                        if not t.done:
                            dfa_groups[t.next_dfa_state].append(k)

                    # one batched forward pass per dfa module
                    for next_dfa, group_idxs in dfa_groups.items():
                        next_dqn = dqn_dict[next_dfa]
                        s_primes = np.array([batch[k].s_prime for k in group_idxs], dtype=np.float32)

                        # online net selects best actions, target net evaluates
                        q_online = next_dqn.q_net(s_primes, training=False)
                        best_actions = tf.argmax(q_online, axis=1, output_type=tf.int32)
                        q_target = next_dqn.target_net(s_primes, training=False)
                        idx_pairs = tf.stack([tf.range(len(group_idxs)), best_actions], axis=1)
                        next_q = tf.gather_nd(q_target, idx_pairs).numpy()

                        for j, k in enumerate(group_idxs):
                            targets[k] = rewards[k] + discount_factor * next_q[j]

                    loss, td_errors = dqn_dict[current_dfa].update_with_targets(batch, targets, weights)
                    buf.update_priorities(indices, td_errors)

            # hard update target network for the active module only
            if global_step % TARGET_UPDATE_FREQ == 0:
                dqn_dict[current_dfa].hard_update_target()

            next_state_2d.append(next_automaton_state)
            current_state = next_state_2d
            set_of_episode_traces.append(episode_trace)
            iter_number += 1
            global_step += 1

        online_rewards.append(episode_reward_sum)

        # estimate utility of initial state (4,4) under initial dfa state's model
        q_vals = dqn_dict[initial_dfa_state].q_net(
            np.array([[4.0, 4.0]], dtype=np.float32), training=False)
        online_utility.append(float(tf.reduce_max(q_vals).numpy()))

        # append to combined utility for full offline+online plot
        utility.append(online_utility[-1])

        # convergence check: rolling mean of episode returns
        if len(online_rewards) >= CONVERGENCE_WINDOW:
            rolling_mean = np.mean(online_rewards[-CONVERGENCE_WINDOW:])
            if rolling_mean > best_rolling_mean + CONVERGENCE_DELTA:
                best_rolling_mean = rolling_mean

        if (ep_n + 1) % 10 == 0:
            avg_r = np.mean(online_rewards[-10:])
            print(f'Online episode {ep_n + 1}/{ONLINE_MAX_EPISODES}  '
                  f'eps={eps:.3f}  avg_reward={avg_r:.2f}  '
                  f'utility={online_utility[-1]:.4f}  ')

    #############################################################
    # Plot and Save
    #############################################################
    file_path = os.path.dirname(os.path.abspath(__file__))
    history_path = os.path.join(file_path, 'history')
    if not os.path.exists(history_path):
        os.mkdir(history_path)

    # save buffers, utility curve, and training history
    pkl.dump(sars, open(os.path.join(history_path, 'sars.p'), 'wb'))
    pkl.dump(utility, open(os.path.join(history_path, 'utility.p'), 'wb'))
    pkl.dump(dict(history), open(os.path.join(history_path, 'history.p'), 'wb'))
    pkl.dump(online_rewards, open(os.path.join(history_path, 'online_rewards.p'), 'wb'))

    for dfa_state in sorted(dqn_dict.keys()):
        # save each dfa state's DQN weights
        dqn_dict[dfa_state].q_net.save(os.path.join(history_path, f'dqn_{dfa_state}.h5'))
        dqn_dict[dfa_state].target_net.save(os.path.join(history_path, f'target_dqn_{dfa_state}.h5'))

        # save replay buffer state
        if dfa_state in buffer_dict:
            buf_state = buffer_dict[dfa_state].getSaveState()
            pkl.dump(buf_state, open(os.path.join(history_path, f'buffer_{dfa_state}.p'), 'wb'))

    # plot utility convergence with offline/online boundary
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(utility)
    ax1.axvline(x=episode_number, color='r', linestyle='--', label='offline → online')
    ax1.set_xlabel('Training Episode')
    ax1.set_ylabel('Max Q at (4,4)')
    ax1.set_title('Utility Convergence')
    ax1.legend()

    ax2.plot(online_rewards)
    if len(online_rewards) >= CONVERGENCE_WINDOW:
        # plot rolling mean
        rolling = [np.mean(online_rewards[max(0, i - CONVERGENCE_WINDOW + 1):i + 1])
                   for i in range(len(online_rewards))]
        ax2.plot(rolling, color='orange', label=f'rolling mean ({CONVERGENCE_WINDOW})')
    ax2.set_xlabel('Online Episode')
    ax2.set_ylabel('Episode Reward')
    ax2.set_title('Online Rewards')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(history_path, 'training_curves.png'), dpi=150)
    plt.show()