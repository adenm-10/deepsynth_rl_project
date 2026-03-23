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
import h5py

# SYNTH imports
from synth.synth_wrapper import dfa_init
from synth.synth_wrapper import dfa_update
from synth.synth_wrapper import get_next_state

# Hyper-parameters
task_num = 3
discount_factor = 0.95
exploration_episode_number = 200
max_it_number = 200
episode_number = 100
DFA_UPDATE_FREQ = 5
prop = keras.optimizers.RMSprop(learning_rate=0.01)
###

def evaluate_policy_nfq(models, env, task_num, num_episodes=50, max_steps=200):
    successes = 0
    all_steps = []

    for k in range(num_episodes):
        state_2d = list(env.initialiser())
        layout = env.layout(env._world).copy()
        automaton_state = 1
        trace = []
        steps = max_steps

        print(f"Eval Episode: {k}")

        for step in range(max_steps):
            # print(f"Eval Step: {step}")
        
            # Select action: argmax_a Q(x, y, a) from the active module
            model_idx = min(automaton_state - 1, len(models) - 1)
            q_vals = []
            for a in range(env.num_actions):
                inp = np.array([[state_2d[0], state_2d[1], a]], dtype=np.float32)
                q_vals.append(models[model_idx].predict(inp, verbose=0)[0, 0])
            action = int(np.argmax(q_vals))

            next_2d = list(env.take_action(state_2d, action))
            label = layout[next_2d[0]][next_2d[1]]

            if label != env.neutral:
                trace.append(label)
                # Update automaton state using ground-truth automaton
                automaton_state = env.automaton(task_num, automaton_state, label)

            # Update layout for vanishing objects
            if env._vanishing == 1 and \
                    layout[state_2d[0]][state_2d[1]] != env.workbench and \
                    layout[state_2d[0]][state_2d[1]] != env.toolshed:
                layout[state_2d[0]][state_2d[1]] = env.neutral

            # Check task completion
            if env.reward(task_num, np.array(trace)) > 9:
                successes += 1
                steps = step
                break

            state_2d = next_2d

        all_steps.append(steps)

    success_rate = successes / num_episodes
    avg_steps = sum(all_steps) / len(all_steps)
    # Average steps only for successful episodes
    success_steps = [s for s in all_steps if s < max_steps]
    avg_success_steps = sum(success_steps) / len(success_steps) if success_steps else float('inf')

    print(f"Success rate: {success_rate:.2%} ({successes}/{num_episodes})")
    print(f"Avg steps (all): {avg_steps:.1f}")
    print(f"Avg steps (successes only): {avg_success_steps:.1f}")

    return success_rate, avg_steps, avg_success_steps


def run_exploration_nfq(mine_craft_env, task_num):
    """Exploration + DFA synthesis. Returns sar_dict, NFQ_dict, processed_dfa, input_dict.
       Code is identical to the original __main__ exploration loop."""

    MAIN_NFQ = keras.Sequential([
        keras.layers.Dense(128, input_dim=3, activation=tf.nn.relu),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(1, activation='sigmoid')])
    MAIN_NFQ.compile(loss='mean_squared_error',
                     metrics=['mean_squared_error'],
                     optimizer='Adam')
    NFQ_dict = {
        1: MAIN_NFQ
    }

    dfa_states = [0, 1]
    set_of_episode_traces = []
    model_gen = []
    nfa_model = []
    dfa_model = []
    num_states, var, input_dict, hyperparams = dfa_init()
    synth_iter_num = 0

    sar_dict = ddict(list)
    set_of_episode_traces = []
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
            if next_state_label != mine_craft_env.neutral:  # ignoring the neutral label
                episode_trace.append(next_state_label)
                # ### SYNTH ### #
                old_dfa_states = dfa_states.copy()
                # # SYNTH updates the automaton here:
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

                # Create NFQ modules if necessary
                new_dfa_states = list(set(dfa_states) - set(old_dfa_states))
                if new_dfa_states:
                    for i in new_dfa_states:
                        # Initiate new DQN modules
                        NFQ_dict[i] = keras.Sequential([
                            keras.layers.Dense(128, input_dim=3, activation=tf.nn.relu),
                            keras.layers.Dense(128, activation=tf.nn.relu),
                            keras.layers.Dense(1, activation='sigmoid')])
                        NFQ_dict[i].compile(loss='mean_squared_error',
                                            metrics=['mean_squared_error'],
                                            optimizer='Adam')
                # Determine next dfa state
                next_automaton_state = get_next_state(episode_trace, input_dict['event_uniq'], processed_dfa)
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

    return sar_dict, NFQ_dict, processed_dfa, input_dict


def refine_sars_nfq(sar_dict, NFQ_dict):
    """Deduplicate and downsample. Returns sars list, models list, rew index.
       Code is identical to the original __main__ refinement block."""

    sars = []
    models = []
    normal_refinary = []
    rew = None
    for i in range(len(list(sar_dict.keys()))):
        sars.append(np.unique(np.array(sar_dict[list(sar_dict.keys())[i]]), axis=0))
        if sum(sars[i][:, 7] > 9) == 0:
            normal_refinary.append(i)
        else:
            rew = i
        models.append(NFQ_dict[list(sar_dict.keys())[i]])

    if rew is None:
        print('\n please consider tuning exploration parameters, e.g. increasing exploration_episode_number and '
              'max_it_number \n')
        return sars, models, rew

    # refine sars
    exp_size = 1500
    for i in normal_refinary:
        if len(sars[i]) > exp_size:
            sars[i] = np.delete(sars[i], random.sample(range(0, len(sars[i])), len(sars[i]) - exp_size), axis=0)
    reward_column = sars[rew][:, 7]
    indx_rew = np.where([reward_column > 9])
    high_reward_sar_rew = sars[rew][indx_rew[1]]
    sars[rew] = np.delete(sars[rew], random.sample(range(0, len(sars[rew])), len(sars[rew]) - 10 + len(indx_rew[1])),
                          axis=0)
    sars[rew] = np.vstack((sars[rew], high_reward_sar_rew))

    for i, s in enumerate(sars):
        print(f"Automaton state {i}: {len(s)} transitions")
    print(f"Total: {sum(len(s) for s in sars)} transitions")

    return sars, models, rew


def train_nfq(sars, models, episode_number=100, eval_freq=None,
              eval_fn=None):
    """NFQ training loop. Returns models, utility, history.
       If eval_freq and eval_fn are provided, calls eval_fn every eval_freq epochs.
       eval_fn signature: eval_fn(models) -> (success_rate, avg_steps, avg_succ_steps)
       Code inside the loop is identical to the original __main__ training block."""

    history = ddict(list)
    eval_results = {"epochs": [], "sr": [], "steps": []}

    # initialization
    for i in range(len(models)):
        models[i].fit(np.hstack((sars[i][:, 0:2], sars[i][:, 3:4])), sars[i][:, 7:8], epochs=3, verbose=0)

    init_state = np.array([4, 4, 1])
    utility = []
    for i in range(episode_number):
        print(int(i / episode_number * 100), '%')
        neighs = []
        for l in range(4):
            neigh_inputs = []
            neigh_inputs = np.append([4, 4], l).reshape(1, 3)
            neighs.append(models[0].predict(neigh_inputs, verbose=0).item())

        utility.append(max(neighs))
        for j in range(len(models) - 1, -1, -1):
            target = np.zeros(len(sars[j]))
            for k in range(len(sars[j])):
                neigh = []
                for l in range(4):
                    neigh_input = []
                    neigh_input = np.append(sars[j][k, 4:6], l).reshape(1, 3)
                    neigh.append(models[min(int(sars[j][k, 6]) - 1, len(models) - 1)].predict(neigh_input, verbose=0).item())
                target[k] = sars[j][k, 7] + discount_factor * max(neigh)

            history_j = models[j].fit(np.hstack((sars[j][:, 0:2], sars[j][:, 3:4])),
                                      target.reshape(len(np.hstack((sars[j][:, 0:2], sars[j][:, 3:4]))), 1),
                                      epochs=3,
                                      verbose=0)
            history[j].append(history_j)

        # Periodic evaluation callback
        if eval_freq and eval_fn and (i + 1) % eval_freq == 0:
            sr, avg_all, avg_succ = eval_fn(models)
            eval_results["epochs"].append(i + 1)
            eval_results["sr"].append(sr)
            eval_results["steps"].append(avg_succ)
            print(f"    [NFQ] epoch {i+1}: success={sr:.0%}, steps={avg_succ:.1f}")

    return models, utility, history, eval_results


def main():
    mine_craft_env = MineCraft()

    # Exploration
    sar_dict, NFQ_dict, processed_dfa, input_dict = run_exploration_nfq(mine_craft_env, task_num)

    # Refinement
    sars, models, rew = refine_sars_nfq(sar_dict, NFQ_dict)
    if rew is None:
        return

    # Training
    models, utility, history, _ = train_nfq(sars, models, episode_number)

    # Save
    file_path = os.path.dirname(os.path.abspath(__file__))
    history_path = os.path.join(file_path, 'history')
    if not os.path.exists(history_path):
        os.mkdir(history_path)
    pkl.dump(sars, open(os.path.join(history_path, 'sars.p'), 'wb'))
    pkl.dump(utility, open(os.path.join(history_path, 'utility.p'), 'wb'))
    for i in range(len(models) - 1, -1, -1):
        models[i].save(os.path.join(history_path, 'model_' + str(i + 1) + '.h5'))
        for j in range(episode_number):
            pkl.dump(history[i][j].history,
                     open(os.path.join(history_path, 'history_' + str(i + 1) + '_' + str(j + 1) + '.p'), "wb"))
    plt.plot(np.vstack(utility).tolist())
    plt.show()

    evaluate_policy_nfq(models, mine_craft_env, task_num, num_episodes=50, max_steps=200)


if __name__ == '__main__':
    main()