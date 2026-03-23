import os
import sys
import time
import tensorflow as tf
from tensorflow import keras
import pickle as pkl
import numpy as np
import random
from collections import defaultdict as ddict
import matplotlib.pyplot as plt
import h5py

# SYNTH imports
from synth.synth_wrapper import dfa_init
from synth.synth_wrapper import dfa_update
from synth.synth_wrapper import get_next_state

# ============================================================================
# HYPERPARAMETERS
# ============================================================================
discount_factor = 0.95
exploration_episode_number = 20   # number of random exploration episodes to collect traces
max_it_number = 50                # max steps per exploration episode
episode_number = 150              # number of NFQ training epochs (fitted Q-iteration passes)
DFA_UPDATE_FREQ = 1               # how often (in steps) to re-run automata synthesis
prop = keras.optimizers.RMSprop(lr=0.01)

# ============================================================================
# INITIAL NFQ MODULE
# ============================================================================
# This is the Neural Fitted Q-iteration (NFQ) network.
# IMPORTANT: Unlike a standard DQN which takes state -> Q(s, a) for ALL actions,
# this network takes (x, y, action) as a 3D input and outputs a SINGLE scalar Q-value.
# To find the best action, you must loop over all 4 actions and pick the max.
# This is the "fitted Q" style from Riedmiller 2005.
#
# Architecture: input(3) -> Dense(128, relu) -> Dense(128, relu) -> Dense(1, sigmoid)
# The sigmoid output squashes Q-values to [0, 1], which works here because
# rewards are either ~0 (no task completion) or ~10 (task done), and after
# normalization/discounting, targets stay in a bounded range.
MAIN_NFQ = keras.Sequential([
    keras.layers.Dense(128, input_dim=3, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(1, activation='sigmoid')])
MAIN_NFQ.compile(loss='mean_squared_error',
                 metrics=['mean_squared_error'],
                 optimizer='Adam')

# NFQ_dict maps DFA state -> its dedicated neural network.
# This is the MODULAR architecture from the paper (Fig. 3):
# each DFA state gets its own Q-network, so each sub-task is
# learned by a separate module.
# We start with one module for DFA state 1 (the initial DFA state).
NFQ_dict = {
    1: MAIN_NFQ
}

# ============================================================================
# DFA / SYNTH INITIALIZATION
# ============================================================================
# dfa_states tracks which DFA states have been discovered so far.
# The synth algorithm will grow this list as exploration reveals new
# sequential structure in the traces.
dfa_states = [0, 1]
set_of_episode_traces = []
model_gen = []
nfa_model = []
dfa_model = []
# dfa_init() sets up the SAT-based automata synthesis engine.
# Returns: num_states (current DFA size), var/input_dict/hyperparams
# (internal synth state that gets threaded through dfa_update calls).
num_states, var, input_dict, hyperparams = dfa_init()
synth_iter_num = 0

# ============================================================================
# ENVIRONMENT LAYOUT (Frozen Lake variant)
# ============================================================================
# 12x10 grid. Each cell has an integer label:
#   0 = neutral (empty space, no semantic meaning)
#   1 = objective_1 (the goal — reaching this gives reward)
#   2 = objective_2
#   3 = objective_3
#   4 = objective_4
#   5 = unsafe (terminates the episode)
#
# The layout places 2x2 blocks of each label type at specific grid positions.
# This is a simplified version of the frozen-lake benchmarks in Table 1 of the paper.
layout = np.zeros([12, 10], dtype=int)
layout[4:6, 7:9] = np.ones([2, 2], dtype=int) * 1   # objective_1 at rows 4-5, cols 7-8
layout[7:9, 7:9] = np.ones([2, 2], dtype=int) * 2   # objective_2 at rows 7-8, cols 7-8
layout[6:8, 3:5] = np.ones([2, 2], dtype=int) * 3   # objective_3 at rows 6-7, cols 3-4
layout[7:9, 0:2] = np.ones([2, 2], dtype=int) * 4   # objective_4 at rows 7-8, cols 0-1
layout[3:5, 3:5] = np.ones([2, 2], dtype=int) * 5   # unsafe zone at rows 3-4, cols 3-4

layout_dict = {
    'neutral': 0,
    'objective_1': 1,
    'objective_2': 2,
    'objective_3': 3,
    'objective_4': 4,
    'unsafe': 5
}


def reward(ep_trace, layout_dict):
    """
    Non-Markovian reward: depends on the ENTIRE episode trace, not just current state.
    Returns ~10 if objective_1 has been visited at any point in the trace,
    otherwise returns a tiny random number ~0.
    
    This is the sparse reward the paper talks about — you only get signal
    when the full high-level task is accomplished.
    """
    if layout_dict['objective_1'] in ep_trace:
        return round(10 + random.random() / 100, 2)
    else:
        return round(random.random() / 100, 2)


def take_action(current_state, action_indx):
    """
    Stochastic transition function: 90% of the time executes the intended action,
    10% of the time picks a random direction. This matches the stochastic
    frozen-lake benchmarks in the paper (Table 1 notes stochastic MDPs).
    
    Actions: 0=right, 1=up, 2=left, 3=down
    """
    rand_gen = random.random()
    direction_deltas = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    if rand_gen > 0.9:
        next_state = current_state + random.choice(direction_deltas)
    else:
        next_state = current_state + direction_deltas[action_indx]
    # Clip to grid boundaries
    if next_state[0] < 0:
        next_state[0] = 0
    if next_state[0] > 11:
        next_state[0] = 11
    if next_state[1] < 0:
        next_state[1] = 0
    if next_state[1] > 9:
        next_state[1] = 9
    return next_state


# ============================================================================
# PHASE 1: EXPLORATION + TRACE COLLECTION + DFA SYNTHESIS
# ============================================================================
# This is "Step 1: Tracing" and "Step 2: Synth" from Fig. 1 in the paper.
#
# The agent explores randomly, collecting:
#   1. episode_trace: sequence of semantic labels visited (e.g. ['start', 3, 1])
#      -> fed to the Synth algorithm to build/update the DFA
#   2. sar_dict: state-action-reward tuples, keyed by DFA state
#      -> used later for NFQ training (Step 3)
#
# sar_dict[q] contains transitions that occurred while the product MDP
# was in DFA state q. This is the "projection of E onto qi" (Section C, Appendix).

sar_dict = ddict(list)
set_of_episode_traces = []
for ep_n in range(exploration_episode_number):
    ###
    # State representation: [x, y, dfa_state]
    # The DFA state is appended to the grid position — this is the
    # PRODUCT MDP construction (Definition 5.3): S⊗ = S × Q
    # Starting at grid position (0, 9) in DFA state 1.
    current_state = [0, 9, 1]
    iter_number = 1
    episode_trace = ['start']  # traces always begin with 'start' delimiter
    start_time = time.time()

    # Episode loop: run until task is solved (reward > 9) or max steps reached
    while reward(episode_trace, layout_dict) < 9 and iter_number < max_it_number:
        iter_number += 1
        action = random.randint(0, 3)  # purely random exploration
        next_state_2d = list(take_action(current_state[0:-1], action))
        next_state_label = layout[next_state_2d[0]][next_state_2d[1]]

        if next_state_label != 0:  # only track semantically meaningful visits (not neutral)
            episode_trace.append(next_state_label)

            # ===================== SYNTH UPDATE =====================
            # This is the online DFA synthesis loop.
            # The DFA is re-synthesized when:
            #   - We've hit the update frequency threshold, OR
            #   - The current trace leads to an unknown DFA state (-1), OR
            #   - The current trace leads to an empty transition ([])
            # The latter two cases mean the DFA doesn't know what to do
            # with the new observation, so it needs to be expanded.
            old_dfa_states = dfa_states.copy()

            if (iter_number % DFA_UPDATE_FREQ == 0) or \
                    (get_next_state(episode_trace, input_dict['event_uniq'], processed_dfa) == -1) or \
                    (get_next_state(episode_trace, input_dict['event_uniq'], processed_dfa) == []):
                # Concatenate ALL episode traces collected so far into one long trace.
                # The synth algorithm processes this as a single sequence with 'start'
                # delimiters between episodes.
                trace = []
                set_of_episode_traces.append(episode_trace)
                for x in set_of_episode_traces:
                    trace = trace + x
                trace = trace + ['start']

                # dfa_update() runs the SAT-based synthesis:
                # Input: concatenated trace, current DFA state
                # Output: processed_dfa (list of transitions [src, label, dst]),
                #         updated synth state variables
                # The key property (Appendix B): the new DFA is always a SUPERSET
                # of the previous one — states are only added, never removed.
                # This means existing NFQ modules remain valid.
                num_states, processed_dfa, dfa_model, nfa_model, model_gen, var, input_dict = dfa_update(
                    trace, num_states,
                    dfa_model,
                    nfa_model,
                    model_gen, var,
                    input_dict,
                    hyperparams,
                    start_time,
                    synth_iter_num)

                # Extract all DFA states from the transition list
                dfa_states = list(set([dfa_transitions[0] for dfa_transitions in processed_dfa] +
                                      [dfa_transitions[2] for dfa_transitions in processed_dfa]))
                synth_iter_num = synth_iter_num + 1
                set_of_episode_traces = [episode_trace]

            # ===================== LAZY MODULE CREATION =====================
            # If the DFA grew (new states discovered), create fresh NFQ networks
            # for those states. This is the modular architecture from Fig. 3:
            # each DFA state qi gets its own network Bqi.
            new_dfa_states = list(set(dfa_states) - set(old_dfa_states))
            if new_dfa_states:
                for i in new_dfa_states:
                    NFQ_dict[i] = keras.Sequential([
                        keras.layers.Dense(128, input_dim=3, activation=tf.nn.relu),
                        keras.layers.Dense(128, activation=tf.nn.relu),
                        keras.layers.Dense(1, activation='sigmoid')])
                    NFQ_dict[i].compile(loss='mean_squared_error',
                                        metrics=['mean_squared_error'],
                                        optimizer='Adam')

            # Query the DFA: given the full episode trace so far,
            # what DFA state should we be in?
            next_automaton_state = get_next_state(episode_trace, input_dict['event_uniq'], processed_dfa)
        else:
            # Neutral cell — DFA state doesn't change
            next_automaton_state = current_state[-1]

        # Terminate episode if agent entered an unsafe zone
        if episode_trace[-1] == layout_dict['unsafe']:
            break

        # ===================== STORE TRANSITION =====================
        # Build the full product-MDP transition tuple:
        #   [x, y, dfa_state, action, x', y', dfa_state', reward]
        #    0  1      2        3     4   5       6          7
        #
        # Stored in sar_dict keyed by the CURRENT dfa_state.
        # This is the "projection of E onto qi" from the paper —
        # each module only trains on transitions that occurred in its DFA state.
        next_state_2d.append(next_automaton_state)
        sar = current_state + [action] + next_state_2d + [reward(episode_trace, layout_dict)]
        sar_dict[current_state[-1]].append(sar)
        current_state = next_state_2d
        set_of_episode_traces.append(episode_trace)

# ============================================================================
# PHASE 2: DATA PREPARATION
# ============================================================================
# Convert the collected transitions for DFA state 1 into a numpy array
# and deduplicate. In the full pipeline, you'd do this for ALL DFA states.
# (This simplified version only trains one module.)
sar_1 = np.array(sar_dict[1])
sar_1 = np.unique(sar_1, axis=0)
models = [MAIN_NFQ]
history = ddict(list)
sars = [sar_1]

# ===================== EXPERIENCE REFINEMENT =====================
# The replay buffer is heavily imbalanced: most transitions have reward ~0,
# very few have reward ~10. This block oversamples the high-reward transitions
# to ensure the network actually sees the sparse reward signal.
#
# Steps:
# 1. Find indices where reward > 9 (successful task completions)
# 2. Downsample the full buffer to ~10 low-reward samples
# 3. Append ALL high-reward samples back in
# This creates a small, reward-enriched training set.
exp_size = 30
reward_column = sar_1[:, 7]
indx_1 = np.where([reward_column > 9])
high_reward_sar_1 = sar_1[indx_1[1]]
sar_1 = np.delete(sar_1, random.sample(range(0, len(sar_1)), len(sar_1) - 10 + len(indx_1[1])), axis=0)
sar_1 = np.vstack((sar_1, high_reward_sar_1))

# ============================================================================
# PHASE 3: NFQ TRAINING (Step 3 from Fig. 1 — "Deep" box)
# ============================================================================
# This implements Algorithm 1 from the paper's Appendix.
#
# NFQ is an OFFLINE, BATCH RL method (not online like standard DQN):
# 1. Collect a fixed dataset of transitions during exploration (Phase 1)
# 2. Repeatedly recompute targets and refit the network on the SAME data
#
# This is different from DQN which interleaves data collection and training.
# The advantage: NFQ is more sample-efficient for small datasets.
# The disadvantage: no new data is collected during training.

# ===================== INITIALIZATION FIT =====================
# Warm-start: fit each module directly on the observed rewards.
# Input: (x, y, action) — columns [0:2] are (x,y), column [3] is action
# Target: reward — column [7]
# This gives the network a rough initial estimate before the full
# fitted Q-iteration loop refines it with bootstrapped targets.
for i in range(len(models)):
    models[i].fit(np.hstack((sars[i][:, 0:2], sars[i][:, 3:4])), sars[i][:, 7:8], epochs=3, verbose=0)

# ===================== MAIN FITTED Q-ITERATION LOOP =====================
# This is the core training loop. For each epoch:
#   1. Track utility (max Q-value at the initial state) for convergence monitoring
#   2. For each module (backward from accepting DFA states to initial):
#      a. For each transition in the module's buffer:
#         - Compute target = r + gamma * max_a' Q_next_module(s', a')
#         - Note: Q_next_module may be a DIFFERENT module (cross-module backup)
#      b. Fit the module's network on (state, action) -> target
#
# The BACKWARD iteration order (j from len(models)-1 to 0) is critical:
# it ensures that modules closer to the accepting state converge first,
# so their Q-values are stable when earlier modules bootstrap off them.
# This is how extrinsic reward backpropagates through the module chain
# (see Fig. 6a and Fig. 13b in the paper).

init_state = np.array([0, 9, 1])
utility = []
for i in range(episode_number):
    print(int(i / episode_number * 100), '%')

    # ---- Track utility at initial state ----
    # Evaluate Q(init_state, a) for all 4 actions, record the max.
    # This shows whether the agent has learned that the initial state
    # leads to reward (utility should climb toward ~10 * gamma^steps).
    neighs = []
    for l in range(4):
        neigh_inputs = []
        neigh_inputs = np.append([4, 4], l).reshape(1, 3)
        neighs.append(models[0].predict(neigh_inputs))
    utility.append(max(neighs))

    # ---- Backward pass through DFA modules ----
    for j in range(len(models) - 1, -1, -1):
        target = np.zeros(len(sars[j]))

        for k in range(len(sars[j])):
            # For transition k: compute max_a' Q(s', a') using the
            # module corresponding to the NEXT DFA state.
            #
            # sars[j][k, 4:6] = next state (x', y')
            # sars[j][k, 6]   = next DFA state q'
            #
            # CROSS-MODULE BOOTSTRAP:
            # models[min(int(sars[j][k, 6]) - 1, len(models) - 1)]
            # selects the network for DFA state q'. If the agent
            # transitioned to a new DFA state, this is a DIFFERENT
            # network than the one being trained. This is how module
            # Bqi's targets depend on module Bqj's outputs (Section 5,
            # "the output of Bqj directly affects Bqi").
            neigh = []
            for l in range(4):
                # Evaluate Q(s', a') for each of the 4 actions
                # using the NEXT DFA state's module
                neigh_input = []
                neigh_input = np.append(sars[j][k, 4:6], l).reshape(1, 3)
                neigh.append(models[min(int(sars[j][k, 6]) - 1, len(models) - 1)].predict(neigh_input))

            # Standard Bellman target: r + gamma * max_a' Q(s', a')
            target[k] = sars[j][k, 7] + discount_factor * max(neigh)

        # ---- Fit the network on the new targets ----
        # Input: (x, y, action) for all transitions in this module's buffer
        # Target: the bootstrapped Bellman targets computed above
        # This is one "fitted Q-iteration" step — same data, updated targets.
        # After enough epochs, the targets stabilize and Q converges.
        history_j = models[j].fit(np.hstack((sars[j][:, 0:2], sars[j][:, 3:4])),
                                  target.reshape(len(np.hstack((sars[j][:, 0:2], sars[j][:, 3:4]))), 1),
                                  epochs=3,
                                  verbose=0)
        history[j].append(history_j)

# ============================================================================
# PHASE 4: SAVE RESULTS
# ============================================================================
file_path = os.path.dirname(os.path.abspath(__file__))
history_path = os.path.join(file_path, 'history')
if not os.path.exists(history_path):
    os.mkdir(history_path)
pkl.dump(sars, open(os.path.join(history_path, 'sars.p'), 'wb'))
pkl.dump(utility, open(os.path.join(history_path, 'utility.p'), 'wb'))
for i in range(len(models) - 1, -1, -1):
    models[i].save(os.path.join(history_path, 'model_' + str(i + 1) + '.h5'))
    for j in range(episode_number):
        pkl.dump(history[i][j].history, open(os.path.join(history_path, 'history_' + str(i + 1) + '_' + str(j + 1) + '.p'), "wb"))

# Plot utility curve — should show Q-value at initial state increasing
# over training epochs as the reward signal backpropagates.
plt.plot(np.vstack(utility).tolist())
plt.show()