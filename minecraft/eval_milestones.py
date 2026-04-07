"""
Milestone-aware evaluation for modular DQN + DFA online learner.

Drop-in replacements / additions for dqn_online_learner.py:
  1. TASK_MILESTONES dict
  2. evaluate() — now returns milestone_hist and object_visit_rates
  3. plot_milestones() — call after training to generate diagnostic plots

Integration:
  - Replace the existing evaluate() with this one
  - Add TASK_MILESTONES near the other hyperparams
  - Call plot_milestones(eval_history, history_path) alongside existing plot code
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ──────────────────────────────────────────────────────────────
# Milestone definitions (ordered object sequences per task)
# ──────────────────────────────────────────────────────────────
# Object IDs:  wood=2, grass=3, iron=4, gold=5, workbench=6, toolshed=7
TASK_MILESTONES = {
    1: [4, 2, 7],        # iron → wood → toolshed
    2: [3, 7],            # grass → toolshed
    3: [2, 3, 4, 7],     # wood → grass → iron → toolshed
    4: [2, 6],            # wood → workbench
    5: [3, 6],            # grass → workbench  (simplified; task 5 has OR on first step)
    6: [4, 2, 6],         # iron → wood → workbench
    9: [2],               # just wood
}

# All non-neutral, non-obstacle object IDs worth tracking
ALL_OBJECTS = {2: 'wood', 3: 'grass', 4: 'iron', 5: 'gold',
               6: 'workbench', 7: 'toolshed'}


# ──────────────────────────────────────────────────────────────
# Evaluate with milestone + visitation tracking
# ──────────────────────────────────────────────────────────────

def evaluate(env, DQN_dict, processed_dfa, input_dict, task_num,
             n_episodes=5, max_steps=500):
    """
    Greedy eval with two added diagnostics:
      - milestone_hist: list of length (num_milestones + 1), counting
        how many episodes reached each depth (0 = none, len = completed)
      - object_visit_rates: dict {obj_id: fraction of episodes that visited it}
    """
    from synth.synth_wrapper import get_next_state   # keep import local

    milestones = TASK_MILESTONES.get(task_num, [])
    n_milestones = len(milestones)

    # Per-episode accumulators
    episode_rewards = []
    episode_lengths = []
    completions = 0
    milestone_depths = []          # int per episode: how far through the milestone chain
    object_visits_per_ep = []      # set of obj IDs visited per episode

    has_dfa = len(processed_dfa) > 0
    neutral, obstacles = env.neutral, env.obstacles

    for _ in range(n_episodes):
        current_pos = list(env.initialiser())
        current_layout = env.layout(env._world).copy()
        episode_trace = ['start']
        episode_detected_objects = []
        current_dfa_state = 1
        ep_reward = 0.0
        terminal = False
        steps = 0

        # Milestone pointer
        ms_ptr = 0
        visited_objects = set()

        while not terminal and steps < max_steps:
            s = np.array([current_pos[0] / 9.0, current_pos[1] / 9.0],
                         dtype=np.float32)

            active_module = current_dfa_state if current_dfa_state in DQN_dict else 1
            q_vals = DQN_dict[active_module].q_net(
                s.reshape(1, -1), training=False).numpy()[0]
            action = int(np.argmax(q_vals))

            next_pos = list(env.take_action(current_pos, action))
            next_label = current_layout[next_pos[0]][next_pos[1]]
            steps += 1

            # Track objects
            old_obj_set = episode_detected_objects.copy()
            if next_label not in (neutral, obstacles):
                episode_detected_objects.append(int(next_label))
                episode_trace.append(int(next_label))
                visited_objects.add(int(next_label))

                # Advance milestone pointer if this object matches the next milestone
                if ms_ptr < n_milestones and int(next_label) == milestones[ms_ptr]:
                    ms_ptr += 1

            new_obj_set = list(set(episode_detected_objects))

            env_reward = env.reward(task_num, np.array(episode_trace[1:]))
            mu = 1.0  # match training MU
            reward = env_reward + mu * (1 if set(new_obj_set) - set(old_obj_set) else 0)
            episode_detected_objects = new_obj_set
            ep_reward += reward

            terminal = (env_reward > 9) or (steps >= max_steps)
            if env_reward > 9:
                completions += 1

            # DFA state tracking
            if has_dfa:
                nds = get_next_state(
                    episode_trace, input_dict['event_uniq'], processed_dfa)
                if nds is not None and nds not in (-1, -2) and nds in DQN_dict:
                    current_dfa_state = nds

            # Vanishing objects
            if env._vanishing == 1 and \
                    current_layout[current_pos[0]][current_pos[1]] not in (env.workbench, env.toolshed):
                current_layout[current_pos[0]][current_pos[1]] = neutral

            current_pos = next_pos

        episode_rewards.append(ep_reward)
        episode_lengths.append(steps)
        milestone_depths.append(ms_ptr)
        object_visits_per_ep.append(visited_objects)

    # ── Aggregate milestone histogram ──
    # milestone_hist[k] = number of episodes that reached exactly depth k
    milestone_hist = [0] * (n_milestones + 1)
    for d in milestone_depths:
        milestone_hist[d] += 1

    # ── Aggregate object visitation rates ──
    object_visit_rates = {}
    for obj_id in ALL_OBJECTS:
        count = sum(1 for v in object_visits_per_ep if obj_id in v)
        object_visit_rates[obj_id] = count / n_episodes

    return {
        'mean_reward':        float(np.mean(episode_rewards)),
        'std_reward':         float(np.std(episode_rewards)),
        'mean_length':        float(np.mean(episode_lengths)),
        'completion_rate':    completions / n_episodes,
        'per_episode_rewards': episode_rewards,
        'per_episode_lengths': episode_lengths,
        # New fields
        'milestone_hist':     milestone_hist,
        'milestone_depths':   milestone_depths,
        'object_visit_rates': object_visit_rates,
    }


# ──────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────

def plot_milestones(eval_history, task_num, save_dir=None):
    """
    Two diagnostic plots from eval_history (list of (frame, result_dict) tuples).

    Plot 1: Stacked bar — milestone depth distribution over eval checkpoints
    Plot 2: Line plot  — per-object visitation rates over time
    """
    if not eval_history:
        print("No eval history to plot.")
        return

    milestones = TASK_MILESTONES.get(task_num, [])
    n_milestones = len(milestones)

    frames = [e[0] for e in eval_history]
    hists = [e[1]['milestone_hist'] for e in eval_history]
    visit_rates = [e[1]['object_visit_rates'] for e in eval_history]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # ── Plot 1: Stacked bar of milestone depths ──
    ax = axes[0]
    # Build label for each depth
    depth_labels = ['depth 0 (none)']
    for i, obj_id in enumerate(milestones):
        name = ALL_OBJECTS.get(obj_id, str(obj_id))
        depth_labels.append(f'depth {i+1} ({name})')

    hists_arr = np.array(hists, dtype=float)  # (n_evals, n_milestones+1)
    n_evals = len(frames)
    bottoms = np.zeros(n_evals)
    x = np.arange(n_evals)
    bar_width = 0.8

    colors = plt.cm.viridis(np.linspace(0.15, 0.95, n_milestones + 1))
    for depth in range(n_milestones + 1):
        vals = hists_arr[:, depth]
        ax.bar(x, vals, bar_width, bottom=bottoms, label=depth_labels[depth],
               color=colors[depth])
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels([str(f) for f in frames], rotation=45, ha='right', fontsize=7)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Episode count')
    ax.set_title(f'Milestone Depth Distribution (Task {task_num})')
    ax.legend(loc='upper left', fontsize=7)

    # ── Plot 2: Object visitation rates ──
    ax2 = axes[1]
    for obj_id, name in ALL_OBJECTS.items():
        rates = [vr.get(obj_id, 0.0) for vr in visit_rates]
        # Bold the objects that are part of the task milestones
        lw = 2.5 if obj_id in milestones else 1.0
        alpha = 1.0 if obj_id in milestones else 0.4
        ax2.plot(frames, rates, 'o-', label=name, linewidth=lw, alpha=alpha, markersize=4)

    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Fraction of eval episodes')
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title(f'Object Visitation Rates (Task {task_num})')
    ax2.legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'milestone_diagnostics.png'), dpi=150)
    plt.show()