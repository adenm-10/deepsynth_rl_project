"""
Modular DQN with hardcoded ground-truth DFA.
Usage:  python gt_exp.py --task 1
        python gt_exp.py --task 3 --frames 150000
"""

from minecraft.mine_craft import MineCraft
import tensorflow as tf
import numpy as np
import pickle as pkl
import os
import argparse
from collections import defaultdict as ddict
from tqdm import tqdm
import matplotlib.pyplot as plt

from dqn import DQN, Transition, PrioritizedReplayBuffer

# ── Ground-truth DFA definitions ──────────────────────────────
# Each task maps to a list of (from_state, label, to_state).
# State 1 is always the start. Highest state is accepting.
# Unlisted transitions are self-loops.

IRON, WOOD, GRASS, GOLD, WORKBENCH, TOOLSHED = 4, 2, 3, 5, 6, 7

GT_DFAS = {
    1: {  # iron → wood → toolshed
        'transitions': [(1, IRON, 2), (2, WOOD, 3), (3, TOOLSHED, 4)],
        'states': [1, 2, 3, 4],
        'accepting': 4,
    },
    2: {  # grass → toolshed
        'transitions': [(1, GRASS, 2), (2, TOOLSHED, 3)],
        'states': [1, 2, 3],
        'accepting': 3,
    },
    3: {  # wood → grass → iron → toolshed
        'transitions': [(1, WOOD, 2), (2, GRASS, 3), (3, IRON, 4), (4, TOOLSHED, 5)],
        'states': [1, 2, 3, 4, 5],
        'accepting': 5,
    },
    4: {  # wood → workbench
        'transitions': [(1, WOOD, 2), (2, WORKBENCH, 3)],
        'states': [1, 2, 3],
        'accepting': 3,
    },
    5: {  # grass → workbench  (simplified; real task 5 allows grass OR iron first)
        'transitions': [(1, GRASS, 2), (2, WORKBENCH, 3)],
        'states': [1, 2, 3],
        'accepting': 3,
    },
    6: {  # iron → wood → workbench
        'transitions': [(1, IRON, 2), (2, WOOD, 3), (3, WORKBENCH, 4)],
        'states': [1, 2, 3, 4],
        'accepting': 4,
    },
}


def build_transition_lookup(task_num):
    """Build a {(state, label): next_state} dict for fast lookup."""
    dfa = GT_DFAS[task_num]
    return {(s, l): ns for s, l, ns in dfa['transitions']}


def gt_trace_state(trace, lookup):
    s = 1
    for label in trace:
        if label == 'start':
            s = 1
        else:
            s = lookup.get((s, label), s)
    return s


# ── Config ────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--task', type=int, default=1, choices=GT_DFAS.keys())
    p.add_argument('--frames', type=int, default=100_000)
    p.add_argument('--only_needed', action='store_true', default=True)
    p.add_argument('--all_objects', dest='only_needed', action='store_false')
    return p.parse_args()


DISCOUNT_FACTOR = 0.99
MAX_STEPS = 500
BATCH_SIZE = 64
MEM_SIZE = 25_000
MIN_BUFFER = 300
UPDATE_FREQ = 2
LR = 1e-3
HIDDEN = 32
MU = 0.1
PRIORITY_SCALE = 0.6
TARGET_CLIP_LOW = -5.0
TARGET_CLIP_HIGH = 20.0

EPS_INITIAL = 1.0
EPS_FINAL = 0.1
EPS_FINAL_FRAME = 0.05
EPS_ANNEAL = 50_000

EVAL_FREQ = 5000
EVAL_EPS = 10
GRID = 10


# ── Helpers ───────────────────────────────────────────────────

def norm(xy):
    return np.array([xy[0] / (GRID - 1), xy[1] / (GRID - 1)], dtype=np.float32)


def make_dqn():
    d = DQN(gamma=DISCOUNT_FACTOR, eps=0.0, tau=0.005, lr=LR,
            double_dqn=True, dueling_dqn=True,
            state_dim=2, action_dim=4, hidden_dim=HIDDEN)
    d.hard_update_target()
    return d


def calc_eps(frame, total_frames):
    if frame < MIN_BUFFER:
        return EPS_INITIAL
    if frame < MIN_BUFFER + EPS_ANNEAL:
        return EPS_INITIAL - (EPS_INITIAL - EPS_FINAL) * (frame - MIN_BUFFER) / EPS_ANNEAL
    return EPS_FINAL - (EPS_FINAL - EPS_FINAL_FRAME) * (frame - MIN_BUFFER - EPS_ANNEAL) / max(1, total_frames - MIN_BUFFER - EPS_ANNEAL)


def get_action(dqn, frame, state, total_frames, evaluation=False):
    eps = 0.0 if evaluation else calc_eps(frame, total_frames)
    if np.random.rand() < eps:
        return np.random.randint(0, 4)
    q = dqn.q_net(state.reshape(1, -1), training=False).numpy()[0]
    return int(np.argmax(q))


def learn(dfa_key, dqn_dict, buf_dict, frame, total_frames):
    buf = buf_dict[dfa_key]
    if len(buf) < MIN_BUFFER:
        return 0.0
    batch, indices, weights = buf.sample(min(BATCH_SIZE, len(buf)))
    eps = calc_eps(frame, total_frames)
    weights = weights ** (1 - eps)

    rewards_b = np.array([t.r for t in batch], dtype=np.float32)
    targets = rewards_b.copy()

    groups = ddict(list)
    for k, t in enumerate(batch):
        if not t.done:
            groups[t.next_dfa_state].append(k)

    for nds, idxs in groups.items():
        if nds not in dqn_dict:
            continue
        s_primes = np.array([batch[k].s_prime for k in idxs], dtype=np.float32)
        q_online = dqn_dict[nds].q_net(s_primes, training=False)
        best_a = tf.argmax(q_online, axis=1, output_type=tf.int32)
        q_tgt = dqn_dict[nds].target_net(s_primes, training=False)
        next_q = tf.gather_nd(q_tgt, tf.stack([tf.range(len(idxs)), best_a], 1)).numpy()
        for j, k in enumerate(idxs):
            targets[k] = rewards_b[k] + DISCOUNT_FACTOR * next_q[j]

    targets = np.clip(targets, TARGET_CLIP_LOW, TARGET_CLIP_HIGH)
    loss, td_err = dqn_dict[dfa_key].update_with_targets(batch, targets, weights)
    buf.update_priorities(indices, td_err)
    return float(loss)


def evaluate(env, dqn_dict, task_num, lookup):
    results = []
    completions = 0
    for _ in range(EVAL_EPS):
        pos = list(env.initialiser())
        layout = env.layout(env._world).copy()
        trace = ['start']
        detected = []
        dfa_s = 1
        ep_r = 0.0
        for step in range(MAX_STEPS):
            s = norm(pos)
            module = dfa_s if dfa_s in dqn_dict else 1
            q = dqn_dict[module].q_net(s.reshape(1, -1), training=False).numpy()[0]
            a = int(np.argmax(q))
            npos = list(env.take_action(pos, a))
            lbl = layout[npos[0]][npos[1]]

            old_det = detected.copy()
            if lbl not in (env.neutral, env.obstacles):
                detected.append(int(lbl))
                trace.append(int(lbl))
            new_det = list(set(detected))

            env_r = env.reward(task_num, np.array(trace[1:]))
            ep_r += env_r + MU * (1 if set(new_det) - set(old_det) else 0)
            detected = new_det

            if env_r > 9:
                completions += 1

            dfa_s = gt_trace_state(trace, lookup)

            if env._vanishing == 1 and layout[pos[0]][pos[1]] not in (env.workbench, env.toolshed):
                layout[pos[0]][pos[1]] = env.neutral
            pos = npos
            if env_r > 9:
                break
        results.append((ep_r, step + 1))

    rewards = [r[0] for r in results]
    lengths = [r[1] for r in results]
    return {
        'mean_reward': np.mean(rewards), 'std_reward': np.std(rewards),
        'mean_length': np.mean(lengths), 'completion_rate': completions / EVAL_EPS,
    }


# ── Main ──────────────────────────────────────────────────────

if __name__ == '__main__':
    args = parse_args()
    task_num = args.task
    total_frames = args.frames
    dfa_info = GT_DFAS[task_num]
    lookup = build_transition_lookup(task_num)
    learnable_states = [s for s in dfa_info['states'] if s != dfa_info['accepting']]

    print(f"Task {task_num}: states={dfa_info['states']}, "
          f"accepting={dfa_info['accepting']}, "
          f"learnable={learnable_states}, "
          f"only_needed={args.only_needed}")

    env = MineCraft(task_num=task_num, only_needed_objects=args.only_needed)

    dqn_dict = {s: make_dqn() for s in dfa_info['states']}
    buf_dict = {s: PrioritizedReplayBuffer(size=MEM_SIZE, alpha=PRIORITY_SCALE)
                for s in dfa_info['states']}

    frame = 0
    rewards_log, lengths_log, loss_log, eval_log = [], [], [], []
    next_eval = EVAL_FREQ
    pbar = tqdm(total=total_frames, desc=f'GT-DFA T{task_num}', unit='frame', dynamic_ncols=True)

    try:
        while frame < total_frames:
            pos = list(env.initialiser())
            layout = env.layout(env._world).copy()
            trace = ['start']
            detected = []
            dfa_s = 1
            ep_r = 0.0

            for step in range(MAX_STEPS):
                s = norm(pos)
                a = get_action(dqn_dict[dfa_s], frame, s, total_frames)
                npos = list(env.take_action(pos, a))
                lbl = layout[npos[0]][npos[1]]
                frame += 1

                old_det = detected.copy()
                if lbl not in (env.neutral, env.obstacles):
                    detected.append(int(lbl))
                    trace.append(int(lbl))
                new_det = list(set(detected))

                env_r = env.reward(task_num, np.array(trace[1:]))
                r = env_r + MU * (1 if set(new_det) - set(old_det) else 0)
                detected = new_det
                ep_r += r
                done = env_r > 9

                new_dfa_s = gt_trace_state(trace, lookup)

                if env._vanishing == 1 and layout[pos[0]][pos[1]] not in (env.workbench, env.toolshed):
                    layout[pos[0]][pos[1]] = env.neutral

                buf_dict[dfa_s].add(Transition(
                    s=s, a=a, r=float(r), s_prime=norm(npos),
                    done=done, next_dfa_state=new_dfa_s,
                ))

                if frame % UPDATE_FREQ == 0:
                    for ag in learnable_states:
                        if len(buf_dict[ag]) > MIN_BUFFER:
                            loss = learn(ag, dqn_dict, buf_dict, frame, total_frames)
                            loss_log.append(loss)
                            dqn_dict[ag].soft_update_target()

                dfa_s = new_dfa_s
                pos = npos
                pbar.update(1)

                if frame >= next_eval:
                    res = evaluate(env, dqn_dict, task_num, lookup)
                    eval_log.append((frame, res))
                    next_eval += EVAL_FREQ
                    tqdm.write(
                        f'[EVAL @ {frame}]  reward={res["mean_reward"]:.2f}±{res["std_reward"]:.2f}  '
                        f'completion={res["completion_rate"]:.0%}  len={res["mean_length"]:.1f}')

                if done:
                    break

            rewards_log.append(ep_r)
            lengths_log.append(step + 1)

            postfix = {'ep': len(rewards_log), 'eps': f'{calc_eps(frame, total_frames):.3f}'}
            if len(rewards_log) >= 10:
                postfix['avg_r'] = f'{np.mean(rewards_log[-10:]):.1f}'
            if loss_log:
                postfix['loss'] = f'{np.mean(loss_log[-100:]):.4f}'
            if eval_log:
                postfix['compl'] = f'{eval_log[-1][1]["completion_rate"]:.0%}'
            pbar.set_postfix(postfix)

    except KeyboardInterrupt:
        tqdm.write('\nStopped early.')
    pbar.close()

    # ── Save + Plot ──
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'history_gt_dfa_t{task_num}')
    os.makedirs(out, exist_ok=True)
    pkl.dump(rewards_log, open(os.path.join(out, 'rewards.p'), 'wb'))
    pkl.dump(eval_log, open(os.path.join(out, 'eval.p'), 'wb'))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f'Task {task_num} — GT DFA', fontsize=14)

    axes[0, 0].plot(rewards_log, alpha=0.3)
    if len(rewards_log) >= 20:
        rm = [np.mean(rewards_log[max(0, i-19):i+1]) for i in range(len(rewards_log))]
        axes[0, 0].plot(rm, color='orange')
    axes[0, 0].set_title('Training Reward')
    axes[0, 0].set_xlabel('Episode')

    axes[0, 1].plot(lengths_log, alpha=0.3)
    if len(lengths_log) >= 20:
        rm = [np.mean(lengths_log[max(0, i-19):i+1]) for i in range(len(lengths_log))]
        axes[0, 1].plot(rm, color='orange')
    axes[0, 1].axhline(MAX_STEPS, color='red', ls='--', alpha=0.5)
    axes[0, 1].set_title('Training Length')
    axes[0, 1].set_xlabel('Episode')

    if eval_log:
        ef = [e[0] for e in eval_log]
        axes[1, 0].plot(ef, [e[1]['mean_reward'] for e in eval_log], 'o-')
        axes[1, 0].set_title('Eval Reward')
        axes[1, 0].set_xlabel('Frame')
        axes[1, 1].plot(ef, [e[1]['completion_rate'] for e in eval_log], 's-', color='green')
        axes[1, 1].set_ylim(-0.05, 1.05)
        axes[1, 1].set_title('Eval Completion Rate')
        axes[1, 1].set_xlabel('Frame')

    plt.tight_layout()
    plt.savefig(os.path.join(out, 'curves.png'), dpi=150)
    plt.show()