"""
compare_methods.py
──────────────────
Compares Q-value learning curves: DeepSynth (modular NFQ) vs all
combinations of DQN + {Double, Dueling, Automaton}.

Outputs:
  - Individual utility plots per task to output/
  - A collage grid: rows=tasks, cols=methods

Usage:
  python compare_methods.py --tasks 1 3 --seeds 3 --epochs 100
"""

import argparse
import sys
import os
import random
import time
import itertools
import numpy as np
import pickle as pkl
import tensorflow as tf
import matplotlib.pyplot as plt

# ── Parse args before synth imports (synth has its own argparse) ──
_parser = argparse.ArgumentParser()
_parser.add_argument("--tasks", nargs="+", type=int, default=[1, 3])
_parser.add_argument("--seeds", type=int, default=3)
_parser.add_argument("--epochs", type=int, default=100)
_parser.add_argument("--output", type=str, default="output")
_ARGS, _remaining = _parser.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining

from minecraft.mine_craft import MineCraft
from dqn import DQN, Transition

import learner
from learner import run_exploration_nfq, refine_sars_nfq, train_nfq

# ═══════════════════════════════════════════════════════
# Paper hyperparameters
# ═══════════════════════════════════════════════════════
DISCOUNT   = 0.95
HIDDEN_DIM = 128
LR         = 1e-3
FIT_PASSES = 3
BATCH_SIZE = 64

# All 8 DQN flag combinations
DQN_CONFIGS = list(itertools.product([False, True], repeat=3))
# Each element is (double, dueling, automaton)


def config_label(double, dueling, automaton):
    parts = []
    if double:   parts.append("Dbl")
    if dueling:  parts.append("Duel")
    if automaton: parts.append("Aut")
    if not parts:
        return "DQN (vanilla)"
    return f"DQN ({'+'.join(parts)})"


# ═══════════════════════════════════════════════════════
# DQN offline training — returns utility curve only
# ═══════════════════════════════════════════════════════

def train_dqn_offline(sars, num_epochs, double, dueling, automaton):
    state_dim = 3 if automaton else 2

    all_transitions = []
    for s_arr in sars:
        for row in s_arr:
            if automaton:
                s  = row[0:3].astype(np.float32)
                sp = row[4:7].astype(np.float32)
            else:
                s  = row[0:2].astype(np.float32)
                sp = row[4:6].astype(np.float32)
            all_transitions.append(Transition(
                s=s, a=int(row[3]), r=float(row[7]),
                s_prime=sp, done=row[7] > 9.0))

    dqn = DQN(
        gamma=DISCOUNT, eps=0.1, tau=0.0, lr=LR,
        double_dqn=double, dueling_dqn=dueling,
        state_dim=state_dim, action_dim=4, hidden_dim=HIDDEN_DIM,
    )

    if automaton:
        init_s = np.array([[4.0, 4.0, 1.0]], dtype=np.float32)
    else:
        init_s = np.array([[4.0, 4.0]], dtype=np.float32)

    utility = []
    for epoch in range(num_epochs):
        qv = dqn.q_net(init_s, training=False)
        utility.append(float(tf.reduce_max(qv).numpy()))

        for _ in range(FIT_PASSES):
            idxs = list(range(len(all_transitions)))
            random.shuffle(idxs)
            for start in range(0, len(all_transitions), BATCH_SIZE):
                end = min(start + BATCH_SIZE, len(all_transitions))
                batch = [all_transitions[idxs[j]] for j in range(start, end)]
                dqn.update(batch)

        if double:
            dqn.target_net.set_weights(dqn.q_net.get_weights())

        if (epoch + 1) % 10 == 0:
            label = config_label(double, dueling, automaton)
            print(f"      [{label}] epoch {epoch+1}: Q={utility[-1]:.4f}")

    return utility


# ═══════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════

COLORS = {
    "DeepSynth (NFQ)":      "black",
    "DQN (vanilla)":        "tab:gray",
    "DQN (Dbl)":            "tab:blue",
    "DQN (Duel)":           "tab:cyan",
    "DQN (Aut)":            "tab:orange",
    "DQN (Dbl+Duel)":       "tab:purple",
    "DQN (Dbl+Aut)":        "tab:red",
    "DQN (Duel+Aut)":       "tab:olive",
    "DQN (Dbl+Duel+Aut)":   "tab:green",
}


def plot_single_task(task_num, all_utilities, num_epochs, save_dir):
    """Individual plot for one task: all method utility curves."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(1, num_epochs + 1)

    for name, runs in all_utilities.items():
        max_len = max(len(r) for r in runs)
        mat = np.full((len(runs), max_len), np.nan)
        for i, r in enumerate(runs):
            mat[i, :len(r)] = r
        mean = np.nanmean(mat, axis=0)
        std  = np.nanstd(mat, axis=0)
        lw = 2.5 if name == "DeepSynth (NFQ)" else 1.2
        ls = "-" if name == "DeepSynth (NFQ)" else "--"
        ax.plot(x[:max_len], mean, label=name, color=COLORS.get(name),
                linewidth=lw, linestyle=ls)
        ax.fill_between(x[:max_len], mean - std, mean + std,
                        alpha=0.1, color=COLORS.get(name))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("max Q at s₀")
    ax.set_title(f"Task {task_num} — Q-Value Learning")
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    plt.tight_layout()

    path = os.path.join(save_dir, f"task{task_num}_utility.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_collage(all_task_results, tasks, num_epochs, save_dir):
    """
    Grid collage: rows = tasks, cols = methods.
    Each cell is the utility curve for that task+method.
    """
    method_names = list(all_task_results[tasks[0]].keys())
    n_rows = len(tasks)
    n_cols = len(method_names)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3 * n_rows),
                             squeeze=False, sharey="row")

    for row, task_num in enumerate(tasks):
        for col, name in enumerate(method_names):
            ax = axes[row][col]
            runs = all_task_results[task_num][name]
            max_len = max(len(r) for r in runs)
            mat = np.full((len(runs), max_len), np.nan)
            for i, r in enumerate(runs):
                mat[i, :len(r)] = r
            mean = np.nanmean(mat, axis=0)
            std  = np.nanstd(mat, axis=0)
            x = np.arange(1, max_len + 1)

            ax.plot(x, mean, color=COLORS.get(name, "tab:gray"), linewidth=1.5)
            ax.fill_between(x, mean - std, mean + std,
                            alpha=0.15, color=COLORS.get(name, "tab:gray"))

            if row == 0:
                ax.set_title(name, fontsize=8, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"Task {task_num}\nmax Q", fontsize=9)
            if row == n_rows - 1:
                ax.set_xlabel("Epoch", fontsize=8)

            ax.tick_params(labelsize=7)

    plt.suptitle("Q-Value Learning: DeepSynth NFQ vs DQN Ablations",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()

    path = os.path.join(save_dir, "collage.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved collage to {path}")


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

def main():
    args = _ARGS
    num_epochs = args.epochs
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
    os.makedirs(save_dir, exist_ok=True)

    method_names = ["DeepSynth (NFQ)"] + [config_label(*c) for c in DQN_CONFIGS]
    print(f"Methods: {method_names}")
    print(f"Tasks: {args.tasks}  Seeds: {args.seeds}  Epochs: {num_epochs}\n")

    # task -> {method_name -> [utility_curve_per_seed]}
    all_task_results = {}

    for task_num in args.tasks:
        print(f"{'#'*60}")
        print(f"#  TASK {task_num}")
        print(f"{'#'*60}")

        learner.task_num = task_num
        task_results = {name: [] for name in method_names}

        for seed in range(args.seeds):
            print(f"\n── Seed {seed+1}/{args.seeds} ──")
            random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)

            env = MineCraft()

            # ── Shared exploration ──
            print("  Exploration + DFA synthesis...")
            t0 = time.time()
            sar_dict, NFQ_dict, processed_dfa, input_dict = \
                run_exploration_nfq(env, task_num)
            sars, models, rew = refine_sars_nfq(sar_dict, NFQ_dict)
            n_trans = sum(len(s) for s in sars)
            print(f"  Done in {time.time()-t0:.1f}s  ({n_trans} transitions, "
                  f"{len(sars)} automaton states)")

            if rew is None:
                print("  No positive reward — skipping seed.")
                continue

            # ── DeepSynth NFQ ──
            print(f"\n  Training DeepSynth (NFQ)...")
            t0 = time.time()
            _, utility_nfq, _, _ = train_nfq(sars, models, episode_number=num_epochs)
            print(f"  Done in {time.time()-t0:.1f}s")
            task_results["DeepSynth (NFQ)"].append(utility_nfq)

            # ── All DQN combos ──
            for double, dueling, automaton in DQN_CONFIGS:
                label = config_label(double, dueling, automaton)
                print(f"\n  Training {label}...")
                t0 = time.time()
                utility_dqn = train_dqn_offline(
                    sars, num_epochs, double, dueling, automaton)
                print(f"  Done in {time.time()-t0:.1f}s")
                task_results[label].append(utility_dqn)

        all_task_results[task_num] = task_results

        # Individual task plot
        plot_single_task(task_num, task_results, num_epochs, save_dir)

        # Save raw data
        pkl.dump(task_results,
                 open(os.path.join(save_dir, f"task{task_num}_utilities.p"), "wb"))

    # Collage across all tasks
    if len(args.tasks) > 0:
        plot_collage(all_task_results, args.tasks, num_epochs, save_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()