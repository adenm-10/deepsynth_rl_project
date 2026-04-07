"""
Experiment utilities — plotting functions for Hydra-based training.
Config loading and directory setup are handled by Hydra.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf


# ──────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────

_PLOT_DPI = 150
_ROLLING_WINDOW_EP = 20
_ROLLING_WINDOW_LOSS = 100


def _cfg_get(cfg, key, default=None):
    """Safely read from either OmegaConf DictConfig or plain dict."""
    if OmegaConf.is_config(cfg):
        return OmegaConf.select(cfg, key, default=default)
    return cfg.get(key, default)


def _draw_dfa_markers(ax, dfa_change_log):
    if not dfa_change_log:
        return
    labeled = set()
    for f, nm, nt, accepted in dfa_change_log:
        color = 'green' if accepted else 'red'
        ls = '-' if accepted else ':'
        key = 'accepted' if accepted else 'rejected'
        label = f'DFA {key}' if key not in labeled else None
        ax.axvline(x=f, color=color, linestyle=ls, alpha=0.5,
                   linewidth=1.5, label=label)
        labeled.add(key)


def _rolling(data, window):
    return [np.mean(data[max(0, i - window + 1):i + 1])
            for i in range(len(data))]


# ── Individual plot functions ─────────────────────────────────

def _episode_to_frames(episode_lengths):
    """Convert episode index to cumulative frame count."""
    return np.cumsum(episode_lengths)


def plot_train_rewards(ax, rewards, cfg, episode_lengths=None):
    if episode_lengths is not None and len(episode_lengths) == len(rewards):
        x = _episode_to_frames(episode_lengths)
    else:
        x = np.arange(len(rewards))
    ax.plot(x, rewards, alpha=0.3, label='per episode')
    if len(rewards) >= _ROLLING_WINDOW_EP:
        ax.plot(x, _rolling(rewards, _ROLLING_WINDOW_EP),
                color='orange', label=f'rolling mean ({_ROLLING_WINDOW_EP})')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Reward')
    ax.set_title('Training Episode Rewards')
    ax.legend()


def plot_train_length(ax, episode_lengths, cfg):
    max_it = _cfg_get(cfg, 'algo.max_it_number',
                       _cfg_get(cfg, 'max_it_number', 500))
    x = _episode_to_frames(episode_lengths)
    ax.plot(x, episode_lengths, alpha=0.3, label='per episode')
    if len(episode_lengths) >= _ROLLING_WINDOW_EP:
        ax.plot(x, _rolling(episode_lengths, _ROLLING_WINDOW_EP),
                color='orange', label=f'rolling mean ({_ROLLING_WINDOW_EP})')
    ax.axhline(y=max_it, color='red', linestyle='--', alpha=0.5,
               label=f'max ({max_it})')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Steps')
    ax.set_title('Training Episode Length')
    ax.legend()


def plot_train_loss(ax, loss_list, cfg):
    if not loss_list:
        ax.set_title('Training Loss (no data)')
        return
    update_freq = _cfg_get(cfg, 'algo.update_freq',
                            _cfg_get(cfg, 'update_freq', 2))
    x = np.arange(len(loss_list)) * update_freq
    ax.plot(x, loss_list, alpha=0.3, label='per step')
    if len(loss_list) >= _ROLLING_WINDOW_LOSS:
        ax.plot(x, _rolling(loss_list, _ROLLING_WINDOW_LOSS),
                color='orange', label=f'rolling mean ({_ROLLING_WINDOW_LOSS})')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()


def plot_eval_reward(ax, eval_history, dfa_change_log, cfg):
    if not eval_history:
        ax.set_title('Eval Reward (no data)')
        return
    frames = [e[0] for e in eval_history]
    means = [e[1]['mean_reward'] for e in eval_history]
    stds = [e[1]['std_reward'] for e in eval_history]
    ax.plot(frames, means, 'o-', color='tab:blue', label='mean reward')
    ax.fill_between(frames,
                    np.array(means) - np.array(stds),
                    np.array(means) + np.array(stds),
                    alpha=0.2, color='tab:blue', label='±1 std')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Reward')
    ax.set_title('Eval Reward vs Frame')
    ax.legend(fontsize=7)


def plot_eval_completion(ax, eval_history, dfa_change_log, cfg):
    if not eval_history:
        ax.set_title('Eval Completion (no data)')
        return
    frames = [e[0] for e in eval_history]
    comp = [e[1]['completion_rate'] for e in eval_history]
    ax.plot(frames, comp, 's-', color='tab:green')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Completion Rate')
    ax.set_title('Eval Completion Rate vs Frame')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7)


def plot_eval_length(ax, eval_history, dfa_change_log, cfg):
    if not eval_history:
        ax.set_title('Eval Length (no data)')
        return
    max_it = _cfg_get(cfg, 'algo.max_it_number',
                       _cfg_get(cfg, 'max_it_number', 500))
    frames = [e[0] for e in eval_history]
    lens = [e[1]['mean_length'] for e in eval_history]
    ax.plot(frames, lens, '^-', color='tab:purple')
    ax.axhline(y=max_it, color='red', linestyle='--', alpha=0.5,
               label=f'max ({max_it})')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Mean Episode Length')
    ax.set_title('Eval Episode Length vs Frame')
    ax.legend(fontsize=7)


# ── Public API ────────────────────────────────────────────────

def save_plots(rewards, episode_lengths, loss_list,
               eval_history, dfa_change_log, cfg, plots_dir):
    """
    Generate and save all plot variants:
      - 6 individual PNGs
      - group_train.png   (1×3)
      - group_eval.png    (1×3)
      - group_all.png     (2×3)
    """
    os.makedirs(plots_dir, exist_ok=True)

    task = _cfg_get(cfg, 'task.task_num', _cfg_get(cfg, 'task_num', '?'))
    name = _cfg_get(cfg, 'experiment_name', 'unnamed')
    suptitle = f'Task {task} — {name}'

    # ── Individual plots ──
    individual = [
        ('train_rewards',   plot_train_rewards,   [rewards],
         {'episode_lengths': episode_lengths}),
        ('train_length',    plot_train_length,     [episode_lengths], {}),
        ('train_loss',      plot_train_loss,       [loss_list], {}),
        ('eval_reward',     plot_eval_reward,      [eval_history, dfa_change_log], {}),
        ('eval_completion', plot_eval_completion,  [eval_history, dfa_change_log], {}),
        ('eval_length',     plot_eval_length,      [eval_history, dfa_change_log], {}),
    ]
    for fname, plot_fn, args, kwargs in individual:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_fn(ax, *args, cfg, **kwargs)
        fig.suptitle(suptitle, fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f'{fname}.png'), dpi=_PLOT_DPI)
        plt.close(fig)

    # ── Group: training (1×3) ──
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(f'{suptitle} — Training', fontsize=14)
    plot_train_rewards(axes[0], rewards, cfg, episode_lengths=episode_lengths)
    plot_train_length(axes[1], episode_lengths, cfg)
    plot_train_loss(axes[2], loss_list, cfg)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'group_train.png'), dpi=_PLOT_DPI)
    plt.close(fig)

    # ── Group: eval (1×3) ──
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(f'{suptitle} — Evaluation', fontsize=14)
    plot_eval_reward(axes[0], eval_history, dfa_change_log, cfg)
    plot_eval_completion(axes[1], eval_history, dfa_change_log, cfg)
    plot_eval_length(axes[2], eval_history, dfa_change_log, cfg)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'group_eval.png'), dpi=_PLOT_DPI)
    plt.close(fig)

    # ── Group: all (2×3) ──
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle(suptitle, fontsize=14)
    plot_train_rewards(axes[0, 0], rewards, cfg, episode_lengths=episode_lengths)
    plot_train_length(axes[0, 1], episode_lengths, cfg)
    plot_train_loss(axes[0, 2], loss_list, cfg)
    plot_eval_reward(axes[1, 0], eval_history, dfa_change_log, cfg)
    plot_eval_completion(axes[1, 1], eval_history, dfa_change_log, cfg)
    plot_eval_length(axes[1, 2], eval_history, dfa_change_log, cfg)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'group_all.png'), dpi=_PLOT_DPI)
    plt.close(fig)

    print(f"[PLOTS] Saved 9 figures to {plots_dir}")