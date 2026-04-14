# DeepSynth Re-implementation: Online Modular DQN for Automatic Task Segmentation

A re-implementation and extended analysis of the [DeepSynth](https://arxiv.org/pdf/1911.10244.pdf) algorithm (Hasanbeig et al., 2021), adapted for online training in the Minecraft crafting environment. This project was completed as a final project for CS 5180: Reinforcement Learning at Northeastern University.

## Overview

DeepSynth addresses sparse, non-Markovian RL tasks by automatically synthesizing a minimal DFA from exploration traces and using it to decompose the task into Markovian sub-goals, each handled by a dedicated deep RL module. The original paper uses offline Neural Fitted Q-Iteration (NFQ) for its Minecraft experiments.

This project replaces the NFQ modules with **online Dueling Double DQN agents with Prioritized Experience Replay**, enabling evaluation of actual task completion rather than just Q-value convergence. Additional modifications include task-relevant object filtering, a per-step penalty, and intrinsic reward coefficient tuning.

### Key Findings

- Tasks with 1–2 sub-goals converge reliably under the online setup
- Tasks with 3+ sub-goals exhibit compounding instability across the modular architecture — no single design choice is responsible
- Irrelevant objects in the environment pollute traces and cause the SAT synthesizer to produce bloated automata, degrading sample efficiency proportionally

## Original Paper

Hasanbeig, M., Jeppu, N. Y., Abate, A., Melham, T., Kroening, D., "DeepSynth: Automata Synthesis for Automatic Task Segmentation in Deep Reinforcement Learning", AAAI Conference on Artificial Intelligence, 2021. [[PDF]](https://arxiv.org/pdf/1911.10244.pdf)