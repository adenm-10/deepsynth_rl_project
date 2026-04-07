"""
Standalone DFA verification for the DeepSynth pipeline.

Run from the project root:
    python verify_dfa.py

Tests:
  1. Feed known-good traces through dfa_init → dfa_update
  2. Check determinism (no duplicate source+symbol pairs)
  3. Check get_next_state accepts good traces, rejects bad ones
  4. Print human-readable transition table
"""

import sys
import time
import numpy as np
from collections import defaultdict
from termcolor import colored

from synth.synth_wrapper import dfa_init, dfa_update, get_next_state

# Object IDs matching mine_craft.py
NEUTRAL = 1
WOOD = 2
GRASS = 3
IRON = 4
GOLD = 5
WORKBENCH = 6
TOOLSHED = 7

OBJ_NAMES = {
    0: 'start_id', 1: 'neutral', 2: 'wood', 3: 'grass',
    4: 'iron', 5: 'gold', 6: 'workbench', 7: 'toolshed',
}

# ─── Trace sets ───────────────────────────────────────────────

# Task 1 ground truth: iron → wood → toolshed
GOOD_TRACES_TASK1 = [
    ['start', IRON, WOOD, TOOLSHED],                          # minimal
    ['start', GRASS, IRON, WOOD, TOOLSHED],                   # distractor before iron
    ['start', IRON, GRASS, WOOD, TOOLSHED],                   # distractor between iron and wood
    ['start', IRON, WOOD, GRASS, TOOLSHED],                   # distractor between wood and toolshed
    ['start', GOLD, IRON, GRASS, WOOD, TOOLSHED],             # multiple distractors
    ['start', IRON, WOOD, IRON, WOOD, TOOLSHED],              # repeated subgoals
]

BAD_TRACES_TASK1 = [
    ['start', WOOD, IRON, TOOLSHED],                          # wrong order
    ['start', IRON, TOOLSHED],                                # missing wood
    ['start', WOOD, TOOLSHED],                                # missing iron
    ['start', IRON, WOOD],                                    # missing toolshed
    ['start', GRASS, TOOLSHED],                               # task 2 trace
]


def build_concat_trace(traces):
    """Concatenate traces the way dqn_online_learner does before calling dfa_update."""
    concat = []
    for t in traces:
        concat.extend(t)
    concat.append('start')  # trailing start as in training code
    return concat


def check_determinism(processed_dfa, event_uniq):
    """Check no (source, symbol) pair maps to multiple destinations."""
    seen = {}
    violations = []
    for trans in processed_dfa:
        key = (trans[0], trans[1])  # (symbol, source) — note: processed_dfa format is [symbol, source, dest]
        # Actually let's verify the format first
        pass

    # processed_dfa transitions are [symbol, source, dest] based on get_next_state usage:
    #   get_next_state does: [x[2] for x in dfa_model if x[0] == state and x[1] == e_id]
    # So format is [source_state, symbol, dest_state]
    # But the raw print from training showed [np.int64(1), np.int64(1), np.int64(2)]
    # and process_dfa remaps states. Let's just check based on how get_next_state indexes:
    #   x[0] = source, x[1] = symbol, x[2] = dest

    seen = {}
    violations = []
    for trans in processed_dfa:
        src, sym, dst = trans[0], trans[1], trans[2]
        key = (src, sym)
        if key in seen and seen[key] != dst:
            violations.append((key, seen[key], dst))
        seen[key] = dst

    return violations


def trace_dfa_states(trace, event_uniq, processed_dfa):
    """Walk a trace through the DFA, returning the state at each step."""
    states = []
    try:
        temp = [event_uniq.index(x) + 1 for x in trace]
    except ValueError as e:
        return None, f"Symbol not in event_uniq: {e}"

    state = 0  # start state in processed_dfa
    for e_id in temp:
        if e_id == 1:
            state = 0
        next_states = [x[2] for x in processed_dfa if x[0] == state and x[1] == e_id]
        if len(next_states) > 1:
            return None, f"NFA detected at state={state}, symbol={e_id}: dests={next_states}"
        if not next_states:
            states.append((e_id, state, '???'))
            return states, f"Dead end at state={state}, symbol={e_id} ({event_uniq[e_id-1]})"
        state = next_states[0]
        states.append((e_id, state))

    return states, None


def print_transition_table(processed_dfa, event_uniq):
    """Print a readable transition table."""
    # Collect all states
    all_states = set()
    for t in processed_dfa:
        all_states.add(t[0])
        all_states.add(t[2])
    all_states = sorted(all_states)

    # Collect all symbols
    all_symbols = sorted(set(t[1] for t in processed_dfa))

    # Build lookup
    lookup = {}
    for t in processed_dfa:
        lookup[(t[0], t[1])] = t[2]

    # Header
    sym_names = []
    for s in all_symbols:
        if 1 <= s <= len(event_uniq):
            sym_names.append(f"{event_uniq[s-1]}({s})")
        else:
            sym_names.append(str(s))

    header = f"{'State':>8} | " + " | ".join(f"{n:>12}" for n in sym_names)
    print(header)
    print("-" * len(header))

    for state in all_states:
        row = f"{state:>8} | "
        cells = []
        for sym in all_symbols:
            dst = lookup.get((state, sym), '-')
            cells.append(f"{dst:>12}")
        row += " | ".join(cells)
        print(row)


def run_verification(good_traces, bad_traces, label="Task 1"):
    print(f"\n{'='*60}")
    print(f"  DFA Verification: {label}")
    print(f"{'='*60}\n")

    # ── Step 1: Initialize and synthesize ──
    print("[1] Initializing DFA synthesis...")
    num_states, var, input_dict, hyperparams = dfa_init()
    num_states = max(num_states, 5)

    print(f"[2] Feeding {len(good_traces)} good traces to synthesizer...")
    concat = build_concat_trace(good_traces)
    print(f"    Concatenated trace length: {len(concat)}")
    print(f"    Trace: {concat[:50]}{'...' if len(concat) > 50 else ''}")

    start_time = time.time()
    (num_states, processed_dfa, dfa_model, nfa_model,
     model_gen, var, input_dict) = dfa_update(
        concat, num_states, [], [], [], var, input_dict,
        hyperparams, start_time, 0)

    elapsed = time.time() - start_time
    print(f"    Synthesis took {elapsed:.2f}s")
    print(f"    num_states={num_states}, transitions={len(processed_dfa)}")

    # ── Step 2: Print raw DFA ──
    print(f"\n[3] Raw processed_dfa (source, symbol, dest):")
    for t in processed_dfa:
        src, sym_id, dst = t
        sym_name = '?'
        if 1 <= sym_id <= len(input_dict['event_uniq']):
            sym_name = str(input_dict['event_uniq'][sym_id - 1])
        print(f"    state {src} --{sym_name}({sym_id})--> state {dst}")

    # ── Step 3: Transition table ──
    print(f"\n[4] Transition table:")
    print_transition_table(processed_dfa, input_dict['event_uniq'])

    # ── Step 4: Determinism check ──
    print(f"\n[5] Determinism check...")
    violations = check_determinism(processed_dfa, input_dict['event_uniq'])
    if violations:
        for v in violations:
            print(colored(f"    FAIL: (state={v[0][0]}, sym={v[0][1]}) -> {v[1]} AND {v[2]}", 'red'))
    else:
        print(colored("    PASS: DFA is deterministic", 'green'))

    # ── Step 5: Test good traces ──
    print(f"\n[6] Testing good traces (should all be accepted):")
    for trace in good_traces:
        result = get_next_state(trace, input_dict['event_uniq'], processed_dfa)
        status = colored("PASS", 'green') if result not in (-1, -2, None) else colored("FAIL", 'red')
        print(f"    {status}  {trace}  -> final_state={result}")

        # Also show state-by-state walk
        states, err = trace_dfa_states(trace, input_dict['event_uniq'], processed_dfa)
        if err:
            print(colored(f"           Walk error: {err}", 'yellow'))
        elif states:
            walk = " -> ".join(f"s{s[1]}" for s in states)
            print(f"           Walk: {walk}")

    # ── Step 6: Test bad traces ──
    print(f"\n[7] Testing bad traces (behavior may vary — check if ordering matters):")
    for trace in bad_traces:
        result = get_next_state(trace, input_dict['event_uniq'], processed_dfa)
        # For bad traces we want to see that the final state differs from good traces
        # or that the DFA rejects/gets stuck
        print(f"    {trace}  -> final_state={result}")
        states, err = trace_dfa_states(trace, input_dict['event_uniq'], processed_dfa)
        if err:
            print(colored(f"           Walk error: {err}", 'yellow'))
        elif states:
            walk = " -> ".join(f"s{s[1]}" for s in states)
            print(f"           Walk: {walk}")

    # ── Step 7: Check if good vs bad reach different final states ──
    print(f"\n[8] Summary:")
    good_finals = set()
    for trace in good_traces:
        r = get_next_state(trace, input_dict['event_uniq'], processed_dfa)
        if r not in (-1, -2, None):
            good_finals.add(r)
    bad_finals = set()
    for trace in bad_traces:
        r = get_next_state(trace, input_dict['event_uniq'], processed_dfa)
        if r not in (-1, -2, None):
            bad_finals.add(r)

    print(f"    Good trace final states: {good_finals}")
    print(f"    Bad trace final states:  {bad_finals}")
    overlap = good_finals & bad_finals
    if overlap:
        print(colored(f"    WARNING: Overlap in final states: {overlap}", 'yellow'))
        print(f"    DFA may not distinguish good from bad traces!")
    elif good_finals:
        print(colored(f"    Good and bad traces reach different states — looks correct!", 'green'))

    return processed_dfa, input_dict


if __name__ == '__main__':
    run_verification(GOOD_TRACES_TASK1, BAD_TRACES_TASK1, label="Task 1")