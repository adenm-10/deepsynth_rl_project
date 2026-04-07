"""
DFA update guard for dqn_online_learner.py

Walks stored successful traces through both old and new DFAs.
Rejects the new DFA if any trace produces a different state sequence.

Integration:
  1. Add `GUARD_DFA_UPDATE = True` to hyperparams section
  2. Import: `from dfa_guard import guarded_dfa_update`
  3. Replace the dfa_update call block with `guarded_dfa_update(...)` (see example below)
"""

from synth.synth_wrapper import dfa_update, get_next_state
from termcolor import colored


def walk_trace(trace, event_uniq, processed_dfa):
    """
    Walk a trace through a processed DFA, returning the full state sequence.
    Returns None if the trace hits a dead end or symbol not found.
    """
    states = [0]  # start state in processed_dfa
    try:
        temp = [event_uniq.index(x) + 1 for x in trace]
    except ValueError:
        return None

    state = 0
    for e_id in temp:
        if e_id == 1:
            state = 0
            states.append(0)
            continue
        next_states = [x[2] for x in processed_dfa if x[0] == state and x[1] == e_id]
        if len(next_states) != 1:
            return None
        state = next_states[0]
        states.append(state)
    return tuple(states)


def check_dfa_compatible(old_dfa, old_event_uniq, new_dfa, new_event_uniq, traces, verbose=False):
    """
    Check whether old and new DFAs produce identical state sequences
    for all given traces.

    Returns (is_compatible, details_string)
    """
    if not old_dfa or not traces:
        return True, "no old DFA or no traces to check"

    for i, trace in enumerate(traces):
        old_seq = walk_trace(trace, old_event_uniq, old_dfa)
        new_seq = walk_trace(trace, new_event_uniq, new_dfa)

        if old_seq is None:
            # Old DFA couldn't handle this trace — new one can only be better
            continue

        if new_seq is None:
            msg = f"trace {i}: new DFA dead-ends where old DFA didn't"
            if verbose:
                print(colored(f"[DFA GUARD] REJECT: {msg}", 'red'))
            return False, msg

        if old_seq != new_seq:
            msg = (f"trace {i}: state sequence changed\n"
                   f"  old: {old_seq}\n"
                   f"  new: {new_seq}")
            if verbose:
                print(colored(f"[DFA GUARD] REJECT: {msg}", 'red'))
            return False, msg

    return True, "all traces compatible"


def guarded_dfa_update(trace, num_states, dfa_model, nfa_model, model_gen,
                       var, input_dict, hyperparams, start_time, synth_iter_num,
                       successful_traces, guard_enabled=True, verbose=False):
    """
    Drop-in wrapper around dfa_update that optionally rejects destructive updates.

    Args:
        trace ... synth_iter_num: same as dfa_update
        successful_traces: list of successful episode traces (list of lists)
        guard_enabled: if False, behaves identically to dfa_update
        verbose: print guard decisions

    Returns:
        Same tuple as dfa_update:
          (num_states, processed_dfa, dfa_model, nfa_model, model_gen, var, input_dict)
        Plus one extra bool at the end:
          accepted: whether the new DFA was accepted
    """
    # Save old state
    old_processed_dfa = getattr(guarded_dfa_update, '_last_processed_dfa', [])
    old_event_uniq = getattr(guarded_dfa_update, '_last_event_uniq', [])

    # Run synthesis
    (new_num_states, new_processed_dfa, new_dfa_model, new_nfa_model,
     new_model_gen, new_var, new_input_dict) = dfa_update(
        trace, num_states, dfa_model, nfa_model, model_gen,
        var, input_dict, hyperparams, start_time, synth_iter_num)

    if not guard_enabled or not old_processed_dfa:
        # No old DFA to compare — accept unconditionally
        guarded_dfa_update._last_processed_dfa = new_processed_dfa
        guarded_dfa_update._last_event_uniq = new_input_dict['event_uniq']
        if verbose:
            print(colored("[DFA GUARD] ACCEPT (no previous DFA to compare)", 'green'))
        return (new_num_states, new_processed_dfa, new_dfa_model, new_nfa_model,
                new_model_gen, new_var, new_input_dict, True)

    # Check compatibility
    compatible, reason = check_dfa_compatible(
        old_processed_dfa, old_event_uniq,
        new_processed_dfa, new_input_dict['event_uniq'],
        successful_traces, verbose=verbose)

    if compatible:
        guarded_dfa_update._last_processed_dfa = new_processed_dfa
        guarded_dfa_update._last_event_uniq = new_input_dict['event_uniq']
        if verbose:
            print(colored(f"[DFA GUARD] ACCEPT: {reason}", 'green'))
        return (new_num_states, new_processed_dfa, new_dfa_model, new_nfa_model,
                new_model_gen, new_var, new_input_dict, True)
    else:
        if verbose:
            print(colored(f"[DFA GUARD] REJECT: {reason} — keeping old DFA", 'yellow'))
        # Return old values unchanged (but keep updated nfa/model_gen/var
        # so synthesis state isn't lost for next attempt)
        return (num_states, old_processed_dfa, dfa_model, nfa_model,
                new_model_gen, new_var, input_dict, False)