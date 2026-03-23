"""
DFA synthesis manager.

Encapsulates all automaton state — trace accumulation, SAT-based
synthesis, DFA transitions, and DQN module spawning — behind a
minimal interface so the training loop stays clean.

Usage in the learner:

    dfa_mgr = DFAManager(update_freq=1000, min_frames=50000,
                         module_factory=make_dqn_module)
    ...
    grew = dfa_mgr.step(episode_trace, frame_number, start_time)
    next_q = dfa_mgr.current_state
    new_agents = dfa_mgr.pop_new_agents()   # dict of freshly created modules
"""

from __future__ import annotations
from typing import Callable, Optional, Any

from synth.synth_wrapper import dfa_init, dfa_update, get_next_state


class DFAManager:
    """Thin wrapper around the SYNTH automaton synthesis loop."""

    def __init__(
        self,
        update_freq: int = 1000,
        min_frames: int = 50000,
        module_factory: Optional[Callable[[], Any]] = None,
    ):
        """
        Args:
            update_freq:     How often (in env frames) to re-run SAT synthesis.
            min_frames:      Minimum exploration frames before first synthesis.
            module_factory:  Callable that returns a fresh DQN agent.  Called
                             whenever the DFA grows a new state.  If None,
                             the caller is responsible for creating modules.
        """
        self.update_freq = update_freq
        self.min_frames = min_frames
        self._module_factory = module_factory

        # Synth internals
        self._num_states: int = 0
        self._var: Any = None
        self._input_dict: dict = {}
        self._hyperparams: Any = None
        self._model_gen: list = []
        self._nfa_model: list = []
        self._dfa_model: list = []
        self._processed_dfa: list = []
        self._synth_iter: int = 0

        # DFA bookkeeping
        self._dfa_states: list[int] = [0, 1]
        self._current_state: int = 1
        self._initial_state: int = 1

        # Trace accumulation
        self._episode_trace: list[str] = ['start']
        self._trace_archive: list[list[str]] = []

        # Newly spawned agents waiting to be collected
        self._pending_agents: dict[int, Any] = {}

        # Bootstrap
        (self._num_states, self._var,
         self._input_dict, self._hyperparams) = dfa_init()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_state(self) -> int:
        return self._current_state

    @property
    def initial_state(self) -> int:
        return self._initial_state

    @property
    def dfa_states(self) -> list[int]:
        return list(self._dfa_states)

    @property
    def episode_trace(self) -> list[str]:
        return self._episode_trace

    @property
    def is_ready(self) -> bool:
        """True once enough frames have passed to start synthesis."""
        return len(self._processed_dfa) > 0

    # ------------------------------------------------------------------
    # Trace management
    # ------------------------------------------------------------------

    def append_event(self, event: str):
        """Add a labelling event to the current episode trace."""
        self._episode_trace.append(event)

    def reset_episode(self):
        """Start a fresh episode trace (called at env reset)."""
        self._episode_trace = ['start']

    # ------------------------------------------------------------------
    # Core step — call once per env frame after MIN_DFA_FRAMES
    # ------------------------------------------------------------------

    def step(
        self,
        frame_number: int,
        start_time: float,
    ) -> bool:
        """
        Run one tick of the DFA manager.

        1. Re-synthesise the automaton if the schedule demands it or
           the current trace leads to an invalid DFA state.
        2. Compute the next DFA state from the episode trace.
        3. Spawn new DQN modules if the DFA grew.

        Returns:
            True if the DFA was re-synthesised this tick.
        """
        if frame_number < self.min_frames:
            return False

        resynthesised = False
        old_dfa_states = self._dfa_states.copy()

        # --- Should we re-run synthesis? ---
        needs_update = (frame_number % self.update_freq == 0)
        if not needs_update and self._processed_dfa:
            ns = get_next_state(
                self._episode_trace,
                self._input_dict['event_uniq'],
                self._processed_dfa,
            )
            if ns == -1 or ns == []:
                needs_update = True

        if needs_update:
            self._run_synthesis(start_time)
            resynthesised = True

        # --- Spawn modules for any new DFA states ---
        new_states = list(set(self._dfa_states) - set(old_dfa_states))
        if new_states and self._module_factory is not None:
            for s in new_states:
                self._pending_agents[s] = self._module_factory()

        # --- Advance current DFA state ---
        if self._processed_dfa:
            ns = get_next_state(
                self._episode_trace,
                self._input_dict['event_uniq'],
                self._processed_dfa,
            )
            if ns is not None and ns != -1 and ns != []:
                self._current_state = ns

        return resynthesised

    # ------------------------------------------------------------------
    # Life-lost / episode boundary hooks
    # ------------------------------------------------------------------

    def on_life_lost(self):
        """Archive the current trace and reset to initial DFA state."""
        self._trace_archive.append(self._episode_trace)
        self._episode_trace = ['start']
        self._current_state = self._initial_state

    def on_episode_end(self):
        """Called when the episode terminates (before env reset)."""
        # trace is archived in on_life_lost or here
        if self._episode_trace not in self._trace_archive:
            self._trace_archive.append(self._episode_trace)

    # ------------------------------------------------------------------
    # Module spawning
    # ------------------------------------------------------------------

    def pop_new_agents(self) -> dict[int, Any]:
        """
        Return and clear the dict of DQN agents that were created since
        the last call.  Keys are DFA state ids.
        """
        agents = self._pending_agents
        self._pending_agents = {}
        return agents

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_synthesis(self, start_time: float):
        """Accumulate traces and invoke the SAT synthesiser."""
        self._trace_archive.append(self._episode_trace)

        # Flatten all archived traces into one sequence
        flat_trace: list[str] = []
        for t in self._trace_archive:
            flat_trace.extend(t)
        flat_trace.append('start')

        (self._num_states, self._processed_dfa,
         self._dfa_model, self._nfa_model,
         self._model_gen, self._var, self._input_dict) = dfa_update(
            flat_trace,
            self._num_states,
            self._dfa_model,
            self._nfa_model,
            self._model_gen,
            self._var,
            self._input_dict,
            self._hyperparams,
            start_time,
            self._synth_iter,
        )

        self._dfa_states = list(set(
            [t[0] for t in self._processed_dfa]
            + [t[2] for t in self._processed_dfa]
        ))
        self._synth_iter += 1

        # Keep only the live episode trace going forward
        self._trace_archive = [self._episode_trace]