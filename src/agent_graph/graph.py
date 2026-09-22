"""The graph and the executor that walks it.

A :class:`Graph` is a set of named nodes plus the edges that decide which node
runs next. Executing it means starting at the entry node and following edges
until one leads to :data:`END`, threading a :class:`~agent_graph.state.State`
through every node on the way.

Edges come in two kinds. A *static* edge always leads to the same node. A
*conditional* edge asks a router — any callable that reads the state and names
the next node — and so can branch, or loop back to a node that already ran.
Loops are the point: an agent that reviews, revises, and reviews again is a
cycle. They are also how a runtime hangs, so every run carries a **step
budget** and aborts with :class:`StepBudgetExceeded` rather than spinning
forever. The reasoning behind the design is documented in ``docs/execution.md``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Final

from agent_graph.checkpoint import Checkpoint, Checkpointer, check_run_id
from agent_graph.state import State

__all__ = ["DEFAULT_STEP_BUDGET", "END", "Graph", "GraphError", "StepBudgetExceeded"]

NodeFn = Callable[[State], State]
"""A node: reads a state snapshot, returns the snapshot that leaves it."""

Router = Callable[[State], str]
"""A conditional edge: reads a state snapshot, names the node to run next."""

END: Final[str] = "__end__"
"""The terminal marker. An edge leading here ends the run.

Spelled as a reserved node name rather than a sentinel object so that routers,
error messages, checkpoints, and future graph exports all handle it as an
ordinary JSON-native string.
"""

DEFAULT_STEP_BUDGET: Final[int] = 100
"""Node executions allowed per run when the caller does not say otherwise.

High enough that no sensible acyclic graph trips it, low enough that a runaway
cycle stops in seconds instead of burning a budget of real API calls.
"""


class GraphError(Exception):
    """A graph is malformed, or a router named a node it was not allowed to."""


class StepBudgetExceeded(GraphError):
    """A run needed more node executions than its step budget allowed."""


class Graph:
    """Named nodes and the edges between them, executable via :meth:`invoke`.

    Every mutator returns ``self``, so a graph reads as one chained expression.
    Structure is checked in two places: the mutators reject what they can see
    on their own (duplicate names, an edge leaving a node twice), and
    :meth:`validate` catches what only makes sense once the whole graph is
    known (a dangling target, a node with no way out).
    """

    def __init__(self) -> None:
        self._nodes: dict[str, NodeFn] = {}
        self._edges: dict[str, str] = {}
        self._conditionals: dict[str, tuple[Router, tuple[str, ...]]] = {}
        self._entry: str | None = None

    def add_node(self, name: str, fn: NodeFn) -> Graph:
        """Register ``fn`` under ``name``.

        Raises ``ValueError`` if the name is empty, reserved, or already taken.
        """
        self._check_node_name(name)
        if name in self._nodes:
            raise ValueError(f"node {name!r} is already defined")
        if not callable(fn):
            raise TypeError(f"node {name!r} must be callable, got {type(fn).__name__}")
        self._nodes[name] = fn
        return self

    def add_edge(self, source: str, target: str) -> Graph:
        """Always run ``target`` after ``source``. ``target`` may be :data:`END`."""
        self._check_source_is_free(source)
        if not isinstance(target, str) or not target:
            raise ValueError(f"edge target must be a non-empty string, got {target!r}")
        self._edges[source] = target
        return self

    def add_conditional_edges(self, source: str, router: Router, targets: Iterable[str]) -> Graph:
        """Ask ``router`` which node follows ``source``, restricted to ``targets``.

        ``targets`` is declared up front rather than inferred from whatever the
        router happens to return: it lets :meth:`validate` reject a dangling
        branch before the run starts, keeps the graph statically inspectable,
        and turns a router typo into a clear error instead of a silent jump.
        """
        self._check_source_is_free(source)
        if not callable(router):
            raise TypeError(f"router for node {source!r} must be callable")
        declared = tuple(dict.fromkeys(targets))
        if not declared:
            raise ValueError(f"conditional edges from node {source!r} declare no targets")
        for target in declared:
            if not isinstance(target, str) or not target:
                raise ValueError(f"edge target must be a non-empty string, got {target!r}")
        self._conditionals[source] = (router, declared)
        return self

    def set_entry_point(self, name: str) -> Graph:
        """Start runs at ``name``."""
        self._check_node_name(name)
        self._entry = name
        return self

    def validate(self) -> None:
        """Raise :class:`GraphError` unless the graph can be walked.

        Called by :meth:`invoke`; call it directly to fail at build time.
        """
        if self._entry is None:
            raise GraphError("no entry point set; call set_entry_point()")
        if self._entry not in self._nodes:
            raise GraphError(f"entry point {self._entry!r} is not a defined node")
        for source, target in self._edges.items():
            self._check_defined(source, (target,))
        for source, (_, targets) in self._conditionals.items():
            self._check_defined(source, targets)
        for name in self._nodes:
            if name not in self._edges and name not in self._conditionals:
                raise GraphError(
                    f"node {name!r} has no outgoing edge; add one to {END} to end the run there"
                )

    def invoke(
        self,
        state: State,
        *,
        step_budget: int = DEFAULT_STEP_BUDGET,
        run_id: str | None = None,
        checkpointer: Checkpointer | None = None,
    ) -> State:
        """Walk from the entry node to :data:`END`, returning the final state.

        At most ``step_budget`` nodes run; a run that still has not reached
        :data:`END` raises :class:`StepBudgetExceeded`. When ``checkpointer`` is
        given, a checkpoint is saved after each node completes, so a killed run
        can be inspected — or resumed — from the last node that actually
        finished; ``run_id`` names that run and is required alongside it, and is
        checked for backend-portability before any node runs, so an unusable id
        cannot cost an API call on its way to failing.

        **Resuming is what a repeated ``run_id`` means.** If that run already
        has checkpoints, the walk picks up at the node its last checkpoint's
        edge leads to, rather than at the entry node — so the entry node's side
        effects are not repeated — and three things follow from taking the
        checkpoint as the run's position:

        - The checkpointed state wins: ``state`` is the input for a run that has
          not started yet, and is ignored by one that has. Resuming from
          anything but the last completed node's own output would make the
          history a replay of a run that never happened.
        - The step counter continues from ``step + 1``, so ``step_budget``
          bounds the run as a whole — a restart is not a fresh budget — and the
          checkpointer's strictly increasing steps keep holding.
        - Nothing runs at all if that edge already leads to :data:`END`; the
          finished run's final state comes straight back.

        Pass a ``run_id`` of its own to a run that should start from scratch.
        """
        if not isinstance(state, State):
            raise TypeError(f"state must be a State, got {type(state).__name__}")
        if not isinstance(step_budget, int) or isinstance(step_budget, bool) or step_budget < 1:
            raise ValueError(f"step_budget must be a positive int, got {step_budget!r}")
        if checkpointer is not None:
            if run_id is None:
                raise ValueError("run_id is required when a checkpointer is given")
            # Before validate(), and well before the entry node: an id no
            # Checkpoint could carry dooms the run, so it must not be found out
            # only once a node has already done irreversible work.
            check_run_id(run_id)
        self.validate()

        current = self._entry
        assert current is not None  # validate() proved it
        steps = 0
        if checkpointer is not None:
            assert run_id is not None  # guarded above
            latest = checkpointer.load_latest(run_id)
            if latest is not None:
                state, current, steps = self._resume_from(latest)
        while current != END:
            if steps >= step_budget:
                raise StepBudgetExceeded(
                    f"run exceeded its step budget of {step_budget} with node {current!r} "
                    "still pending; the graph is cycling without reaching a terminal edge"
                )
            result = self._nodes[current](state)
            if not isinstance(result, State):
                raise TypeError(
                    f"node {current!r} must return a State, got {type(result).__name__}"
                )
            state = result
            if checkpointer is not None:
                assert run_id is not None  # guarded above
                checkpointer.save(Checkpoint(run_id, steps, current, state))
            steps += 1
            current = self._next(current, state)
        return state

    def _resume_from(self, latest: Checkpoint) -> tuple[State, str, int]:
        """Return the state, node, and step count a run picks up with.

        The checkpoint records the node that *finished*, so where to go next is
        that node's outgoing edge — asked again, of the state the node left
        behind, exactly as the first attempt would have asked it.
        """
        if latest.node not in self._nodes:
            raise GraphError(
                f"cannot resume run {latest.run_id!r}: its latest checkpoint names node "
                f"{latest.node!r}, which this graph does not define"
            )
        return latest.state, self._next(latest.node, latest.state), latest.step + 1

    def _next(self, node: str, state: State) -> str:
        """Name the node that follows ``node`` given ``state``."""
        if node in self._edges:
            return self._edges[node]
        router, targets = self._conditionals[node]
        target = router(state)
        if target not in targets:
            raise GraphError(
                f"router for node {node!r} returned {target!r}, which is not among its "
                f"declared targets {list(targets)}"
            )
        return target

    def _check_defined(self, source: str, targets: Iterable[str]) -> None:
        """Raise unless ``source`` is a node and every target is a node or :data:`END`."""
        if source not in self._nodes:
            raise GraphError(f"edge leaves undefined node {source!r}")
        for target in targets:
            if target != END and target not in self._nodes:
                raise GraphError(f"edge from node {source!r} leads to undefined node {target!r}")

    def _check_source_is_free(self, source: str) -> None:
        """Raise unless ``source`` is a usable name with no outgoing edge yet.

        One rule per node keeps "what runs next" answerable by reading a single
        edge, and makes a second, conflicting rule an error at build time
        rather than a silent precedence surprise at run time.
        """
        self._check_node_name(source)
        if source in self._edges or source in self._conditionals:
            raise ValueError(f"node {source!r} already has an outgoing edge")

    @staticmethod
    def _check_node_name(name: str) -> None:
        """Raise unless ``name`` is a non-empty string that is not reserved."""
        if not isinstance(name, str) or not name:
            raise ValueError(f"node name must be a non-empty string, got {name!r}")
        if name == END:
            raise ValueError(f"node name {END!r} is reserved for the terminal marker")
