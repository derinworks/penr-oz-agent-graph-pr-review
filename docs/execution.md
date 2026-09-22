# Execution

`agent_graph.Graph` holds named nodes and the edges between them; `Graph.invoke` walks it.
A node is any callable that takes a `State` and returns one. An edge decides what runs next
— either always the same node, or whichever node a router picks. The walk starts at the
entry point and ends when an edge leads to `END`.

```python
from agent_graph import END, Graph, State


def draft(state: State) -> State:
    return state.update(revisions=state.get("revisions", 0) + 1)


def good_enough(state: State) -> str:
    return END if state["revisions"] >= 3 else "draft"


graph = (
    Graph()
    .add_node("draft", draft)
    .add_conditional_edges("draft", good_enough, [END, "draft"])
    .set_entry_point("draft")
)

graph.invoke(State())["revisions"]  # 3 — the cycle ran until the router said stop
```

## Why cycles are first class

A pipeline that only ever moves forward can be a list. The reason this is a graph is that
useful agent work loops: review a diff, request changes, re-read the diff, review again.
The node that decides "again or done" is the same node either way, so the runtime has to
allow an edge that points backwards.

That makes non-termination a real failure mode rather than a theoretical one — a router
reading a channel a node forgot to write will happily loop forever. So a cycle is allowed,
but never unbounded.

## The trade-off: how to bound a run

Three ways to stop a runaway loop were considered.

**Wall-clock timeout.** Abort the run after N seconds.

- Bounds the thing an operator actually cares about — latency — and needs no knowledge of
  the graph's shape.
- Not reproducible: the same graph on the same input aborts at a different node depending
  on how slow the model felt that minute, so a test for the abort is inherently flaky.
- Says nothing about cost. A fast loop can burn hundreds of API calls inside the timeout.

**Per-node visit cap.** Allow each node to run at most N times.

- Localizes the diagnosis: the error names the node that spun.
- Needs per-node bookkeeping and a per-node number to tune, and misses the failure where
  two nodes ping-pong between each other, each staying under its own cap.

**Whole-run step budget.** Allow at most N node executions per run, then abort. *Chosen.*

- Deterministic: the same graph and input abort at the same point every time, so
  `test_unbounded_cycle_aborts_when_the_step_budget_is_exceeded` is an ordinary assertion
  rather than a race.
- One number, and it is the number that correlates with cost — a step is roughly a model
  call, so the budget is a spend ceiling.
- Catches every shape of loop, including mutual recursion between nodes, because it counts
  the walk rather than any single node.
- Blunt: it cannot tell a legitimately long run from a stuck one, so a graph that genuinely
  needs many steps must say so via `step_budget=`. `DEFAULT_STEP_BUDGET` is 100 — well
  above any sensible acyclic graph, low enough that a runaway stops in seconds.

The budget counts node executions, not edge traversals, and it is exclusive: a run allowed
`step_budget=3` may execute three nodes, and raises `StepBudgetExceeded` only when a fourth
is required.

## Structure is checked before anything runs

`invoke` calls `validate()` first, so a malformed graph fails before a single node — and a
single API call — happens. Validation rejects a missing or undefined entry point, an edge
into a node that was never defined, and any node with no way out. That last rule is why
ending a run takes an explicit `add_edge("post", END)`: a node that simply has no outgoing
edge is far more often a forgotten edge than a deliberate terminal, and guessing which one
it is would turn a build mistake into a run that quietly stops early.

Conditional edges declare their targets up front for the same reason. The router is a black
box, so without a declared target list a typo could only surface as a jump at run time —
possibly on the rare branch, in production. Declaring them lets `validate()` see every
branch, and makes an off-list return value a clear `GraphError` naming both what came back
and what was allowed.

Each node gets exactly one outgoing rule. Allowing both a static edge and a conditional one
would raise the question of which wins, and any answer is a precedence rule someone has to
remember; a second rule is rejected at build time instead.

## Checkpoints

Pass a `checkpointer` and a `run_id` and a [checkpoint](checkpointing.md) is saved after
every node completes, numbered from 0:

```python
from agent_graph import MemoryCheckpointer

checkpointer = MemoryCheckpointer()
graph.invoke(State(), run_id="pr-42", checkpointer=checkpointer)

[(cp.step, cp.node) for cp in checkpointer.load_history("pr-42")]
```

Each pass through a cycle is its own checkpoint, so the history is the run's timeline rather
than a per-node overwrite. Because a checkpoint is written as its node finishes, a run that
aborts on its step budget still leaves the full history of what did run — which is usually
how you find the channel the router was waiting on.

## Resuming

Invoking again with a `run_id` that already has checkpoints resumes that run instead of
restarting it: the walk begins at whatever the last checkpoint's node routes to, carrying
that checkpoint's state, and the step counter continues from `step + 1`.

Reusing a run id *is* the resume request — there is no separate flag. The alternative,
resuming only when asked, makes the accident the default: a restarted process that passes
its run id through would re-run the entry node's side effect, then hit the checkpointer's
strictly-increasing-step rule and fail on the write. Starting over is the case that can say
so unambiguously, by naming a run of its own.

Two consequences worth stating. The checkpointed state wins over the `state` argument,
which is the input for a run that has not started and is ignored by one that has — resuming
from anything but the last completed node's own output would leave a history of a run that
never happened. And the step budget bounds the run, not the attempt: a resumed run inherits
the steps already spent, because the budget is a ceiling on what the run may cost and a
restart does not refund what the first attempt already paid.
