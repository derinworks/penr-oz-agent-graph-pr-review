import pytest

from agent_graph import (
    END,
    Checkpoint,
    Graph,
    GraphError,
    MemoryCheckpointer,
    State,
    StepBudgetExceeded,
)


def append(tag: str):
    """Return a node that records that it ran, in order."""

    def node(state: State) -> State:
        return state.update(trail=[*state.get("trail", []), tag])

    return node


def counter(state: State) -> State:
    return state.update(count=state.get("count", 0) + 1)


# --- acceptance: a linear graph runs to completion ------------------------------


def test_linear_graph_runs_every_node_in_order() -> None:
    graph = (
        Graph()
        .add_node("fetch", append("fetch"))
        .add_node("review", append("review"))
        .add_node("post", append("post"))
        .add_edge("fetch", "review")
        .add_edge("review", "post")
        .add_edge("post", END)
        .set_entry_point("fetch")
    )
    assert graph.invoke(State())["trail"] == ["fetch", "review", "post"]


def test_single_node_graph_runs_to_completion() -> None:
    graph = Graph().add_node("only", counter).add_edge("only", END).set_entry_point("only")
    assert graph.invoke(State())["count"] == 1


def test_state_threads_through_nodes_without_mutating_the_input() -> None:
    graph = (
        Graph()
        .add_node("a", counter)
        .add_node("b", counter)
        .add_edge("a", "b")
        .add_edge("b", END)
        .set_entry_point("a")
    )
    start = State(count=0)
    assert graph.invoke(start)["count"] == 2
    assert start["count"] == 0


# --- acceptance: a cyclic graph runs to completion -----------------------------


def test_cyclic_graph_loops_until_its_router_routes_to_end() -> None:
    graph = (
        Graph()
        .add_node("revise", counter)
        .add_conditional_edges(
            "revise",
            lambda state: END if state["count"] >= 3 else "revise",
            [END, "revise"],
        )
        .set_entry_point("revise")
    )
    assert graph.invoke(State(count=0))["count"] == 3


def test_conditional_edge_picks_among_branches() -> None:
    graph = (
        Graph()
        .add_node("triage", append("triage"))
        .add_node("approve", append("approve"))
        .add_node("reject", append("reject"))
        .add_conditional_edges(
            "triage",
            lambda state: "approve" if state["ok"] else "reject",
            ["approve", "reject"],
        )
        .add_edge("approve", END)
        .add_edge("reject", END)
        .set_entry_point("triage")
    )
    assert graph.invoke(State(ok=True))["trail"] == ["triage", "approve"]
    assert graph.invoke(State(ok=False))["trail"] == ["triage", "reject"]


def test_a_node_may_be_revisited_within_one_run() -> None:
    graph = (
        Graph()
        .add_node("work", append("work"))
        .add_node("check", counter)
        .add_edge("work", "check")
        .add_conditional_edges(
            "check",
            lambda state: END if state["count"] >= 2 else "work",
            [END, "work"],
        )
        .set_entry_point("work")
    )
    assert graph.invoke(State())["trail"] == ["work", "work"]


# --- acceptance: an unbounded cycle aborts on the step budget ------------------


def test_unbounded_cycle_aborts_when_the_step_budget_is_exceeded() -> None:
    graph = Graph().add_node("spin", counter).add_edge("spin", "spin").set_entry_point("spin")
    with pytest.raises(StepBudgetExceeded, match="step budget of 5"):
        graph.invoke(State(), step_budget=5)


def test_step_budget_defaults_high_enough_to_stop_a_runaway_cycle() -> None:
    graph = Graph().add_node("spin", counter).add_edge("spin", "spin").set_entry_point("spin")
    with pytest.raises(StepBudgetExceeded):
        graph.invoke(State())


def test_a_run_needing_exactly_the_budget_succeeds() -> None:
    graph = (
        Graph()
        .add_node("tick", counter)
        .add_conditional_edges(
            "tick",
            lambda state: END if state["count"] >= 3 else "tick",
            [END, "tick"],
        )
        .set_entry_point("tick")
    )
    assert graph.invoke(State(count=0), step_budget=3)["count"] == 3
    with pytest.raises(StepBudgetExceeded):
        graph.invoke(State(count=0), step_budget=2)


def test_step_budget_must_be_a_positive_int() -> None:
    graph = Graph().add_node("a", counter).add_edge("a", END).set_entry_point("a")
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="step_budget must be a positive int"):
            graph.invoke(State(), step_budget=bad)


# --- structural validation -----------------------------------------------------


def test_invoke_requires_an_entry_point() -> None:
    graph = Graph().add_node("a", counter).add_edge("a", END)
    with pytest.raises(GraphError, match="no entry point set"):
        graph.invoke(State())


def test_entry_point_must_be_a_defined_node() -> None:
    graph = Graph().add_node("a", counter).add_edge("a", END).set_entry_point("missing")
    with pytest.raises(GraphError, match="entry point 'missing' is not a defined node"):
        graph.invoke(State())


def test_edge_to_undefined_node_is_rejected() -> None:
    graph = Graph().add_node("a", counter).add_edge("a", "ghost").set_entry_point("a")
    with pytest.raises(GraphError, match="leads to undefined node 'ghost'"):
        graph.invoke(State())


def test_conditional_target_to_undefined_node_is_rejected() -> None:
    graph = (
        Graph()
        .add_node("a", counter)
        .add_conditional_edges("a", lambda _: END, [END, "ghost"])
        .set_entry_point("a")
    )
    with pytest.raises(GraphError, match="leads to undefined node 'ghost'"):
        graph.invoke(State())


def test_node_without_an_outgoing_edge_is_rejected() -> None:
    graph = (
        Graph()
        .add_node("a", counter)
        .add_node("orphan", counter)
        .add_edge("a", END)
        .set_entry_point("a")
    )
    with pytest.raises(GraphError, match="node 'orphan' has no outgoing edge"):
        graph.invoke(State())


def test_validate_can_be_called_before_running() -> None:
    graph = Graph().add_node("a", counter).add_edge("a", END).set_entry_point("a")
    assert graph.validate() is None


def test_duplicate_node_name_is_rejected() -> None:
    graph = Graph().add_node("a", counter)
    with pytest.raises(ValueError, match="node 'a' is already defined"):
        graph.add_node("a", counter)


def test_a_node_may_have_only_one_outgoing_rule() -> None:
    graph = Graph().add_node("a", counter).add_edge("a", END)
    with pytest.raises(ValueError, match="node 'a' already has an outgoing edge"):
        graph.add_edge("a", "a")
    with pytest.raises(ValueError, match="node 'a' already has an outgoing edge"):
        graph.add_conditional_edges("a", lambda _: END, [END])


def test_end_is_reserved_as_a_node_name() -> None:
    with pytest.raises(ValueError, match="is reserved for the terminal marker"):
        Graph().add_node(END, counter)


def test_node_name_must_be_a_non_empty_string() -> None:
    with pytest.raises(ValueError, match="node name must be a non-empty string"):
        Graph().add_node("", counter)


def test_node_must_be_callable() -> None:
    with pytest.raises(TypeError, match="must be callable"):
        Graph().add_node("a", "not a function")


def test_conditional_edges_require_at_least_one_target() -> None:
    graph = Graph().add_node("a", counter)
    with pytest.raises(ValueError, match="declare no targets"):
        graph.add_conditional_edges("a", lambda _: END, [])


# --- runtime contract ----------------------------------------------------------


def test_router_returning_an_undeclared_target_is_rejected() -> None:
    graph = (
        Graph()
        .add_node("a", counter)
        .add_conditional_edges("a", lambda _: "elsewhere", [END])
        .set_entry_point("a")
    )
    with pytest.raises(GraphError, match="returned 'elsewhere', which is not among"):
        graph.invoke(State())


def test_node_returning_a_non_state_is_rejected() -> None:
    graph = Graph().add_node("a", lambda _: {"count": 1}).add_edge("a", END).set_entry_point("a")
    with pytest.raises(TypeError, match="node 'a' must return a State, got dict"):
        graph.invoke(State())


def test_invoke_requires_a_state() -> None:
    graph = Graph().add_node("a", counter).add_edge("a", END).set_entry_point("a")
    with pytest.raises(TypeError, match="state must be a State, got dict"):
        graph.invoke({"count": 0})


# --- checkpointing -------------------------------------------------------------


def test_checkpoints_are_saved_after_every_node() -> None:
    checkpointer = MemoryCheckpointer()
    graph = (
        Graph()
        .add_node("a", append("a"))
        .add_node("b", append("b"))
        .add_edge("a", "b")
        .add_edge("b", END)
        .set_entry_point("a")
    )
    final = graph.invoke(State(), run_id="pr-42", checkpointer=checkpointer)

    history = checkpointer.load_history("pr-42")
    assert [(cp.step, cp.node) for cp in history] == [(0, "a"), (1, "b")]
    assert history[0].state["trail"] == ["a"]
    assert checkpointer.load_latest("pr-42").state == final


def test_checkpoints_record_each_pass_of_a_cycle_separately() -> None:
    checkpointer = MemoryCheckpointer()
    graph = (
        Graph()
        .add_node("tick", counter)
        .add_conditional_edges(
            "tick",
            lambda state: END if state["count"] >= 3 else "tick",
            [END, "tick"],
        )
        .set_entry_point("tick")
    )
    graph.invoke(State(count=0), run_id="run-1", checkpointer=checkpointer)

    history = checkpointer.load_history("run-1")
    assert [cp.step for cp in history] == [0, 1, 2]
    assert [cp.state["count"] for cp in history] == [1, 2, 3]


def test_checkpointer_requires_a_run_id() -> None:
    graph = Graph().add_node("a", counter).add_edge("a", END).set_entry_point("a")
    with pytest.raises(ValueError, match="run_id is required"):
        graph.invoke(State(), checkpointer=MemoryCheckpointer())


def test_checkpoints_survive_a_run_that_aborts_on_its_budget() -> None:
    checkpointer = MemoryCheckpointer()
    graph = Graph().add_node("spin", counter).add_edge("spin", "spin").set_entry_point("spin")
    with pytest.raises(StepBudgetExceeded):
        graph.invoke(State(count=0), step_budget=3, run_id="doomed", checkpointer=checkpointer)

    history = checkpointer.load_history("doomed")
    assert [cp.step for cp in history] == [0, 1, 2]
    assert history[-1].state["count"] == 3


# --- resuming ------------------------------------------------------------------


def test_a_killed_run_resumes_from_its_last_checkpoint() -> None:
    checkpointer = MemoryCheckpointer()
    calls: list[str] = []
    fuse = ["b"]  # the process dies the first time node "b" runs

    def node(tag: str):
        def run(state: State) -> State:
            calls.append(tag)
            if fuse and fuse[0] == tag:
                fuse.pop()
                raise RuntimeError(f"process killed while running {tag!r}")
            return state.update(trail=[*state.get("trail", []), tag])

        return run

    graph = (
        Graph()
        .add_node("a", node("a"))
        .add_node("b", node("b"))
        .add_node("c", node("c"))
        .add_edge("a", "b")
        .add_edge("b", "c")
        .add_edge("c", END)
        .set_entry_point("a")
    )
    with pytest.raises(RuntimeError, match="process killed"):
        graph.invoke(State(), run_id="pr-42", checkpointer=checkpointer)
    assert calls == ["a", "b"]
    assert [(cp.step, cp.node) for cp in checkpointer.load_history("pr-42")] == [(0, "a")]

    # The restart hands back the latest checkpoint, exactly as docs/checkpointing.md shows.
    latest = checkpointer.load_latest("pr-42")
    assert latest is not None
    resumed = graph.invoke(latest.state, run_id="pr-42", checkpointer=checkpointer)

    assert calls == ["a", "b", "b", "c"]  # "a" is not re-executed
    assert resumed["trail"] == ["a", "b", "c"]
    assert [(cp.step, cp.node) for cp in checkpointer.load_history("pr-42")] == [
        (0, "a"),
        (1, "b"),
        (2, "c"),
    ]


def test_a_resumed_run_continues_the_step_counter_and_state_of_the_checkpoint() -> None:
    checkpointer = MemoryCheckpointer()
    checkpointer.save(Checkpoint("pr-7", 4, "a", State(trail=["a"])))
    graph = (
        Graph()
        .add_node("a", append("a"))
        .add_node("b", append("b"))
        .add_edge("a", "b")
        .add_edge("b", END)
        .set_entry_point("a")
    )

    final = graph.invoke(State(trail=["ignored"]), run_id="pr-7", checkpointer=checkpointer)

    assert final["trail"] == ["a", "b"]  # the checkpointed state wins over the argument
    assert [(cp.step, cp.node) for cp in checkpointer.load_history("pr-7")] == [(4, "a"), (5, "b")]


def test_a_resumed_run_does_not_get_a_fresh_step_budget() -> None:
    checkpointer = MemoryCheckpointer()
    graph = Graph().add_node("spin", counter).add_edge("spin", "spin").set_entry_point("spin")
    with pytest.raises(StepBudgetExceeded):
        graph.invoke(State(count=0), step_budget=3, run_id="doomed", checkpointer=checkpointer)

    with pytest.raises(StepBudgetExceeded):
        graph.invoke(State(count=0), step_budget=3, run_id="doomed", checkpointer=checkpointer)
    assert [cp.step for cp in checkpointer.load_history("doomed")] == [0, 1, 2]  # nothing ran

    with pytest.raises(StepBudgetExceeded):
        graph.invoke(State(count=0), step_budget=5, run_id="doomed", checkpointer=checkpointer)
    history = checkpointer.load_history("doomed")
    assert [cp.step for cp in history] == [0, 1, 2, 3, 4]  # a raised budget spends the remainder
    assert history[-1].state["count"] == 5


def test_resuming_a_finished_run_runs_nothing_and_returns_its_final_state() -> None:
    checkpointer = MemoryCheckpointer()
    graph = Graph().add_node("only", counter).add_edge("only", END).set_entry_point("only")
    final = graph.invoke(State(count=0), run_id="done", checkpointer=checkpointer)

    assert graph.invoke(State(count=99), run_id="done", checkpointer=checkpointer) == final
    assert [cp.step for cp in checkpointer.load_history("done")] == [0]


def test_resuming_a_run_whose_checkpointed_node_is_gone_is_rejected() -> None:
    checkpointer = MemoryCheckpointer()
    checkpointer.save(Checkpoint("pr-9", 0, "retired", State()))
    graph = Graph().add_node("a", counter).add_edge("a", END).set_entry_point("a")

    with pytest.raises(GraphError, match="cannot resume run 'pr-9'"):
        graph.invoke(State(), run_id="pr-9", checkpointer=checkpointer)


def test_an_unusable_run_id_is_rejected_before_any_node_runs() -> None:
    ran: list[str] = []

    def node(state: State) -> State:
        ran.append("a")  # stands in for an irreversible API call
        return state

    graph = Graph().add_node("a", node).add_edge("a", END).set_entry_point("a")

    with pytest.raises(ValueError, match="run_id must match"):
        graph.invoke(State(), run_id="../bad", checkpointer=MemoryCheckpointer())
    assert ran == []
