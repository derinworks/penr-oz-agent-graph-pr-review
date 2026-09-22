# penr-oz-agent-graph-pr-review

A minimal agentic graph runtime — typed state, conditional edges, checkpointing, and a worked PR-review example.

## Status

The core runtime works: typed state ([docs](docs/state.md)), a graph executor with
conditional edges and a step budget ([docs](docs/execution.md)), and checkpointing
([docs](docs/checkpointing.md)). The PR-review example lands in a later milestone.

## Example

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

graph.invoke(State())["revisions"]  # 3
```

Cycles are allowed; unbounded ones are not. Every run carries a step budget and raises
`StepBudgetExceeded` rather than spinning forever.

## Requirements

- Python 3.11+

## Getting started

```bash
git clone https://github.com/derinworks/penr-oz-agent-graph-pr-review.git
cd penr-oz-agent-graph-pr-review
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Development

```bash
ruff check .          # lint
ruff format --check . # formatting
pytest                # tests
```

CI runs the same three steps on Python 3.11–3.13 for every push to `main` and every pull request.

## Project layout

```
src/agent_graph/   # the runtime package
docs/              # design notes and trade-offs
tests/             # test suite
```

## License

[MIT](LICENSE)
