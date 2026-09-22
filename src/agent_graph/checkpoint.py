"""Durable snapshots of a run, saved after every node.

A :class:`Checkpoint` records the :class:`~agent_graph.state.State` as it stood
once a node finished, tagged with the run it belongs to and its position within
that run. A :class:`Checkpointer` persists checkpoints so that a killed run
resumes from the last completed node instead of restarting. The reasoning
behind the design is documented in ``docs/checkpointing.md``.
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from agent_graph.state import State

__all__ = ["Checkpoint", "Checkpointer", "FileCheckpointer", "MemoryCheckpointer"]

_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

_FIELDS = ("node", "run_id", "state", "step")


def is_run_id(run_id: object) -> bool:
    """Return whether ``run_id`` is an id every backend can key on as-is."""
    return isinstance(run_id, str) and _RUN_ID.fullmatch(run_id) is not None


def check_run_id(run_id: object) -> None:
    """Raise ``ValueError`` unless ``run_id`` is an id every backend can key on.

    The rule lives here, next to the pattern, so that :class:`Checkpoint`, the
    backends, and callers that want to reject a bad id *before* doing work
    (``Graph.invoke``) all enforce exactly one definition of it.
    """
    if not is_run_id(run_id):
        raise ValueError(
            f"run_id must match {_RUN_ID.pattern!r} so any backend can key on it, got {run_id!r}"
        )


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """The state of one run as it stood after ``node`` completed step ``step``.

    ``run_id`` names the run and is restricted to ``[A-Za-z0-9._-]`` (it may
    not start with ``.``) so that every backend — including directories on a
    filesystem — can key on it as-is. ``step`` is the checkpoint's position in
    the run, starting at 0 and strictly increasing.
    """

    run_id: str
    step: int
    node: str
    state: State

    def __post_init__(self) -> None:
        # fullmatch, not match: `$` also matches before a trailing newline, so
        # `match` would accept "pr-42\n" (e.g. an id straight off readline())
        # and mint a second, near-invisible run directory beside "pr-42".
        check_run_id(self.run_id)
        if not isinstance(self.step, int) or isinstance(self.step, bool) or self.step < 0:
            raise ValueError(f"step must be a non-negative int, got {self.step!r}")
        if not isinstance(self.node, str) or not self.node:
            raise ValueError(f"node must be a non-empty string, got {self.node!r}")
        if not isinstance(self.state, State):
            raise TypeError(f"state must be a State, got {type(self.state).__name__}")

    def to_json(self) -> str:
        """Serialize to a JSON object string.

        The state is serialized through :meth:`State.to_json`, so its
        round-trip guarantees (and ``TypeError`` / ``ValueError`` refusals)
        apply unchanged.
        """
        channels = json.loads(self.state.to_json())
        record = {"run_id": self.run_id, "step": self.step, "node": self.node, "state": channels}
        return json.dumps(record, sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> Checkpoint:
        """Rebuild a checkpoint from :meth:`to_json` output."""
        record = json.loads(payload)
        if not isinstance(record, dict):
            raise ValueError(f"expected a JSON object, got {type(record).__name__}")
        if tuple(sorted(record)) != _FIELDS:
            raise ValueError(f"expected fields {list(_FIELDS)}, got {sorted(record)}")
        if not isinstance(record["state"], dict):
            raise ValueError(f"state must be a JSON object, got {type(record['state']).__name__}")
        return cls(record["run_id"], record["step"], record["node"], State(record["state"]))


class Checkpointer(ABC):
    """Persists a checkpoint after every node so a run resumes, not restarts.

    Within a run, steps must strictly increase: :meth:`save` rejects a step at
    or below the last saved one, which keeps "latest" unambiguous and the
    history replayable. To resume a run, take :meth:`load_latest` and continue
    from its ``state`` at ``step + 1``.
    """

    @abstractmethod
    def save(self, checkpoint: Checkpoint) -> None:
        """Persist ``checkpoint``.

        Raises ``ValueError`` if its step is not greater than the last saved
        step of the same run.
        """

    @abstractmethod
    def load_latest(self, run_id: str) -> Checkpoint | None:
        """Return the newest checkpoint of ``run_id``, or ``None`` if it has none."""

    @abstractmethod
    def load_history(self, run_id: str) -> list[Checkpoint]:
        """Return every checkpoint of ``run_id``, oldest first."""


class MemoryCheckpointer(Checkpointer):
    """Keeps checkpoints in a dict.

    Convenient for tests and in-process replay; a process crash loses
    everything, so durability requires :class:`FileCheckpointer`.
    """

    def __init__(self) -> None:
        self._runs: dict[str, list[Checkpoint]] = {}

    def save(self, checkpoint: Checkpoint) -> None:
        history = self._runs.setdefault(checkpoint.run_id, [])
        if history and checkpoint.step <= history[-1].step:
            raise ValueError(
                f"step {checkpoint.step} is not after the last saved step "
                f"{history[-1].step} of run {checkpoint.run_id!r}"
            )
        history.append(checkpoint)

    def load_latest(self, run_id: str) -> Checkpoint | None:
        history = self._runs.get(run_id)
        return history[-1] if history else None

    def load_history(self, run_id: str) -> list[Checkpoint]:
        return list(self._runs.get(run_id, ()))


class FileCheckpointer(Checkpointer):
    """Writes one JSON file per checkpoint under ``root/<run_id>/<step>.json``.

    Each file is written to a temporary name and atomically renamed into
    place, so a checkpoint either exists complete or not at all: killing the
    process mid-write leaves the previous checkpoint as the latest, and a
    restart resumes from the last node that actually finished.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self._root = Path(root)

    def save(self, checkpoint: Checkpoint) -> None:
        run_dir = self._root / checkpoint.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        entries = self._entries(run_dir)
        if entries and checkpoint.step <= entries[-1][0]:
            raise ValueError(
                f"step {checkpoint.step} is not after the last saved step "
                f"{entries[-1][0]} of run {checkpoint.run_id!r}"
            )
        final = run_dir / f"{checkpoint.step:08d}.json"
        tmp = run_dir / f"{checkpoint.step:08d}.tmp"
        with open(tmp, "w", encoding="utf-8") as file:
            file.write(checkpoint.to_json())
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp, final)

    def load_latest(self, run_id: str) -> Checkpoint | None:
        entries = self._entries(self._run_dir(run_id))
        if not entries:
            return None
        return self._read(run_id, *entries[-1])

    def load_history(self, run_id: str) -> list[Checkpoint]:
        entries = self._entries(self._run_dir(run_id))
        return [self._read(run_id, step, path) for step, path in entries]

    def _run_dir(self, run_id: str) -> Path | None:
        # An id Checkpoint would reject can never have been saved (and must
        # not be used as a path), so its run directory is treated as absent.
        if not is_run_id(run_id):
            return None
        return self._root / run_id

    @staticmethod
    def _entries(run_dir: Path | None) -> list[tuple[int, Path]]:
        if run_dir is None or not run_dir.is_dir():
            return []
        return sorted(
            (int(path.stem), path) for path in run_dir.glob("*.json") if path.stem.isdigit()
        )

    @staticmethod
    def _read(run_id: str, step: int, path: Path) -> Checkpoint:
        checkpoint = Checkpoint.from_json(path.read_text(encoding="utf-8"))
        if checkpoint.run_id != run_id or checkpoint.step != step:
            raise ValueError(
                f"checkpoint file {path} claims run {checkpoint.run_id!r} step "
                f"{checkpoint.step}, which does not match its location"
            )
        return checkpoint
