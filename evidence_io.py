"""Crash-aware local evidence: paired atomic snapshots and write-ahead command IDs."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import traceback
import uuid


def atomic_json_write(path, value):
    path = Path(path)
    payload = json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix="."+path.name+"-", suffix=".tmp")
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(5):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.01 * (attempt+1))  # Retry a file replacement, never an environment command.
    finally:
        if temporary.exists():
            temporary.unlink()


class EpisodeCheckpoint:
    def __init__(self, root):
        self.root = Path(root)
        self.directory = self.root / "checkpoints"
        self.directory.mkdir(exist_ok=False)
        self.session = uuid.uuid4().hex
        self.sequence = 0

    def save(self, manifest, result):
        self.sequence += 1
        identity = f"{self.session}:{self.sequence}"
        # Attempt parity is unsafe after an interrupted save: a later attempt
        # could overwrite the pair still named by checkpoint.json. Always use
        # the other slot relative to the actual committed pointer.
        pointer = self.root / 'checkpoint.json'
        if pointer.exists():
            reference = json.loads(pointer.read_bytes())
            slots = [candidate for candidate in (0, 1) if
                reference.get('manifest') == f'checkpoints/slot-{candidate}-manifest.json' and
                reference.get('result') == f'checkpoints/slot-{candidate}-result.json']
            if len(slots) != 1:
                raise ValueError('Checkpoint pointer does not name one committed slot')
            read_episode_checkpoint(self.root)
            slot = 1 - slots[0]
        else:
            slot = 1
        manifest = {**manifest, "checkpoint_id": identity}
        result = {**result, "checkpoint_id": identity}
        paths = {"manifest": f"checkpoints/slot-{slot}-manifest.json", "result": f"checkpoints/slot-{slot}-result.json"}
        # The old pointer remains usable until both files of the other slot exist.
        atomic_json_write(self.root / paths["manifest"], manifest)
        atomic_json_write(self.root / paths["result"], result)
        atomic_json_write(self.root / "checkpoint.json", {"checkpoint_id": identity, **paths})
        # Compatibility exports. Readers needing a coherent pair use the pointer.
        atomic_json_write(self.root / "auto_result.json", result)
        atomic_json_write(self.root / "manifest.json", manifest)


def read_episode_checkpoint(root):
    root = Path(root).resolve()
    pointer = root / "checkpoint.json"
    if not pointer.exists():
        return (json.loads((root / "manifest.json").read_text(encoding="utf-8")),
                json.loads((root / "auto_result.json").read_text(encoding="utf-8")))
    for _ in range(3):
        content = pointer.read_bytes()
        reference = json.loads(content)
        paths = [(root / reference[key]).resolve() for key in ("manifest", "result")]
        if any(not path.is_relative_to(root) for path in paths):
            raise ValueError("Checkpoint points outside its episode")
        manifest, result = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        if pointer.read_bytes() != content:
            continue
        if any(value.get("checkpoint_id") != reference["checkpoint_id"] for value in (manifest, result)):
            raise ValueError("Checkpoint pair identity mismatch; do not score stale mixed evidence")
        return manifest, result
    raise RuntimeError("Checkpoint changed while being read; retry inspection, not execution")


class DurableRecordLog:
    """Exclusive append-only evaluator records;IO failure propagates,never replay IO."""
    def __init__(self, path):
        self.stream = Path(path).open('x', encoding='utf-8', newline='\n')

    def append(self, value):
        self.stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self):
        self.stream.close()


class CommandJournal:
    def __init__(self, path):
        self.path = Path(path)
        self.stream = self.path.open("x", encoding="utf-8", newline="\n")
        self.session = uuid.uuid4().hex
        self.sequence = 0

    def _write(self, value):
        self.stream.write(json.dumps(value, ensure_ascii=False, default=str, allow_nan=False) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def begin(self, origin, command, before=None):
        self.sequence += 1
        identity = f"{self.session}:{self.sequence}"
        digest = (hashlib.sha256(json.dumps(before, sort_keys=True, default=str, allow_nan=False).encode()).hexdigest()
                  if before is not None else None)
        self._write({"kind": "command_start", "command_id": identity, "origin": origin,
                     "utc": datetime.now(timezone.utc).isoformat(), "command": command, "before_state_sha256": digest})
        return identity

    def end(self, identity, metadata):
        self._write({"kind": "command_end", "command_id": identity, "utc": datetime.now(timezone.utc).isoformat(),
                     "metadata": metadata})

    def error(self, identity, exception):
        self._write({"kind": "command_error", "command_id": identity, "execution_outcome": "unknown",
                     "utc": datetime.now(timezone.utc).isoformat(), "error": repr(exception), "traceback": traceback.format_exc()})

    def close(self):
        self.stream.close()


def inspect_command_journal(path):
    lines = Path(path).read_bytes().splitlines(keepends=True)
    records = []
    truncated_tail = False
    for index, line in enumerate(lines):
        if not line.endswith(b"\n"):
            if index != len(lines)-1:
                raise ValueError("Truncated command record is not the last line")
            truncated_tail = True
            break
        records.append(json.loads(line))
    pending = {}
    started = set()
    for row in records:
        if row["kind"] == "command_start":
            if row["command_id"] in started:
                raise ValueError("Duplicate command start")
            started.add(row["command_id"])
            pending[row["command_id"]] = row
        elif row["kind"] == "command_end":
            if row["command_id"] not in pending:
                raise ValueError("Command end without a matching start")
            pending.pop(row["command_id"])
        elif row["kind"] == "command_error":
            if row["command_id"] not in pending:
                raise ValueError("Command error without a matching start")
            pending[row["command_id"]] = {**pending[row["command_id"]], "error": row["error"], "execution_outcome": "unknown"}
        else:
            raise ValueError("Unknown command journal event")
    return {"records": records, "unresolved_commands": list(pending.values()), "truncated_tail": truncated_tail,
            "replay_policy": "Never automatically replay an unresolved command; inspect fresh environment state first."}
