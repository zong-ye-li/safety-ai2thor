"""Owned SSH stdio worker. One cached VLM, no environment access or public socket."""
import contextlib
import argparse
import json
import os
from pathlib import Path
import re
import signal
import sys
import traceback
import io
import shlex
import shutil
import subprocess
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference_rpc import PROTOCOL, RPCError, decode_frames, read_message, source_hashes, validate_config, write_message
from action_protocol import ProtocolError


def build_client(config):
    from llm_client import HuggingFaceClient
    options = {key: value for key, value in config.items()
               if key not in {"history_turns", "action_guide_mode", "gpu_memory_gib", "cpu_memory_gib"}}
    options["max_memory"] = {0: f"{config['gpu_memory_gib']}GiB", "cpu": f"{config['cpu_memory_gib']}GiB"}
    client = HuggingFaceClient(**options)
    client.history_turns = config["history_turns"]
    client.action_guide_mode = config["action_guide_mode"]
    return client


def serve(input_stream, output_stream, factory=build_client, root=ROOT, mailbox_directory=None):
    client = None
    session = None
    sequence = 0
    while True:
        model_call_started = False
        try:
            request = read_message(input_stream)
        except EOFError:
            return
        response = {key: request.get(key) for key in ("protocol", "session", "sequence", "request_id", "operation")}
        try:
            if set(request) != {"protocol", "session", "sequence", "request_id", "operation", "payload"}:
                raise RPCError("Unexpected request envelope fields")
            if request["protocol"] != PROTOCOL or not isinstance(request["session"], str) or not re.fullmatch(r"[0-9a-f]{32}", request["session"]):
                raise RPCError("Unsupported protocol or session")
            if session is None:
                session = request["session"]
            sequence += 1
            if (request["session"] != session or type(request["sequence"]) is not int or request["sequence"] != sequence
                    or request["request_id"] != f"{session}:{sequence}"):
                raise RPCError("Duplicate, stale or out-of-session inference request")
            operation, payload = request["operation"], request["payload"]
            if not isinstance(payload, dict):
                raise RPCError("Inference payload must be an object")
            if operation == "hello" and client is None and sequence == 1:
                if set(payload) != {"config", "source_sha256"}:
                    raise RPCError("Invalid handshake payload")
                expected = source_hashes(root)
                if payload["source_sha256"] != expected:
                    raise RPCError("Inference source differs; deploy the matching reviewed snapshot")
                config = validate_config(payload["config"])
                client = factory(config)
                runtime = dict(client.runtime_info)
                if mailbox_directory is not None:
                    runtime['remote_mailbox_directory'] = str(mailbox_directory)
                result = {"runtime": runtime, "source_sha256": expected}
            elif client is None:
                raise RPCError("A valid source/model handshake is required first")
            elif operation in {"chat", "goal", "review"}:
                fields = ({"request", "observation", "frames"} if operation == "chat" else
                          {"role", "observation", "frames"} if operation == "review" else {"observation", "frames"})
                if set(payload) != fields or not isinstance(payload["observation"], (str, dict)):
                    raise RPCError("Invalid inference observation payload")
                if operation == "chat" and not isinstance(payload["request"], (str, dict)):
                    raise RPCError("Invalid policy request")
                frames = decode_frames(payload["frames"])
                def progress(snapshot):
                    write_message(output_stream, {**response, "event": "progress", "progress": snapshot})
                client.inference_progress = progress
                try:
                    # Never attach an earlier successful call's stats to a new
                    # failure before generation/preparation finishes.
                    client.last_call_stats = {}
                    model_call_started = True
                    if operation == "chat":
                        _, raw = client.chat(payload["request"], payload["observation"], frames)
                    elif operation == 'review':
                        _,raw=client.review_dispute(payload['role'],payload['observation'],frames)
                    else:
                        raw = client.generate_task_goal(payload["observation"], frames)
                finally:
                    client.inference_progress = None
                if not isinstance(raw, str):
                    raise RPCError("Model output must be text")
                result = {"raw_response": raw, "stats": getattr(client, "last_call_stats", {})}
            elif operation == "goal_protocol" and set(payload) == {"protocol"}:
                client.set_goal_protocol(payload["protocol"])
                result = {"goal_protocol": client.goal_protocol}
            elif operation in {"reset", "close"} and not payload:
                if operation == "reset":
                    client.reset_history()
                result = {"status": operation}
            else:
                raise RPCError("Unsupported inference operation")
            write_message(output_stream, {**response, "ok": True, "result": result})
            if operation == "close":
                return
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(),
                     "stage": "model_call" if model_call_started else "request_validation"}
            if model_call_started:
                error["stats"] = getattr(client, "last_call_stats", {})
            # Retry only a completed, rejected generation. Bad envelopes,
            # interrupted generation, OOM and unknown outcomes stay fatal.
            stats = error.get('stats', {})
            retryable = (model_call_started and isinstance(exc, ProtocolError)
                         and isinstance(stats.get('raw_generation'), str)
                         and type(stats.get('output_tokens')) is int)
            error.update(retryable=retryable, session_usable=retryable)
            write_message(output_stream, {**response, "ok": False,
                "error": error})
            if not retryable:
                return  # Invalid requests or uncertain generation invalidate the session.


def atomic_bytes(path, data):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class MailboxInput:
    def __init__(self, directory, idle_timeout=43200):
        self.directory, self.sequence, self.idle_timeout = Path(directory), 0, idle_timeout

    def readline(self, limit):
        self.sequence += 1
        path = self.directory / f'{self.sequence}.request'
        deadline = time.monotonic() + self.idle_timeout
        while not path.exists():
            if time.monotonic() >= deadline:
                return b''
            time.sleep(.2)
        with path.open('rb') as stream:
            return stream.readline(limit)


class MailboxOutput:
    def __init__(self, directory):
        self.directory = Path(directory)

    def write(self, data):
        message = read_message(io.BytesIO(data))
        sequence = message['sequence']
        suffix = 'progress' if message.get('event') == 'progress' else 'result'
        path = self.directory / f'{sequence}.{suffix}'
        if suffix == 'result' and path.exists():
            raise RPCError('A completed response must never be overwritten')
        atomic_bytes(path, data)
        return len(data)

    def flush(self):
        pass


def mailbox_worker(directory, factory=build_client, root=ROOT):
    directory = Path(directory)
    # Exclusive ownership: a second worker cannot replay any accepted request.
    with (directory / 'worker.claim').open('x') as stream:
        stream.write(str(os.getpid()))
    try:
        serve(MailboxInput(directory), MailboxOutput(directory), factory=factory, root=root,
              mailbox_directory=directory)
    finally:
        atomic_bytes(directory / 'worker.exited', str(os.getpid()).encode())


def worker_environment_arguments(environ):
    """Override stale tmux-server cache settings with this SSH process's settings."""
    keys = ('HOME', 'HF_HOME', 'HF_HUB_CACHE', 'XDG_CACHE_HOME',
            'HUGGINGFACE_HUB_CACHE', 'TRANSFORMERS_CACHE',
            'PYTHONPATH', 'PYTHONDONTWRITEBYTECODE', 'TRANSFORMERS_OFFLINE',
            'PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF')
    arguments = ['env']
    for key in keys:
        if key not in environ:
            arguments.extend(['-u', key])
    arguments.extend(key+'='+environ[key] for key in keys if key in environ)
    arguments.extend(['CUDA_VISIBLE_DEVICES='+environ.get('CUDA_VISIBLE_DEVICES', '0'),
                      'HF_HUB_OFFLINE=1', 'TOKENIZERS_PARALLELISM=false'])
    return arguments


def mailbox_root(root, session_root=None):
    directory = Path(session_root) if session_root is not None else Path(root)/'inference-sessions'
    if not directory.is_absolute():
        raise RPCError('Mailbox root must be an absolute path on the inference host')
    return directory


def attach_mailbox(request, output_stream, root=ROOT, launch=None, wait_timeout=43200,
                   session_root=None):
    session, sequence = request.get('session'), request.get('sequence')
    if (request.get('protocol') != PROTOCOL or not isinstance(session, str)
            or not re.fullmatch(r'[0-9a-f]{32}', session)
            or type(sequence) is not int or sequence < 1
            or request.get('request_id') != f'{session}:{sequence}'):
        raise RPCError('Invalid mailbox request identity')
    directory = mailbox_root(root, session_root)/session
    if launch is None and not directory.exists():
        if os.name != 'posix':
            raise RPCError('SSH mailbox inference requires a POSIX host with fcntl and tmux')
        if shutil.which('tmux') is None:
            raise RPCError('SSH inference requires tmux on the remote PATH; install it before launching')
    directory.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    created = False
    try:
        directory.mkdir(mode=0o700)
        created = True
    except FileExistsError:
        pass
    data = io.BytesIO()
    write_message(data, request)
    data = data.getvalue()
    path = directory / f'{sequence}.request'
    # A lock covers publication only, never model generation or SSH attachment.
    import fcntl
    with (directory / 'publication.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if created and (sequence != 1 or request.get('operation') != 'hello'):
            raise RPCError('A new mailbox requires the initial handshake')
        if path.exists():
            if path.read_bytes() != data:
                raise RPCError('Request identity collision; refusing replacement')
        else:
            if not created and not (directory / 'worker.claim').exists():
                raise RPCError('Worker startup uncertain; refusing a second launch')
            if sequence > 1 and not (directory / f'{sequence-1}.result').exists():
                raise RPCError('Previous request has no completed response')
            if (directory / 'worker.exited').exists():
                raise RPCError('Worker exited; refusing new generation')
            atomic_bytes(path, data)
        if created:
            if launch is None:
                command = shlex.join([*worker_environment_arguments(os.environ),
                                     sys.executable, '-B', '-u', str(Path(root) / 'tools/serve_inference.py'),
                                     '--mailbox-worker', str(directory)])
                command += ' >>' + shlex.quote(str(directory / 'worker.log')) + ' 2>&1'
                subprocess.run(['tmux', 'new-session', '-d', '-s', 'safety-ai2thor-' + session, command], check=True)
            else:
                launch(directory)
    deadline = time.monotonic() + wait_timeout
    previous = None
    while time.monotonic() < deadline:
        result = directory / f'{sequence}.result'
        if result.exists():
            output_stream.write(result.read_bytes())
            output_stream.flush()
            return
        progress = directory / f'{sequence}.progress'
        if progress.exists():
            current = progress.read_bytes()
            if current != previous:
                output_stream.write(current)
                output_stream.flush()
                previous = current
        if (directory / 'worker.exited').exists():
            raise RPCError('Owned worker exited without this response; no replay')
        time.sleep(.2)
    raise TimeoutError('Mailbox attachment deadline; original worker/result retained')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--mailbox-worker', type=Path)
    mode.add_argument('--mailbox-attach', action='store_true')
    parser.add_argument('--session-root', help='Absolute mailbox root; applies only to --mailbox-attach')
    args = parser.parse_args()
    if args.session_root is not None and not args.mailbox_attach:
        parser.error('--session-root requires --mailbox-attach')
    os.environ["HF_HUB_OFFLINE"] = "1"
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    for name in ("SIGTERM", "SIGHUP"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), terminate)
    # Third-party loading/generation prints must never contaminate the JSONL wire.
    input_stream, output_stream = sys.stdin.buffer, sys.stdout.buffer
    with contextlib.redirect_stdout(sys.stderr):
        if args.mailbox_worker is not None:
            mailbox_worker(args.mailbox_worker)
        elif args.mailbox_attach:
            attach_mailbox(read_message(input_stream), output_stream, session_root=args.session_root)
        else:
            serve(input_stream, output_stream)


if __name__ == "__main__":
    main()
