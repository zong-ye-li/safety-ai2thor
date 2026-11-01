"""Private SSH inference transport: bounded JSONL, lossless RGB, one request in flight.

Unity and action execution remain local. Only a completed, rejected generation
permits bounded fresh inference attempts. Uncertain requests are not regenerated.
This module does not import Torch.
"""
import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import queue
import re
import shlex
import subprocess
import threading
import time
import uuid
from discovery_protocol import GOAL_PROTOCOLS

PROTOCOL = "safety-inference/1"
MAX_LINE_BYTES = 16 * 1024 * 1024
MAX_FRAMES = 8
MAX_SIDE = 1024


class RPCError(RuntimeError):
    pass


class RemoteInferenceError(RPCError):
    """The remote worker returned an error; transport delivered its response."""
    def __init__(self, error, transport=None):
        if not isinstance(error, dict) or not isinstance(error.get('stats', {}), dict):
            raise RPCError('Invalid remote error evidence')
        super().__init__('Remote inference failed: ' + json.dumps(error))
        self.remote_error = error
        self.stats = error.get('stats', {})
        self.transport = transport
        self.retryable = (error.get('retryable') is True
                          and error.get('session_usable') is True
                          and error.get('stage') == 'model_call'
                          and error.get('type') == 'ProtocolError'
                          and isinstance(self.stats.get('raw_generation'), str)
                          and type(self.stats.get('output_tokens')) is int)
        self.session_usable = self.retryable


def write_message(stream, message):
    data = json.dumps(message, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
    if len(data) > MAX_LINE_BYTES:
        raise RPCError("Inference message exceeds the byte budget")
    stream.write(data)
    stream.flush()
    return len(data)


def read_message(stream):
    data = stream.readline(MAX_LINE_BYTES + 1)
    if not data:
        raise EOFError("Inference stream closed")
    if len(data) > MAX_LINE_BYTES or not data.endswith(b"\n"):
        raise RPCError("Oversized or incomplete inference message")
    def invalid_constant(value):
        raise RPCError("Nonfinite JSON value: " + value)
    try:
        message = json.loads(data.decode("utf-8"), parse_constant=invalid_constant)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise RPCError("Invalid inference JSON") from exc
    if not isinstance(message, dict):
        raise RPCError("Inference message must be an object")
    return message


def source_hashes(root):
    root = Path(root)
    files = sorted(root.glob("*.py")) + [root / "tools" / "serve_inference.py"]
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def encode_frames(frames):
    import numpy as np
    from PIL import Image
    if not isinstance(frames, (list, tuple)) or not 1 <= len(frames) <= MAX_FRAMES:
        raise RPCError("Inference requires one to eight RGB frames")
    result = []
    for frame in frames:
        array = np.asarray(frame)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
            raise RPCError("Inference frames must be native uint8 RGB")
        height, width = array.shape[:2]
        if not (1 <= width <= MAX_SIDE and 1 <= height <= MAX_SIDE):
            raise RPCError("Inference frame dimensions exceed the pixel budget")
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format="PNG")
        data = buffer.getvalue()
        result.append({"encoding": "png", "width": width, "height": height,
                       "sha256": hashlib.sha256(data).hexdigest(),
                       "data": base64.b64encode(data).decode("ascii")})
    return result


def decode_frames(frames):
    import numpy as np
    from PIL import Image
    if not isinstance(frames, list) or not 1 <= len(frames) <= MAX_FRAMES:
        raise RPCError("Inference requires one to eight encoded frames")
    result = []
    for frame in frames:
        if not isinstance(frame, dict) or set(frame) != {"encoding", "width", "height", "sha256", "data"}:
            raise RPCError("Invalid inference image envelope")
        if frame["encoding"] != "png" or any(type(frame[key]) is not int or not 1 <= frame[key] <= MAX_SIDE
                                               for key in ("width", "height")):
            raise RPCError("Invalid image encoding or dimensions")
        try:
            data = base64.b64decode(frame["data"], validate=True)
        except (ValueError, TypeError) as exc:
            raise RPCError("Invalid image base64") from exc
        if hashlib.sha256(data).hexdigest() != frame["sha256"]:
            raise RPCError("Inference image SHA mismatch")
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != (frame["width"], frame["height"]):
                raise RPCError("Encoded image differs from its RGB descriptor")
            if getattr(image, "n_frames", 1) != 1:
                raise RPCError("Animated inference images are unsupported")
            image.load()
            result.append(np.array(image, copy=True))
    return result


def validate_config(config):
    if isinstance(config, dict):
        config = {"goal_protocol": "open_observation", "reasoning_token_budget": 2048,
                  "top_p": 0.95, "top_k": 64, **config}
        config = {"seed": None, "reasoning_presence_penalty": 0, 'reasoning_repetition_penalty': 1, 'sdpa_backend': 'auto', **config}
        config = {'provider': 'auto', 'quantization': 'none', **config}
    fields = {"model_name", "revision", "cache_dir", "max_new_tokens", "temperature", "enable_thinking",
              "dtype", "gpu_memory_gib", "cpu_memory_gib", "attn_implementation", "max_input_tokens",
               "action_decoding", "policy_protocol", "goal_decoding", "history_turns", "action_guide_mode", "goal_protocol", "reasoning_token_budget", "top_p", "top_k", "seed", "reasoning_presence_penalty", 'sdpa_backend'}
    fields.update({'provider', 'quantization', 'reasoning_repetition_penalty'})
    if not isinstance(config, dict) or set(config) != fields:
        raise RPCError("Unexpected inference configuration fields")
    if not isinstance(config["model_name"], str) or not config["model_name"]:
        raise RPCError("A model name is required")
    if not isinstance(config["revision"], str) or not re.fullmatch(r"[0-9a-f]{40}", config["revision"]):
        raise RPCError("SSH inference requires a pinned 40-character model revision")
    for name, low, high in (("max_new_tokens", 1, 32768), ("max_input_tokens", 1, 32768),
                            ("reasoning_token_budget", 1, 24576),
                            ("top_k", 1, 256),
                            ("history_turns", 0, 32), ("cpu_memory_gib", 1, 512)):
        if type(config[name]) is not int or not low <= config[name] <= high:
            raise RPCError("Invalid inference limit: " + name)
    for name, low, high in (("temperature", 0, 2), ("top_p", 0.01, 1), ("gpu_memory_gib", 0.1, 128),
                            ("reasoning_presence_penalty", 0, 2), ('reasoning_repetition_penalty', 1, 2)):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise RPCError("Invalid inference numeric setting: " + name)
    choices = {"dtype": {"float16", "float32", "bfloat16"}, "action_decoding": {"schema", "none"},
               'provider': {'auto', 'qwen', 'gemma4', 'generic'},
               'quantization': {'none', 'quanto-int8', 'quanto-float8'},
               "goal_decoding": {"schema", "none"}, "policy_protocol": {"action_json", "qwen_xml", "auto"},
               "action_guide_mode": {"none", "contextual", "full"}, "attn_implementation": {"eager", "sdpa"},
               'sdpa_backend': {'auto', 'efficient'},
               "goal_protocol": GOAL_PROTOCOLS}
    for name, values in choices.items():
        if config[name] not in values:
            raise RPCError("Unsupported inference setting: " + name)
    if config['quantization'] != 'none' and config['dtype'] != 'float16':
        raise RPCError('Weight-only quantization requires float16 compute')
    from model_provider import resolve_provider, GEMMA12B, GEMMA12B_REVISION
    if resolve_provider(config['provider'], config['model_name']) == 'gemma4' and (
            config['model_name'] != GEMMA12B or config['revision'] != GEMMA12B_REVISION
            or config['policy_protocol'] != 'action_json'):
        raise RPCError('Gemma requires pinned 12B/action_json configuration')
    if type(config["enable_thinking"]) is not bool or not (config["cache_dir"] is None or isinstance(config["cache_dir"], str)):
        raise RPCError("Invalid thinking or cache configuration")
    if config['reasoning_presence_penalty'] and not config['enable_thinking']:
        raise RPCError('Reasoning presence penalty requires thinking mode')
    if config['reasoning_repetition_penalty'] != 1 and not config['enable_thinking']:
        raise RPCError('Reasoning repetition penalty requires thinking mode')
    if config['sdpa_backend'] == 'efficient' and config['attn_implementation'] != 'sdpa':
        raise RPCError('Efficient-only backend requires sdpa attention')
    if config['enable_thinking'] and config['max_new_tokens'] < config['reasoning_token_budget'] + 513:
        raise RPCError("Thinking requires a closing token and >=512 reserved final tokens")
    if config['seed'] is not None and (type(config['seed']) is not int or not 0 <= config['seed'] < 2**63):
        raise RPCError('Invalid model RNG seed')
    return config


class JsonLineSession:
    """One owned child process; bounded reads and deadline includes writing stdin."""
    def __init__(self, command, stderr=None, timeout=180):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Inference timeout must be positive and finite")
        self.timeout = timeout
        self.session = uuid.uuid4().hex
        self.sequence = 0
        self.closed = False
        self.lock = threading.Lock()
        self.inbox = queue.Queue(maxsize=2)
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=stderr, creationflags=flags)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            while True:
                self.inbox.put(read_message(self.process.stdout), timeout=1)
        except BaseException as exc:
            try:
                self.inbox.put(exc, timeout=1)
            except queue.Full:
                pass

    def exchange(self, operation, payload, timeout=None):
        if self.closed:
            raise RPCError("Inference session is closed; no automatic replay")
        if not self.lock.acquire(blocking=False):
            raise RPCError("Only one inference request may be in flight")
        started = time.monotonic()
        duration = self.timeout if timeout is None else timeout
        deadline = started + duration
        self.sequence += 1
        self.last_progress = None
        request = {"protocol": PROTOCOL, "session": self.session, "sequence": self.sequence,
                   "request_id": f"{self.session}:{self.sequence}", "operation": operation, "payload": payload}
        sent = queue.Queue(maxsize=1)
        def send():
            try:
                sent.put(write_message(self.process.stdin, request))
            except BaseException as exc:
                sent.put(exc)
        try:
            threading.Thread(target=send, daemon=True).start()
            count = sent.get(timeout=max(0, deadline-time.monotonic()))
            if isinstance(count, BaseException):
                raise RPCError("Failed to send inference request") from count
            while True:
                response = self.inbox.get(timeout=max(0, deadline-time.monotonic()))
                if isinstance(response, BaseException):
                    raise RPCError("Inference worker closed or sent invalid data") from response
                for key in ("protocol", "session", "sequence", "request_id", "operation"):
                    if response.get(key) != request[key]:
                        raise RPCError("Inference response identity mismatch: " + key)
                if response.get("event") != "progress":
                    break
                if (set(response) != ((set(request) - {"payload"}) | {"event", "progress"})
                        or not isinstance(response.get("progress"), dict)):
                    raise RPCError("Invalid inference progress envelope")
                # Evidence only: never a result and never extends the hard deadline.
                self.last_progress = {**response["progress"], "request_id": request["request_id"],
                    "received_seconds": time.monotonic()-started}
            if response.get("ok") is not True:
                raise RPCError("Remote inference failed: " + json.dumps(response.get("error"), ensure_ascii=False))
            if not isinstance(response.get("result"), dict):
                raise RPCError("Invalid inference result")
            return response["result"], {"request_id": request["request_id"], "request_bytes": count,
                                          "round_trip_seconds": time.monotonic()-started}
        except queue.Empty as exc:
            self.close()
            raise TimeoutError("Inference deadline exceeded; session closed without replay") from exc
        except BaseException:
            self.close()
            raise
        finally:
            self.lock.release()

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.reader.join(timeout=1)
        for stream in (self.process.stdin, self.process.stdout):
            try:
                stream.close()
            except OSError:
                pass


class DurableSSHSession:
    """Reconnect to the same durable request; never restart a worker or generation."""
    def __init__(self, command, stderr=None, timeout=180, evidence_root=None):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('Inference timeout must be positive and finite')
        self.command, self.stderr, self.timeout = command, stderr, timeout
        self.session, self.sequence = uuid.uuid4().hex, 0
        self.closed, self.last_progress = False, None
        self.lock = threading.Lock()
        self.process = None
        self.evidence = Path(evidence_root) / ('inference-session-' + self.session)
        self.evidence.mkdir()

    def _stop_attachment(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            for stream in (self.process.stdin, self.process.stdout):
                stream.close()

    def exchange(self, operation, payload, timeout=None):
        if self.closed:
            raise RPCError('Inference session is closed; original mailbox retained')
        if not self.lock.acquire(blocking=False):
            raise RPCError('Only one inference request may be in flight')
        started = time.monotonic()
        duration = self.timeout if timeout is None else timeout
        if not math.isfinite(duration) or duration <= 0:
            self.lock.release()
            raise ValueError('Inference timeout must be positive and finite')
        deadline = started + duration
        self.sequence += 1
        self.last_progress = None
        request = dict(protocol=PROTOCOL, session=self.session, sequence=self.sequence,
                       request_id=f'{self.session}:{self.sequence}', operation=operation, payload=payload)
        data = io.BytesIO()
        attempts = 0
        try:
            count = write_message(data, request)
            wire = data.getvalue()
            with (self.evidence / f'{self.sequence}.request').open('xb') as stream:
                stream.write(wire)
                stream.flush()
                os.fsync(stream.fileno())
            while time.monotonic() < deadline:
                attempts += 1
                inbox = queue.Queue()
                flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                                stderr=self.stderr, creationflags=flags)
                process = self.process
                def receive():
                    try:
                        while True:
                            inbox.put(read_message(process.stdout))
                    except BaseException as exc:
                        inbox.put(exc)
                def send():
                    try:
                        process.stdin.write(wire)
                        process.stdin.flush()
                    except (OSError, ValueError):
                        pass  # EOF/exit below decides whether attachment is recoverable.
                reader = threading.Thread(target=receive, daemon=True)
                sender = threading.Thread(target=send, daemon=True)
                reader.start()
                sender.start()
                retry = False
                try:
                    while True:
                        response = inbox.get(timeout=max(0, deadline-time.monotonic()))
                        if isinstance(response, BaseException):
                            if not isinstance(response, EOFError):
                                raise RPCError('Invalid mailbox response; no replay') from response
                            code = process.wait(timeout=max(.001, min(5, deadline-time.monotonic())))
                            if code != 255:
                                raise RPCError(f'Mailbox attachment exited {code}; original request retained') from response
                            retry = True
                            break
                        for key in ('protocol', 'session', 'sequence', 'request_id', 'operation'):
                            if response.get(key) != request[key]:
                                raise RPCError('Inference response identity mismatch: ' + key)
                        if response.get('event') == 'progress':
                            if (set(response) != ((set(request)-{'payload'}) | {'event', 'progress'})
                                    or not isinstance(response['progress'], dict)):
                                raise RPCError('Invalid inference progress envelope')
                            self.last_progress = {**response['progress'], 'request_id': request['request_id'],
                                                  'received_seconds': time.monotonic()-started}
                            continue
                        with (self.evidence / f'{self.sequence}.result').open('xb') as stream:
                            write_message(stream, response)
                            os.fsync(stream.fileno())
                        if response.get('ok') is not True:
                            raise RemoteInferenceError(response.get('error'), dict(
                                request_id=request['request_id'], request_bytes=count,
                                round_trip_seconds=time.monotonic()-started,
                                attachment_attempts=attempts, session=self.session))
                        if not isinstance(response.get('result'), dict):
                            raise RPCError('Invalid inference result envelope')
                        return response['result'], dict(request_id=request['request_id'], request_bytes=count,
                            round_trip_seconds=time.monotonic()-started, attachment_attempts=attempts,
                            session=self.session)
                finally:
                    self._stop_attachment()
                    reader.join(timeout=1)
                    sender.join(timeout=1)
                if retry:
                    threading.Event().wait(min(2, max(0, deadline-time.monotonic())))
            raise TimeoutError('Inference deadline exceeded; remote request/result retained without regeneration')
        except RemoteInferenceError as exc:
            self.closed = not exc.session_usable
            raise
        except queue.Empty as exc:
            self.closed = True
            raise TimeoutError('Inference deadline exceeded; remote request/result retained without regeneration') from exc
        except BaseException:
            self.closed = True
            raise
        finally:
            self.lock.release()

    def close(self):
        self.closed = True
        self._stop_attachment()


class SSHInferenceClient:
    supports_native_images = True
    supports_decision_review = True

    def __init__(self, config, host, project, python, stderr_path, cuda_device="0", timeout=180,
                 session_root=None):
        config = validate_config(config)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", host) or not re.fullmatch(r"\d+", cuda_device):
            raise ValueError("Use an SSH alias and one numeric CUDA device")
        if session_root is not None and not PurePosixPath(session_root).is_absolute():
            raise ValueError('SSH session root must be an absolute path on the remote host')
        self.model_name = config["model_name"]
        self.revision = config["revision"]
        self.history_turns = config["history_turns"]
        self.action_guide_mode = config["action_guide_mode"]
        self.last_call_stats = {}
        self.stderr = Path(stderr_path).open("xb")
        self.transport = None
        command = ("cd -- " + shlex.quote(project) + " && exec " + shlex.join([
            "env", "CUDA_VISIBLE_DEVICES="+cuda_device, "HF_HUB_OFFLINE=1", "TOKENIZERS_PARALLELISM=false",
            python, "-B", "-u", "tools/serve_inference.py", "--mailbox-attach",
            *(['--session-root', session_root] if session_root is not None else [])]))
        try:
            self.transport = DurableSSHSession(["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2", host, command],
                stderr=self.stderr, timeout=timeout, evidence_root=Path(stderr_path).parent)
            expected = source_hashes(Path(__file__).resolve().parent)
            hello, stats = self.transport.exchange("hello", {"config": config, "source_sha256": expected})
            if hello.get("source_sha256") != expected:
                raise RPCError("Inference source handshake mismatch")
            self.runtime_info = {**hello["runtime"], "transport": "ssh_tmux_mailbox_lossless_png",
                                 "remote_session": self.transport.session,
                                 "local_transport_evidence": str(self.transport.evidence),
                                 "remote_host": host, "remote_project": project, "handshake": stats,
                                 "source_sha256": expected, "model_revision_requested": self.revision}
            self.policy_protocol = hello["runtime"]["policy_protocol"]
            self.goal_protocol = hello["runtime"].get("goal_protocol", "open_observation")
        except BaseException:
            self.close()
            raise

    def _invoke(self, operation, payload):
        from copy import deepcopy
        attempts = []
        retries = 2 if operation in {'chat', 'goal', 'review'} else 0
        for attempt in range(retries + 1):
            self.last_call_stats = {"status": "transport_pending", "model_attempts": deepcopy(attempts)}
            try:
                value, transport = self.transport.exchange(operation, payload)
                stats = value.get("stats", {})
                attempts.append({'attempt': attempt, 'status': 'response_received',
                                 'transport': transport, 'stats': deepcopy(stats)})
                self.last_call_stats = {**stats, "transport": transport,
                    "model_attempts": attempts, "model_request_attempts": attempt + 1,
                    "model_retry_count": attempt}
                return value
            except RemoteInferenceError as exc:
                attempts.append({'attempt': attempt, 'status': 'remote_error',
                                 'transport': exc.transport, 'error': deepcopy(exc.remote_error)})
                self.last_call_stats = {**exc.stats,
                    "status": "remote_model_error" if exc.remote_error.get('stage') == 'model_call' else "remote_request_error",
                    "remote_error": exc.remote_error, "transport": exc.transport,
                    "transport_status": "response_received", "model_attempts": deepcopy(attempts),
                    "model_request_attempts": attempt + 1, "model_retry_count": attempt,
                    "last_remote_progress": getattr(self.transport, "last_progress", None)}
                if not exc.retryable or attempt == retries:
                    raise
                # A new sequence is a new inference only after the earlier
                # generation returned a complete rejection. No native dispatch
                # occurs here; a missing/uncertain response never takes this path.
            except BaseException as exc:
                self.last_call_stats.update(status="transport_error", error=repr(exc),
                    model_attempts=deepcopy(attempts), model_request_attempts=attempt + 1,
                    last_remote_progress=getattr(self.transport, "last_progress", None))
                raise

    def chat(self, user_message, scene_state, frames):
        value = self._invoke("chat", {"request": user_message, "observation": scene_state, "frames": encode_frames(frames)})
        raw = value["raw_response"]
        if not isinstance(raw, str):
            self.close()
            raise RPCError("Inference response text is not a string")
        if self.policy_protocol == "qwen_xml":
            from native_tools import parse_native_call
            from action_protocol import ProtocolError
            try:
                parsed = parse_native_call(raw)
            except ProtocolError as exc:
                parsed = {"thought": "", "action": {"action": "FormatError"}, "format_error": str(exc)}
        else:
            from llm_client import parse_llm_response
            parsed = parse_llm_response(raw)
        return parsed, raw

    def generate_task_goal(self, scene_state, frames):
        return self._invoke("goal", {"observation": scene_state, "frames": encode_frames(frames)})["raw_response"]

    def review_dispute(self, role, packet, frames):
        from decision_review import parse_review, parse_selection_review
        if role not in ('planner_review','auditor','selection_inspector'):
            raise ValueError('Unknown reviewer role')
        raw=self._invoke('review',{'role':role,'observation':packet,'frames':encode_frames(frames)})['raw_response']
        return (parse_selection_review(raw) if role=='selection_inspector' else parse_review(raw)),raw

    def reset_history(self):
        self._invoke("reset", {})

    def set_goal_protocol(self, protocol):
        if protocol not in GOAL_PROTOCOLS:
            raise ValueError("Unknown goal protocol")
        result = self._invoke("goal_protocol", {"protocol": protocol})
        if result.get("goal_protocol") != protocol:
            self.close()
            raise RPCError("Goal protocol acknowledgement mismatch")
        self.goal_protocol = protocol
        self.runtime_info["goal_protocol"] = protocol

    def close(self):
        if self.transport is not None and not self.transport.closed:
            try:
                self.transport.exchange("close", {}, timeout=5)
            except (RPCError, TimeoutError, OSError):
                pass
            finally:
                self.transport.close()
        if not self.stderr.closed:
            self.stderr.close()
