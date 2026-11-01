"""Run one VLM-controlled robot episode with explicit output and verification boundaries."""
import argparse
import json
import sys
import time
import traceback
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from action_protocol import ProtocolError
from llm_client import HuggingFaceClient, OllamaClient, OpenAICompatibleClient
from runtime_support import load_conditions, source_manifest, verify_completion, verify_task, native_dispatch_feedback, project_dispatch_feedback, NATIVE_DISPATCH_VERSION
from thor_wrapper import ThorWrapper
from video_processor import VideoProcessor
from policy_io import observation_packet, policy_request
from model_evidence import ModelEvidence
from tool_registry import available_tools, validate_available_action
from survey import InitialSurvey
from run_profiles import RunArgumentParser, PROFILES
from evidence_io import EpisodeCheckpoint
from open_discovery import OpenDiscovery


def build_llm(args):
    from generation_controls import validate_reasoning_presence_penalty, validate_reasoning_repetition_penalty
    presence = validate_reasoning_presence_penalty(getattr(args, 'reasoning_presence_penalty', 0))
    repetition = validate_reasoning_repetition_penalty(getattr(args, 'reasoning_repetition_penalty', 1))
    if repetition != 1 and args.llm not in {'huggingface', 'ssh'}:
        raise ValueError('Reasoning repetition penalty is supported only by direct HF and SSH inference')
    if presence and args.llm not in {'huggingface', 'ssh'}:
        raise ValueError('Reasoning presence penalty is supported only by direct HF and SSH inference')
    if args.llm == "ollama":
        client = OllamaClient(model_name=args.model, host=args.llm_host or "http://localhost:11434")
    elif args.llm == "openai":
        client = OpenAICompatibleClient(model_name=args.model,
            base_url=args.llm_host or "http://localhost:8000/v1", api_key=args.api_key)
    elif args.llm == "ssh":
        from inference_rpc import SSHInferenceClient
        if not args.gpu_memory_gib:
            raise ValueError("SSH inference requires an explicit --gpu-memory-gib budget")
        config = {"model_name": args.model, "revision": args.model_revision, "cache_dir": args.cache_dir,
                  'provider': getattr(args, 'provider', 'auto'), 'quantization': getattr(args, 'quantization', 'none'),
                  "max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
                  "enable_thinking": args.thinking, "dtype": args.dtype, "gpu_memory_gib": args.gpu_memory_gib,
                  "reasoning_token_budget": getattr(args, "reasoning_token_budget", 2048),
                  "reasoning_presence_penalty": presence,
                  "reasoning_repetition_penalty": repetition,
                  "top_p": args.top_p, "top_k": args.top_k,
                  "seed": args.seed,
                  "cpu_memory_gib": args.cpu_memory_gib, "attn_implementation": args.attention,
                  "sdpa_backend": args.sdpa_backend,
                  "max_input_tokens": args.max_input_tokens, "action_decoding": args.action_decoding,
                  "goal_decoding": args.goal_decoding, "goal_protocol": "open_observation", "policy_protocol": args.policy_protocol,
                  "history_turns": args.history_turns, "action_guide_mode": args.action_guide}
        client = SSHInferenceClient(config, args.ssh_host, args.ssh_project, args.ssh_python,
            args.inference_stderr_log, cuda_device=args.ssh_cuda_device, timeout=args.inference_timeout,
            session_root=args.ssh_session_root)
    elif args.llm == "huggingface":
        memory = None
        if args.gpu_memory_gib:
            memory = {0: f"{args.gpu_memory_gib}GiB", "cpu": f"{args.cpu_memory_gib}GiB"}
        client = HuggingFaceClient(model_name=args.model, cache_dir=args.cache_dir,
            provider=getattr(args, 'provider', 'auto'), quantization=getattr(args, 'quantization', 'none'),
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            enable_thinking=args.thinking, revision=args.model_revision,
            reasoning_token_budget=getattr(args, "reasoning_token_budget", 2048),
            reasoning_presence_penalty=presence,
            reasoning_repetition_penalty=repetition,
            top_p=args.top_p, top_k=args.top_k,
            seed=args.seed,
            dtype=args.dtype, max_memory=memory, attn_implementation=args.attention,
            sdpa_backend=args.sdpa_backend,
            max_input_tokens=args.max_input_tokens, action_decoding=args.action_decoding,
            policy_protocol=args.policy_protocol, goal_decoding=args.goal_decoding, goal_protocol="open_observation")
    else:
        raise ValueError(f"Unsupported inference backend: {args.llm}")
    if getattr(args, 'require_all_weights_on_gpu', False):
        placement=getattr(client, 'runtime_info', {}).get('parameter_bytes_by_device', {})
        if (not isinstance(placement, dict) or not placement or any(
                not isinstance(device, str) or not device.startswith('cuda:') or type(size) is not int or size <= 0
                for device,size in placement.items())):
            client.close()
            raise ValueError('This configuration requires all model weights on GPU; no inference or CPU offload trial')
    client.history_turns = args.history_turns
    client.action_guide_mode = args.action_guide
    return client


def run_auto(args, env_factory=ThorWrapper, llm_factory=None):
    if (getattr(args,'planning_record_tables',False) and not getattr(args,'planning_shared_inputs',False)):
        raise ValueError('Record tables require lossless shared inputs before environment startup')
    if (getattr(args,'discovery_patrol_context',False) or getattr(args,'discovery_end_review',False)) and not getattr(args,'discovery_planning_v2',False):
        raise ValueError('Continual discovery options require planning v2 before environment startup')
    if getattr(args,'planning_label_review',False):
        if not getattr(args,'planning_target_correspondence',False):
            raise ValueError('Label review requires target correspondence before environment startup')
        if llm_factory is None and args.llm not in {'huggingface','ssh'}:
            raise ValueError('Label review requires HF/SSH review support before environment startup')
    if (getattr(args,'planning_target_correspondence',False)
            and not (getattr(args,'planning_target_reports',False) and getattr(args,'policy_protocol',None)=='action_json')):
        raise ValueError('Target correspondence requires target reports and action_json before environment startup')
    if (getattr(args,'planning_context_evidence',False)
            and not (getattr(args,'discovery_planning_v2',False) and getattr(args,'planning_operation_history',False))):
        raise ValueError('Context evidence requires planning v2 and operation history before environment startup')
    if (getattr(args,'planning_target_reports',False)
            and not (getattr(args,'planning_capability_feasibility',False)
                     and getattr(args,'planning_task_continuity',False)
                     and getattr(args,'planning_operation_history',False))):
        raise ValueError('Actor target reports require capability feasibility, task continuity and operation history before environment startup')
    if (getattr(args,'physics_clock','legacy')=='owned-v2'
            and args.editor_port is None and not args.local_executable):
        raise ValueError('Owned-v2 scaled environment time requires an explicitly qualified custom player')
    if getattr(args,'planning_shared_images',False) and not getattr(args,'discovery_planning_v2',False):
        raise ValueError('Shared images require planning v2 before environment startup')
    if getattr(args,'planning_shared_inputs',False) and not getattr(args,'discovery_planning_v2',False):
        raise ValueError('Shared inputs require planning v2 before environment startup')
    if (getattr(args,'planning_operation_history',False)
            and not (getattr(args,'discovery_planning_v2',False) and getattr(args,'planning_capability_provenance',False))):
        raise ValueError('Operation history requires planning v2 and provenance before environment startup')
    if (getattr(args,'planning_visual_inspection',False)
            and not getattr(args,'discovery_planning_v2',False)):
        raise ValueError('Visual inspection requires planning v2 before environment startup')
    if (getattr(args,'planning_capability_feasibility',False)
            and not (getattr(args,'discovery_planning_v2',False) and getattr(args,'planning_capability_provenance',False))):
        raise ValueError('Capability feasibility requires planning v2 and provenance before environment startup')
    if (getattr(args,'planning_capability_feasibility',False)
            and getattr(args,'policy_protocol','action_json')!='action_json'):
        raise ValueError('Capability feasibility requires action_json before environment startup')
    if (getattr(args, 'planning_task_continuity', False)
            and not (getattr(args, 'discovery_planning_v2', False)
                     and getattr(args, 'planning_capability_provenance', False))):
        raise ValueError('Task continuity requires planning v2 and capability provenance before environment startup')
    if (getattr(args, 'planning_capability_provenance', False)
            and not getattr(args, 'discovery_planning_v2', False)):
        raise ValueError('Capability/source provenance requires discovery planning v2 before environment startup')
    if args.goal_pipeline != "open-observation":
        raise ValueError("Only the open-observation pipeline is included in this public package")
    if args.coverage_positions:
        raise ValueError("Target-independent movement integration is not yet validated for open discovery; use model-selected observation motions")
    # A supplied factory lends its client; only the default factory transfers ownership.
    owns_llm = llm_factory is None
    if llm_factory is None:
        llm_factory = build_llm
    if args.ai2thor_source:
        sys.path.insert(0, str(Path(args.ai2thor_source).resolve()))
    output = Path(args.output_dir) if args.output_dir else Path("outputs") / (
        args.scene + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    conditions = load_conditions(args.completion_conditions) if args.completion_conditions else None
    explicit_reference_conditions = deepcopy(conditions)
    config = dict(vars(args))
    config["api_key"] = "[redacted]"
    manifest = {**source_manifest(Path(__file__).parent), "config": config,
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "observation": "RGB plus simulator visible-object aliases and inventory; optional simulator depth",
                "status": "starting"}
    results = []
    goal_attempts = []
    task = args.task
    env = None
    llm = None
    started = time.monotonic()
    termination = "not_started"
    primary_error = None
    verifier = {"status": "unknown", "reason": "No completion request yet."}
    task_baseline = None
    task_contract = None
    stage = "preparing_episode"
    events = (output / "events.jsonl").open("x", encoding="utf-8")
    model_evidence = ModelEvidence(output)
    checkpoint = EpisodeCheckpoint(output)
    manifest["model_calls"] = "model_calls.jsonl"
    manifest["input_images"] = "model_inputs/"
    manifest["command_journal"] = "commands.jsonl"
    manifest["checkpoint_pointer"] = "checkpoint.json"
    if explicit_reference_conditions is not None:
        manifest["external_reference_conditions"] = explicit_reference_conditions

    def execute(action, origin):
        from runtime_support import ReceivedExecutionError
        contract = getattr(env, 'native_dispatch_contract', None)
        if contract is not None and contract != NATIVE_DISPATCH_VERSION:
            raise ValueError('Unsupported IO-owner native dispatch contract;no command sent')
        try:
            if getattr(env, "supports_command_origin", False):
                result=env.execute_action(action, origin=origin)
            else:
                result=env.execute_action(action)
            project_dispatch_feedback(result, require_receipt=contract is not None)
            from held_appearance import verify
            if action.get('action') not in {'RequestReplan','Done'}:
                try: verify(env,save)
                except BaseException as exc:
                    if result.get('native_dispatch_status') != 'outcome_received': raise
                    raise ReceivedExecutionError(result,'held_appearance_verification',exc) from exc
        except ReceivedExecutionError as exc:
            try:
                record(dict(kind='execution_postprocessing_failure',origin=origin,action=deepcopy(action),**exc.execution))
            except BaseException as record_error:
                # A secondary journal error must not replace the acknowledged
                # outcome on its way to task/operation retention.
                exc.execution['audit_retention_error'] = str(record_error)
            raise
        return result

    def record(item):
        results.append(item)
        events.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
        events.flush()

    def save():
        manifest["last_stage"] = stage
        manifest["last_command_id"] = getattr(env, "last_command_id", None)
        manifest["render_contract"] = getattr(env, "render_contract", None)
        manifest['held_appearance_contract']=deepcopy(getattr(env,'held_appearance_contract',None))
        if env is not None and getattr(env, 'physics_clock', None) is not None:
            manifest['physical_time'] = env.physics_description()
        manifest["elapsed_seconds"] = time.monotonic() - started
        data = {"task": task, "scene": args.scene, "steps": results,
                "goal_attempts": goal_attempts, "termination": termination,
                "verification": verifier, "action_audit": env.audit if env else []}
        checkpoint.save(manifest, data)

    try:
        save()
        stage = "environment_initialization"
        save()
        env = env_factory(scene=args.scene, width=args.width, height=args.height,
            render_depth=args.depth, render_instance_seg=True,
            platform=args.platform, editor_port=args.editor_port,
            local_executable_path=args.local_executable, command_journal=output / "commands.jsonl")
        manifest["initial_state"] = env.snapshot()
        if args.scenario:
            stage = "scenario_setup"
            if args.scenario in {"held-mug-counter", "held-egg-counter"}:
                task, conditions, setup = env.setup_held_problem(args.placement_reference, args.scenario,
                    1 if args.held_opacity is None else args.held_opacity)
            else:
                task, conditions, setup = env.setup_problem(args.scenario)
            manifest["scenario_setup"] = setup
            task_contract = setup.get("task_contract")
            if task_contract is not None:
                manifest["task_contract"] = task_contract
            manifest["completion_conditions"] = conditions
        stage='held_rgb_configuration'
        from held_appearance import configure
        configure(env,args,save)
        if getattr(args, 'physics_clock', 'legacy') in {'owned-v1','owned-v2'}:
            from physics_time import SETTLING_CONTRACT
            stage = 'physical_clock_initialization'
            manifest['physical_clock_records'] = 'physical_clock.jsonl'
            # Dedicated durable evaluator log;not a policy-selected action/event budget.
            from evidence_io import DurableRecordLog
            clock_log = DurableRecordLog(output / 'physical_clock.jsonl')
            clock_options=({'environment_time':True} if args.physics_clock=='owned-v2' else {})
            manifest['physics_settling'] = env.start_physics_clock(clock_log.append, SETTLING_CONTRACT,**clock_options)
            model_evidence.wait_boundary = env.check_physics_wait
        task_baseline = deepcopy(env.snapshot())
        manifest["task_initial_state"] = task_baseline
        if task and args.completion_mode == "automatic" and conditions:
            verifier = verify_task(env.snapshot(), conditions, task_baseline, args.allow_destructive_tools)
            if verifier["status"] == "complete":
                termination = "verified_state_achieved"
                manifest["status"] = termination
                manifest["completion_origin"] = "preexisting_task_state; no policy call"
                return output
        survey = InitialSurvey()
        stage = "initial_survey"
        save()
        survey.capture(env)
        # Inspection motions are part of the same recorded environment-action budget.
        for index in range(args.initial_scan):
            if len(env.audit) >= args.max_steps:
                break
            execution = execute({"action": "RotateRight", "degrees": 90}, "survey")
            record({"kind": "initial_scan", "step": index, "action_name": "RotateRight",
                    "action": {"action": "RotateRight", "degrees": 90}, **execution})
            survey.capture(env)
            verifier = verify_task(env.snapshot(), conditions, task_baseline, args.allow_destructive_tools)
            if verifier["preservation"]["status"] == "violation":
                termination = "constraint_violation"
                return output
        manifest["initial_survey"] = survey.description()
        stage = "model_initialization"
        save()
        if args.llm == "ssh":
            if owns_llm:
                args.inference_stderr_log = str(output / "inference-worker.log")
            manifest["inference_stderr_log"] = getattr(args, "inference_stderr_log", None)
        if model_evidence.wait_boundary is not None:
            model_evidence.wait_boundary()
        llm = llm_factory(args)
        if model_evidence.wait_boundary is not None:
            model_evidence.wait_boundary()
        manifest["model"] = {"name": args.model, "revision_requested": args.model_revision,
                             "revision_resolved": getattr(getattr(getattr(llm, "model", None), "config", None), "_commit_hash", None),
                             "runtime": getattr(llm, "runtime_info", {})}
        processor = VideoProcessor()

        def input_frames():
            process = processor.native_frames if getattr(llm, "supports_native_images", False) else processor.encode_frames
            frames = env.get_recent_frames(args.frame_window)
            return process(frames, args.frame_width, args.frame_height)

        if not task:
            stage = "open_observation_discovery"
            discovery = OpenDiscovery()
            manifest["open_discovery"] = discovery.report
            manifest["task_origin"] = "model_free_observation_hypothesis_and_goal"
            manifest["completion_scope"] = "Discovery, evidence and local action completion scored separately; external reference evaluator-only"
            def discovery_action(action, origin):
                if len(env.audit) >= args.max_steps:
                    return {"success": False, "category": "budget_exhausted", "error": "No environment action budget remains", **native_dispatch_feedback('not_sent')}
                result = execute(action, origin)
                try:
                    record({"kind": origin, "action": action, **result})
                    save()
                except BaseException as exc:
                    if result.get('native_dispatch_status') != 'outcome_received': raise
                    from runtime_support import ReceivedExecutionError
                    raise ReceivedExecutionError(result,'execution_checkpoint',exc) from exc
                return result
            termination = discovery.run(env, llm, args, survey, model_evidence, discovery_action, save, task_baseline, record)
            manifest["status"] = termination
            return output
        physical_failures = 0
        rejected_done = 0
        termination = "step_limit"
        for step in range(args.max_steps):
            if len(env.audit) >= args.max_steps:
                break
            previous = next((row for row in reversed(results) if row["kind"] == "policy"), None)
            policy_history = [row for row in results if row["kind"] == "policy"]
            policy_history = policy_history[-args.history_turns:] if args.history_turns else []
            managed_completion = args.completion_mode == "automatic" and bool(conditions)
            prompt = policy_request(task, previous, completion_managed=managed_completion, execution_history=policy_history,
                                    allow_destructive=args.allow_destructive_tools, task_contract=task_contract, tool_scope=args.tool_scope)
            parsed = None
            for attempt in range(args.format_retries + 1):
                stage = "policy_input_preparation"
                frames = input_frames()
                scene_summary = observation_packet(env.get_visible_objects_summary())
                if "recentFrames" in scene_summary:
                    scene_summary["recentFrames"] = scene_summary["recentFrames"][-args.frame_window:]
                stage = "policy_inference"
                save()
                parsed, raw = model_evidence.invoke("action", step, attempt, prompt, scene_summary, frames,
                    lambda: llm.chat(prompt, json.dumps(scene_summary), frames), lambda: getattr(llm, "last_call_stats", {}))
                if parsed["action"]["action"] != "FormatError":
                    break
                error = parsed.get("format_error", "Invalid action JSON")
                record({"kind": "format_retry", "step": step, "attempt": attempt,
                        "model_call_id": f"call-{model_evidence.sequence:06d}",
                        "action_name": "FormatError", "action": {"action": "FormatError"},
                        "success": False, "error": error, "raw_response": raw,
                        "call_stats": getattr(llm, "last_call_stats", {})})
                prompt = policy_request(task, previous, format_error=error, completion_managed=managed_completion, execution_history=policy_history,
                                        allow_destructive=args.allow_destructive_tools, task_contract=task_contract, tool_scope=args.tool_scope)
            if parsed["action"]["action"] == "FormatError":
                termination = "format_retry_exhausted"
                break
            action = parsed["action"]
            stage = "action_validation"
            audit_before = len(env.audit)
            try:
                validate_available_action(action, available_tools(scene_summary, managed_completion, args.allow_destructive_tools,
                    task_contract if args.tool_scope == "task" else None))
                eligible = True
            except ProtocolError as exc:
                eligible = False
                execution = {"success": False, "error": str(exc), "category": "precondition_not_met", **native_dispatch_feedback('not_sent')}
            if eligible and action["action"] == "Done":
                verifier = verify_task(env.snapshot(), conditions, task_baseline, args.allow_destructive_tools)
                execution = {"success": verifier["status"] == "complete",
                             "error": "" if verifier["status"] == "complete" else verifier["reason"], **native_dispatch_feedback('not_sent')}
            elif eligible:
                stage = "environment_execution"
                save()
                execution = execute(action, "policy")
            record({"kind": "policy", "step": step, "thought": parsed.get("thought", ""),
                    "model_call_id": f"call-{model_evidence.sequence:06d}",
                    "decision_source": "model",
                    "action_name": action["action"], "action": action, "raw_response": raw,
                    "call_stats": getattr(llm, "last_call_stats", {}),
                    "observation": env.get_scene_state(),
                    **execution})
            print(f"Step {step}: {action['action']} success={execution['success']}", flush=True)
            save()
            if execution.get('category') in {'preservation_violation','preservation_unknown'}:
                termination='constraint_violation' if execution['category']=='preservation_violation' else 'preservation_unknown'
                break
            stage = "postcondition_verification"
            state_verification = verify_task(env.snapshot(), conditions, task_baseline, args.allow_destructive_tools)
            if state_verification["preservation"]["status"] == "violation":
                verifier = state_verification
                termination = "constraint_violation"
                break
            if eligible and action["action"] == "Done":
                if verifier["status"] == "complete":
                    termination = "verified_complete"
                    break
                if verifier["status"] == "unknown":
                    termination = "unverified_stop"
                    break
                rejected_done += 1
                if rejected_done >= 3:
                    termination = "repeated_incomplete_done"
                    break
                continue
            if args.completion_mode == "automatic" and conditions:
                verifier = state_verification
                if verifier["status"] == "complete":
                    termination = "verified_state_achieved"
                    manifest["completion_origin"] = "controller_postcondition_after_action"
                    break
            if execution["success"]:
                physical_failures = 0
            elif len(env.audit) > audit_before and execution.get("category", "physical_failure") in {"physical_failure", "collision"}:
                physical_failures += 1
            else:
                # Type incompatibility, a closed receiver or capacity requires a
                # different choice/state, not an automatic spatial rotation.
                physical_failures = 0
            if physical_failures >= 3 and len(env.audit) < args.max_steps:
                stage = "recovery_execution"
                save()
                recovery = execute({"action": "RotateRight", "degrees": 90}, "recovery")
                record({"kind": "recovery", "step": step, "action_name": "RotateRight",
                        "action": {"action": "RotateRight", "degrees": 90}, **recovery})
                physical_failures = 0
                verifier = verify_task(env.snapshot(), conditions, task_baseline, args.allow_destructive_tools)
                if verifier["preservation"]["status"] == "violation":
                    termination = "constraint_violation"
                    break
        manifest["status"] = termination
    except BaseException as exc:
        primary_error = exc
        termination = "interrupted" if isinstance(exc, KeyboardInterrupt) else "runtime_error"
        manifest.update(status=termination, error=repr(exc), failure_stage=stage,
                        error_type=type(exc).__name__, traceback=traceback.format_exc())
        from runtime_support import NativeDispatchError
        if isinstance(exc, NativeDispatchError):
            manifest['failed_native_dispatch'] = deepcopy(exc.execution)
        if hasattr(exc, 'execution_failure_feedback'):
            manifest['failed_execution_feedback'] = deepcopy(exc.execution_failure_feedback)
        manifest["last_model_call_stats"] = getattr(llm, "last_call_stats", {})
        raise
    finally:
        try:
            if env is not None:
                manifest["setup_action_audit"] = getattr(env, "setup_audit", [])
                # Score the terminal physical state even if the policy did not ask Done.
                # This does not convert a step limit or exception into a successful stop.
                try:
                    manifest["final_state_verification"] = verify_task(env.snapshot(), conditions, task_baseline, args.allow_destructive_tools)
                except Exception as exc:
                    manifest["final_state_verification"] = {"status": "unknown", "reason": repr(exc)}
            if env is not None and args.export_video and env.get_all_frames():
                try:
                    metadata = VideoProcessor().export_mp4(env.get_all_frames(), str(output / "episode_record.mp4"), fps=args.video_fps)
                    manifest["video"] = "episode_record.mp4"
                    manifest["video_metadata"] = metadata
                except Exception as exc:
                    manifest["video_error"] = repr(exc)
            if manifest["status"] == "starting":
                manifest["status"] = termination
            if primary_error is None:
                save()
            else:
                from runtime_support import retain_after_failure
                feedback = getattr(primary_error,'execution_failure_feedback',{})
                manifest['failed_execution_feedback'] = feedback
                retain_after_failure(primary_error,feedback,'terminal_checkpoint',save)
        finally:
            def cleanup(stage, callback):
                if primary_error is None:
                    callback()
                else:
                    from runtime_support import retain_after_failure
                    feedback = getattr(primary_error,'execution_failure_feedback',{})
                    retain_after_failure(primary_error,feedback,stage,callback)
            cleanup('events_close', events.close)
            cleanup('model_evidence_close', model_evidence.close)
            if 'clock_log' in locals():
                cleanup('clock_log_close', clock_log.close)
            try:
                # Injected clients may be reused by a batch owner; that owner closes them.
                if llm is not None and owns_llm and hasattr(llm, "close"):
                    cleanup('model_client_close', llm.close)
            finally:
                if env is not None:
                    cleanup('environment_close', env.close)
    print(f"Saved {termination}: {output}", flush=True)
    return output


def build_parser():
    parser = RunArgumentParser(description=__doc__)
    parser.add_argument('--physics-clock', choices=('legacy', 'owned-v1','owned-v2'), default='legacy',
                        help='Opt-in sampled settling and frozen model waits;requires separate native qualification')
    parser.add_argument("--profile",choices=tuple(PROFILES),default="weak-2b",
                        help="Coherent model/IO defaults; explicit flags override the selected profile")
    parser.add_argument("--scene", default="FloorPlan1")
    parser.add_argument("--llm", choices=["ollama", "openai", "huggingface", "ssh"], default="huggingface")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--model-revision")
    parser.add_argument("--llm-host")
    parser.add_argument("--api-key", default="not-needed")
    parser.add_argument("--cache-dir")
    parser.add_argument("--ssh-host")
    parser.add_argument("--ssh-project",
                        help="Matching remote source snapshot, relative to the SSH login home or absolute")
    parser.add_argument("--ssh-python")
    parser.add_argument("--ssh-cuda-device", default="0")
    parser.add_argument("--ssh-session-root", help="Absolute writable remote mailbox root; default: <ssh-project>/inference-sessions")
    parser.add_argument("--inference-timeout", type=float, default=180)
    parser.add_argument("--task")
    parser.add_argument("--goal-pipeline", choices=["open-observation"], default="open-observation",
                        help="Model-formed open problem hypotheses")
    parser.add_argument("--discovery-rounds", type=int, default=3, help="Bounded observation/hypothesis/re-evaluation rounds")
    parser.add_argument("--discovery-revision", action="store_true", help="Record a fresh-context model revision of each valid draft before execution")
    parser.add_argument('--discovery-planning-v2', action='store_true', help='Opt-in explicit uncertainty/recall/outcome contract, full planning memory and terminal read-only review')
    parser.add_argument('--planning-capability-provenance', action='store_true',
                        help='Opt-in static API capability, input/output roles and scoped model-authored task/offer provenance; requires planning v2')
    parser.add_argument('--planning-task-continuity', action='store_true',
                        help='Opt-in exact original/current task intent and received history-field coverage; requires planning v2 and capability provenance')
    parser.add_argument('--planning-capability-feasibility',action='store_true',
                        help='Separate authorized API invocation from advisory current feasibility; requires planning v2 and provenance')
    parser.add_argument('--planning-target-reports',action='store_true',
                        help='Actor-selected typed identity reports without native effects;requires feasibility,task continuity and operation history')
    parser.add_argument('--planning-target-correspondence',action='store_true',
                        help='Bind actor-declared intent/RGB selection before object-directed dispatch;requires target reports and action_json')
    parser.add_argument('--planning-label-review',action='store_true',
                        help='Independent pre-dispatch target review and bounded label arbitration; requires correspondence and HF/SSH')
    parser.add_argument('--planning-visual-inspection',action='store_true',
                        help='Model-selected read-only regions of previously exposed RGB;requires planning v2')
    parser.add_argument('--planning-operation-history',action='store_true',
                        help='Trace actual model/controller operations with bounded explicit coverage;requires planning v2 and provenance')
    parser.add_argument('--planning-record-tables',action='store_true',
                        help='Opt in to lossless historical record columns; current views/tools/intent remain literal')
    parser.add_argument('--planning-shared-inputs',action='store_true',
                        help='Reversibly share exact repeated input values;images/current scene/tools stay literal;requires planning v2')
    parser.add_argument('--planning-shared-images',action='store_true',
                        help='Share exact model-input RGB rasters per call while preserving observation identities;requires planning v2')
    parser.add_argument('--discovery-recall-limit', type=int, default=8, help='Separate total RGB recall request allowance in planning v2; no native commands')
    parser.add_argument('--history-page-bytes', type=int, default=16384,
                        help='Planning-memory event JSON byte budget per recall; separate from actual processor token limit')
    parser.add_argument("--rgb-observation-actor", action="store_true", help="Use the same recorded discovery RGB/order with own-body contracts and no implicit actor history for observe decisions")
    parser.add_argument("--scenario", choices=["faucet-off", "fridge-closed", "held-mug-counter", "held-egg-counter"])
    parser.add_argument("--placement-reference", help="Passed native placement reference report; evaluator only")
    parser.add_argument('--planning-context-evidence',action='store_true',
                        help='Separate delivered non-RGB input field citations from visual evidence; candidate protocol')
    parser.add_argument('--held-appearance',choices=('auto','required','opaque'),default='auto',
                        help='Auto-enable RGB opacity on custom players;required refuses unsupported deployment;opaque is an explicit control')
    parser.add_argument("--held-opacity", type=float,default=None,
                        help="Episode-wide RGB-only held opacity;custom auto defaults0.5 to retain the held outline,legacy held fixtures1")
    parser.add_argument("--tool-scope", choices=["affordance", "task"], default="affordance",
                        help="Optional explicit task-type constraints; does not select an object alias")
    parser.add_argument("--completion-conditions", help="Evaluator-only JSON postconditions")
    parser.add_argument("--completion-mode", choices=["automatic", "model"], default="automatic",
                        help="Automatic stops on explicit postconditions; model mode separately measures correct Done")
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--format-retries", type=int, default=2)
    parser.add_argument("--history-turns", type=int, default=6)
    parser.add_argument("--action-guide", choices=["none", "contextual", "full"], default="contextual")
    parser.add_argument("--allow-destructive-tools", action="store_true",
                        help="Explicitly enable BreakObject, DirtyObject and ThrowObject; recorded in run config")
    parser.add_argument("--policy-protocol", choices=["auto", "action_json", "qwen_xml"], default="action_json",
                        help="Use processor-native Qwen tools when advertised, or explicitly select legacy JSON")
    parser.add_argument("--action-decoding", choices=["none", "schema"], default="schema",
                        help="HF protocol-specific JSON/native grammar filtering; all backends validate before execution")
    parser.add_argument("--goal-decoding", choices=["none", "schema"], default="schema",
                        help="HF typed goal proposal decoding; valid proposals do not certify room safety")
    parser.add_argument("--max-input-tokens", type=int, default=4096,
                        help="Post-processor token budget; evict old history, never current observations")
    parser.add_argument("--initial-scan", type=int, choices=[0, 4], default=4)
    parser.add_argument('--discovery-patrol-context',action='store_true',
                        help='Future comparison only: explicitly describe a bounded continuing safety mission')
    parser.add_argument('--discovery-end-review',action='store_true',
                        help='Future comparison only: offer one neutral end review per task within existing budgets')
    parser.add_argument("--coverage-positions", type=int, choices=range(0, 33), default=0,
                        help="Opt-in native-map ordinary movement survey, up to 32 sampled positions; global max-steps includes navigation")
    parser.add_argument("--width", type=int, default=600)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--frame-window", type=int, default=4)
    parser.add_argument("--frame-width", type=int, default=448)
    parser.add_argument("--frame-height", type=int, default=448)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument('--reasoning-repetition-penalty', type=float, default=1,
                        help='Thinking-only multiplicative repetition penalty (1 disables; prompt/final excluded)')
    parser.add_argument("--reasoning-presence-penalty", type=float, default=0,
                        help="Opt-in 0..2 presence subtraction on generated reasoning only; prompt/final excluded")
    parser.add_argument("--seed", type=int, help="Optional initial model RNG seed; recorded, not reset between calls")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--require-all-weights-on-gpu", action="store_true")
    parser.add_argument("--reasoning-token-budget", type=int, default=2048,
                        help="Bound thinking tokens; max-new-tokens must also reserve closing tag and >=512 final tokens")
    parser.add_argument("--dtype", choices=["float16", "float32", "bfloat16"], default="float16")
    parser.add_argument('--provider', choices=['auto', 'qwen', 'gemma4', 'generic'], default='auto')
    parser.add_argument('--quantization', choices=['none', 'quanto-int8', 'quanto-float8'], default='none')
    parser.add_argument("--attention", default="eager")
    parser.add_argument('--sdpa-backend', choices=['auto', 'efficient'], default='auto',
                        help='Efficient-only SDPA with explicit GQA head expansion; fail rather than fall back to quadratic math attention')
    parser.add_argument("--gpu-memory-gib", type=float)
    parser.add_argument("--cpu-memory-gib", type=int, default=64)
    parser.add_argument("--depth", action="store_true")
    parser.add_argument("--platform", choices=["cloud", "default"], default="default")
    parser.add_argument("--editor-port", type=int)
    parser.add_argument("--local-executable")
    parser.add_argument("--ai2thor-source")
    parser.add_argument("--output-dir")
    parser.add_argument("--export-video", action="store_true")
    parser.add_argument("--video-fps", type=float, default=2)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if min(args.max_steps, args.frame_window, args.max_new_tokens, args.width, args.height,
           args.frame_width, args.frame_height) < 1:
        raise SystemExit("Step, frame, token and image sizes must be positive")
    if args.format_retries < 0 or args.history_turns < 0 or args.video_fps <= 0:
        raise SystemExit("Retry/history counts must be nonnegative and video FPS positive")
    if args.discovery_rounds < 1:
        raise SystemExit("Discovery rounds must be positive")
    if args.max_input_tokens < 1:
        raise SystemExit("Input token budget must be positive")
    if args.scenario and args.completion_conditions:
        raise SystemExit("Choose scenario-generated or explicit completion conditions")
    run_auto(args)
