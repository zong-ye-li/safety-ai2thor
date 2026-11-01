"""Fixed evaluator-only scenes; problem names, setup and answers never reach the model."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from main import build_parser, build_llm, run_auto
from thor_wrapper import ThorWrapper
from evidence_io import atomic_json_write, read_episode_checkpoint
from runtime_support import source_manifest
from scene_readiness import readiness_from_survey, validate_readiness_contract
from physics_time import validate_runtime_clock, assess_owned_survey
from inference_rpc import RPCError, RemoteInferenceError


class BorrowedEnvironment:
    def __init__(self, owner): object.__setattr__(self,'owner',owner)
    def __getattr__(self, name): return getattr(self.owner, name)
    def __setattr__(self, name, value): setattr(self.owner,name,value)
    def close(self): pass


def retained_exhausted_model_case(error, client, episode):
    """Permit a different case only after a fully recorded model-only failure."""
    stats = getattr(client, 'last_call_stats', {})
    transport = getattr(client, 'transport', None)
    attempts = stats.get('model_attempts', [])
    if (not isinstance(error, RemoteInferenceError) or not error.retryable
            or transport is None or transport.closed
            or stats.get('model_retry_count') != 2 or len(attempts) != 3
            or stats.get('remote_error') != error.remote_error
            or getattr(error, 'execution_failure_feedback', {}).get('failure_retention_errors')):
        return None
    # An error flag alone is insufficient: every fresh attempt must have a
    # completed generation and an individually identified received response.
    identities = []
    for attempt in attempts:
        if attempt.get('status') != 'remote_error':
            return None
        try:
            reply = RemoteInferenceError(attempt.get('error'), attempt.get('transport'))
        except RPCError:
            return None
        if not reply.retryable or not isinstance(reply.transport, dict):
            return None
        identities.append(reply.transport.get('request_id'))
    if any(not isinstance(value, str) or not value for value in identities) or len(set(identities)) != 3:
        return None
    manifest, result = read_episode_checkpoint(episode)
    if (manifest.get('status') != 'runtime_error'
            or manifest.get('error_type') != 'RemoteInferenceError'
            or manifest.get('last_model_call_stats') != stats
            or manifest.get('failed_native_dispatch')):
        return None
    # Confirm the failed call itself reached the evidence sink; no stale
    # checkpoint or cleanup failure can silently become a recoverable case.
    last = None
    with (Path(episode)/'model_calls.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            last = json.loads(line)
    if last is None or last.get('kind') != 'call_error' or last.get('stats') != stats:
        return None
    return manifest, result


def assess_fixture(fixture, snapshot):
    """Evaluator state/setup validity only; never establish visual problem quality."""
    objects={o['objectId']:o for o in snapshot.get('objects',[])}
    finals=[]
    for state in fixture['states']:
        identity=state.get('measurement_object_id')
        actual=objects.get(identity,{}).get(state['field'])
        finals.append({'measurement_object_id':identity,'field':state['field'],
                       'expected':state['value'],'actual':actual,
                       'established':type(actual) is bool and actual is state['value']})
    setup_ok=all(a['success'] for a in fixture['actions'])
    states_ok=all(s.get('established',False) and s.get('native_success',False) for s in fixture['states'])
    final_ok=all(s['established'] for s in finals)
    heading = fixture.get('heading')
    heading_ok = True
    actual_yaw = snapshot.get('agent', {}).get('rotation', {}).get('y')
    if heading and heading['mode'] == 'absolute_fixed':
        heading_ok = (heading['established'] is True and not isinstance(actual_yaw, bool)
            and isinstance(actual_yaw, (int, float)) and math.isfinite(actual_yaw)
            and abs((actual_yaw - heading['target'] + 180) % 360 - 180) <= 1e-3)
    nearby=None
    if fixture.get('nearby_placement') is not None:
        from validation.scripts.nearby_safety_fixture import assess_nearby
        nearby=assess_nearby(fixture['nearby_placement'],snapshot)
    return {'valid':setup_ok and states_ok and final_ok and heading_ok and (nearby is None or nearby['valid']),
            'nearby_placement':nearby,'all_setup_actions_succeeded':setup_ok,
            'heading_established':heading_ok,'actual_yaw':actual_yaw,
            'initial_state_operations_established':states_ok,'post_settle_states_established':final_ok,
            'final_states':finals,'scope':'Evaluator native configuration only; not problem reasonableness or RGB sufficiency'}


def assess_live_fixture(fixture, snapshot, survey_audit, expected_scan, readiness_contract=None):
    """Recheck the actual episode before any model call, never provide answers to it."""
    assessment = assess_fixture(fixture, snapshot)
    survey_valid = (len(survey_audit) == expected_scan and all(
        row.get('success') is True and row.get('action') == {'action':'RotateRight','degrees':90}
        for row in survey_audit))
    readiness = readiness_from_survey(survey_audit, snapshot, readiness_contract)
    readiness_valid = readiness_contract is None or readiness['status'] == 'ready'
    return {'valid':assessment['valid'] and survey_valid and readiness_valid, 'post_survey_assessment':assessment,
            'scene_readiness':readiness,
            'expected_scan_steps':expected_scan, 'actual_scan_steps':len(survey_audit),
            'survey_valid':survey_valid,
            'scope':'Actual evaluator fixture and fixed survey only; not visual problem quality or policy success'}


def configure(env, case, stove_reference=None):
    """Evaluation fixture only, absent from discovery requests and model messages."""
    target_yaw = case.get('starting_yaw')
    if 'starting_yaw' in case:
        if (isinstance(target_yaw, bool) or not isinstance(target_yaw, (int, float))
                or not math.isfinite(target_yaw) or not 0 <= target_yaw < 360 or case.get('yaw', 0) != 0):
            raise ValueError('Explicit starting_yaw requires finite 0<=yaw<360 and no relative yaw offset')
        before_yaw = env.snapshot().get('agent', {}).get('rotation', {}).get('y')
        if (isinstance(before_yaw, bool) or not isinstance(before_yaw, (int, float)) or not math.isfinite(before_yaw)):
            raise ValueError('Actual own yaw unavailable for fixed heading setup')
    actions = []
    def step(**command):
        event = env.native_step("evaluator_setup", **command)
        env._capture()
        actions.append({"command": command, "command_id": env.last_command_id,
                       "success": bool(event.metadata["lastActionSuccess"]), "error": event.metadata.get("errorMessage", "")})
        return event
    # Controls keep ordinary scene contents. Turn off resource devices and close
    # openable objects without switching room illumination off.
    externally_controlled = {identity for c in (stove_reference or {}).get("controls", [])
                             for identity in c["controlled_ids"]}
    for obj in list(env.snapshot()["objects"]):
        if obj["objectId"] in externally_controlled:
            continue  # Its reference controller is normalized in the same setup loop.
        if obj.get("toggleable") and obj.get("isToggled") and obj["objectType"] != "LightSwitch":
            step(action="ToggleObjectOff", objectId=obj["objectId"], forceAction=True)
        if obj.get("openable") and obj.get("isOpen"):
            step(action="CloseObject", objectId=obj["objectId"], forceAction=True)
    conditions = []
    states = []
    for state in case["states"]:
        matches = sorted((obj for obj in env.snapshot()["objects"] if obj["objectType"] == state["type"]), key=lambda obj:obj["objectId"])
        if not matches:
            states.append({**state, "status": "fixture_type_unavailable"})
            continue
        obj = matches[0]
        controller = obj["objectId"]
        if state["type"] == "StoveBurner" and stove_reference is not None:
            if stove_reference.get("scene") != case["scene"] or stove_reference.get("status") != "verified_control_object_reference":
                raise ValueError("Stove reference scene/status mismatch")
            controls = [c for c in stove_reference["controls"] if c["controller_off_on_off_verified"] and obj["objectId"] in c["controlled_ids"]]
            if len(controls) != 1:
                raise ValueError("Stove reference must identify exactly one verified controller")
            controller = controls[0]["controller_id"]
        action = {"isOpen": "OpenObject", "isToggled": "ToggleObjectOn"}[state["field"]]
        event = step(action=action, objectId=controller, forceAction=True)
        actual = next(row for row in event.metadata["objects"] if row["objectId"] == controller)
        states.append({**state, "native_id": obj["objectId"], "controller_id":controller,
                       "measurement_object_id":controller,"scope":"Controller state; controlled visual effect requires separate review" if controller!=obj["objectId"] else "Direct object state",
                       "established": actual.get(state["field"]) is state["value"],
                       "native_success": bool(event.metadata["lastActionSuccess"])})
        conditions.append({"objectId": controller, "field": state["field"], "equals": False})
    nearby=None
    if 'start_near' in case:
        from validation.scripts.nearby_safety_fixture import place_nearby
        nearby=place_nearby(env,case,step)
    elif target_yaw is not None:
        delta = (target_yaw - before_yaw) % 360
        if min(delta, 360-delta) > 1e-3:
            step(action='RotateRight', degrees=delta)
    elif case["yaw"]:
        step(action="RotateRight", degrees=case["yaw"])
    for _ in range(3): step(action="Pass")
    after_yaw = env.snapshot().get('agent', {}).get('rotation', {}).get('y')
    heading = {'mode':'relative_offset','offset':case.get('yaw',0)}
    if nearby is not None:
        heading={'mode':'nearby_'+nearby['camera_mode'],'target':nearby['requested_yaw'],
                 'scope':nearby['scope']}
    if target_yaw is not None:
        established = (not isinstance(after_yaw, bool) and isinstance(after_yaw, (int,float))
            and math.isfinite(after_yaw) and abs((after_yaw-target_yaw+180)%360-180)<=1e-3)
        heading = {'mode':'absolute_fixed','target':target_yaw,'before':before_yaw,
                   'after':after_yaw,'established':established,
                   'scope':'Evaluator fixed own-heading only; independent of objects, issue and remedy'}
    env._frame_buffer=[]; env.frame_info=[]; env.audit=[]; env._capture()
    return {"source": ("Evaluator nearby setup; camera intervention is declared in nearby_placement"
                       if nearby is not None else
                       "evaluator-only ordinary-object/device-state fixture; fixed spawn/yaw, no target-driven camera/ROI"),
            "case": case, "states": states, "actions": actions, "heading": heading,
            "nearby_placement":nearby,
            "limitations": "Native active/open is a test manipulation, not proof of a reasonable hazard/waste diagnosis"}, conditions


def main():
    parser=build_parser()
    parser.description=__doc__
    parser.add_argument("--cases",type=Path,default=ROOT/"examples/cases.json")
    parser.add_argument("--stove-reference",type=Path,help="Independent evaluator-only native control reference")
    args=parser.parse_args()
    if args.llm not in {"huggingface","ssh"} or not args.output_dir or args.task or args.scenario or args.completion_conditions:
        parser.error("Use HF or explicit SSH inference, a new output root and evaluator case file; no supplied task/answer")
    if args.llm=='ssh' and (args.editor_port is None or args.platform!='default'):
        parser.error('SSH study requires an explicitly owned local player endpoint')
    if args.ai2thor_source:sys.path.insert(0,str(Path(args.ai2thor_source).resolve()))
    cases_content=args.cases.read_bytes(); case_document=json.loads(cases_content); cases=case_document['cases']
    from validation.scripts.public_fixture_contract import validate_recovery_memory_entry
    validate_recovery_memory_entry(cases,root=ROOT)
    readiness_contract = (validate_readiness_contract(case_document['scene_readiness'])
                          if 'scene_readiness' in case_document else None)
    clock_contract = (validate_runtime_clock(case_document['physical_time'])
                      if 'physical_time' in case_document else None)
    if clock_contract is not None and readiness_contract is not None:
        raise ValueError('Separate versioned clock contract cannot overwrite readiness-v1')
    if args.physics_clock in {'owned-v1','owned-v2'} and clock_contract is None:
        raise ValueError('Owned clock research requires an explicit frozen case-file contract')
    if clock_contract is not None:
        args.physics_clock = 'owned-v2' if clock_contract['version']=='owned-clock-runtime-v2' else 'owned-v1'
    stove_reference=json.loads(args.stove_reference.read_text()) if args.stove_reference else None
    output=Path(args.output_dir).resolve(); output.mkdir(parents=True,exist_ok=False)
    report={**source_manifest(ROOT),"type":__doc__,"case_file_sha256":hashlib.sha256(cases_content).hexdigest(),
            "status":"starting","cases":[]}
    report['scene_readiness_contract'] = readiness_contract
    report['physical_time_contract'] = clock_contract
    started=time.monotonic(); env=client=None
    report["stove_reference_sha256"]=hashlib.sha256(args.stove_reference.read_bytes()).hexdigest() if args.stove_reference else None
    def save():
        report["elapsed_seconds"]=time.monotonic()-started
        atomic_json_write(output/"open-discovery-summary.json",report)
    try:
        save()
        env=ThorWrapper(scene=cases[0]["scene"],width=args.width,height=args.height,render_depth=True,
                        render_instance_seg=True,platform=args.platform,editor_port=args.editor_port,
                        local_executable_path=args.local_executable,command_journal=output/"initialization-commands.jsonl")
        report['initial_native_success']=env.snapshot().get('lastActionSuccess') is True
        save()
        if not report['initial_native_success']:
            raise ValueError('Actual initialization failed; no model load or retry')
        for case in cases:
            case_args=argparse.Namespace(**vars(args)); case_args.scene=case["scene"]
            case_args.initial_scan=case["initial_scan"]; case_args.output_dir=str(output/case["id"])
            row={"id":case["id"],"evaluator_case":case,"status":"setup"}; report["cases"].append(row); save()
            print("START_OPEN_CASE",case["id"],flush=True)
            def borrow(**options):
                env.set_command_journal(options["command_journal"]); env.reset(case["scene"])
                row['reset_native_success']=env.snapshot().get('lastActionSuccess') is True
                save()
                if not row['reset_native_success']:
                    row.update(status='fixture_invalid_reset',fixture_valid=False); save()
                    raise ValueError('Actual reset fixture invalid; no model call or retry')
                fixture,conditions=configure(env,case,stove_reference)
                row["evaluator_fixture"]=fixture
                row["fixture_assessment"]=assess_fixture(fixture,env.snapshot())
                # Main's evaluator-only conditions do not select an issue or stop discovery early.
                if conditions:
                    path=output/(case["id"]+"-evaluator-conditions.json")
                    atomic_json_write(path,conditions)
                    # run_auto loaded args before borrowing: conditions are instead
                    # scored separately below, never injected into its model path.
                row["evaluator_conditions"]=conditions
                atomic_json_write(output/(case["id"]+"-fixture.json"),fixture)
                row['fixture_valid']=row['fixture_assessment']['valid']
                save()
                if not row['fixture_valid']:
                    row['status']='fixture_invalid_before_survey'; save()
                    raise ValueError('Actual fixture invalid; retained setup, no model call or retry')
                return BorrowedEnvironment(env)
            def checked_client(options):
                nonlocal client
                row['live_fixture_assessment']=assess_live_fixture(
                    row['evaluator_fixture'],env.snapshot(),env.audit,case['initial_scan'],readiness_contract)
                row['fixture_valid']=row['live_fixture_assessment']['valid']
                if clock_contract is not None:
                    row['owned_clock_readiness'] = assess_owned_survey(env, env.audit,
                        expected_version=clock_contract['version'])
                    row['fixture_valid'] = row['fixture_valid'] and row['owned_clock_readiness']['status'] == 'sampled_ready'
                    row['live_fixture_assessment']['valid'] = row['fixture_valid']
                    atomic_json_write(output/(case['id']+'-pre-model-clock.json'), row['owned_clock_readiness'])
                # Preserve the actual pre-model state independently of the later
                # episode state; this evaluator data never enters model requests.
                atomic_json_write(output/(case['id']+'-pre-model-state.json'),env.snapshot())
                atomic_json_write(output/(case['id']+'-pre-model-readiness.json'),
                                  row['live_fixture_assessment']['scene_readiness'])
                save()
                if not row['fixture_valid']:
                    row['status']='fixture_invalid_after_survey'; save()
                    raise ValueError('Pre-model evaluator checks refused; no model call or retry. '
                                     'See live_fixture_assessment and owned_clock_readiness in the saved summary: '
                                     + json.dumps({'fixture': row['live_fixture_assessment'].get('post_survey_assessment', {}).get('valid'),
                                                   'survey': row['live_fixture_assessment'].get('survey_valid'),
                                                   'clock': row.get('owned_clock_readiness', {}).get('status'),
                                                   'expected_clock_version': clock_contract['version'] if clock_contract else None,
                                                   'actual_clock_version': (row.get('owned_clock_readiness', {}).get('clock') or {}).get('version'),
                                                   'clock_geometry': row.get('owned_clock_readiness', {}).get('survey_geometry', {}).get('status'),
                                                   'clock_missing': row.get('owned_clock_readiness', {}).get('survey_geometry', {}).get('missing', [])}))
                if client is None:
                    client=build_llm(options)
                    report['runtime']=client.runtime_info
                client.reset_history()
                return client
            try:
                episode=run_auto(case_args,env_factory=borrow,llm_factory=checked_client)
            except RemoteInferenceError as exc:
                try:
                    retained = retained_exhausted_model_case(exc, client, case_args.output_dir)
                except (OSError, ValueError, RuntimeError) as validation_error:
                    row['case_continuation_validation_error'] = repr(validation_error)
                    retained = None
                if retained is None:
                    raise
                manifest, result = retained
                from runtime_support import verify_task
                row.update(status='model_response_error_exhausted', error=repr(exc),
                    error_type=type(exc).__name__, discovery=manifest.get('open_discovery'),
                    model_failure=deepcopy(client.last_call_stats),
                    failed_call_native_dispatch_status='not_sent',
                    fixture_valid=row['live_fixture_assessment']['valid'],
                    evaluator_completion=verify_task(env.snapshot(),row['evaluator_conditions'] or None,manifest['task_initial_state']),
                    native_action_count=len(env.audit),
                    continuation='Next independent case uses its own reset and fixture; this case is not resumed')
                save(); print('END_OPEN_CASE',case['id'],row['status'],flush=True)
                continue
            manifest,result=read_episode_checkpoint(episode)
            from runtime_support import verify_task
            # Use final native state only after the discovery loop, with no feedback to it.
            row.update(status=manifest["status"],discovery=manifest.get("open_discovery"),
                       fixture_valid=row['live_fixture_assessment']['valid'],fixture_assessment=row["fixture_assessment"],
                       evaluator_completion=verify_task(env.snapshot(),row["evaluator_conditions"] or None,manifest["task_initial_state"]),
                       native_action_count=len(env.audit),model_goal_completion=[round_.get("goal_verification") for round_ in manifest.get("open_discovery",{}).get("rounds",[])])
            save(); print("END_OPEN_CASE",case["id"],row["status"],flush=True)
        report["status"] = ('complete_with_case_errors' if any(
            row['status'] == 'model_response_error_exhausted' for row in report['cases']) else 'complete')
    except BaseException as exc:
        if report['cases'] and report['cases'][-1].get('status') == 'setup':
            report['cases'][-1].update(status='episode_error', error=repr(exc))
        report.update(status="error",error=repr(exc),traceback=traceback.format_exc()); raise
    finally:
        try:
            if client is not None: client.close()
        finally:
            if env is not None: env.close()
            save()


if __name__=="__main__": main()
