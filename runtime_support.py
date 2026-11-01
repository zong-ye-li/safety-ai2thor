"""Run provenance and evaluator-only completion checks."""
import hashlib
import importlib.metadata
import json
import math
import re
import subprocess
from copy import deepcopy
from pathlib import Path

NATIVE_DISPATCH_VERSION = 'native-dispatch-result-v1'
EXECUTION_PROVENANCE_VERSION = 'decision-execution-provenance-v1'

POSTPROCESSING_FAILURE_VERSION='post-execution-processing-failure-v1'
POSTPROCESSING_STAGES={'observation_capture','observation_projection','target_report',
    'operation_feedback','model_check','actor_control_report','interaction_memory',
    'task_memory','task_verification','action_record','preservation_verification','manifest_save',
    'native_observation_capture','held_appearance_verification','execution_checkpoint','physical_clock_boundary',
    'native_result_projection','native_audit_record','native_journal_record'}


class NativeDispatchError(RuntimeError):
    """IO-owner dispatch facts without a Boolean native action outcome; no replay."""
    def __init__(self, origin, command, status, cause, command_id=None, *, stage,
                 retention_error=None, control_failure=None):
        if status not in {'not_sent', 'outcome_unknown'}:
            raise ValueError('Dispatch failure cannot manufacture an acknowledged outcome')
        self.cause = cause
        self.execution = dict(success=None, error=str(cause), category=None,
            **native_dispatch_feedback(status, command_id),
            native_dispatch_error=dict(stage=stage, origin=origin, command=deepcopy(command),
                exception_type=type(cause).__name__, error=str(cause)))
        if retention_error is not None:
            self.execution['native_dispatch_error']['retention_error'] = dict(
                exception_type=type(retention_error).__name__, error=str(retention_error))
        if control_failure is not None:
            self.execution['native_dispatch_error']['control_failure'] = deepcopy(control_failure)
        super().__init__(str(cause))


class NativeJournalError(RuntimeError):
    """IO returned this endpoint; its end record is not confirmed durable.

    This is not itself a Boolean action receipt. Only the policy IO owner may
    promote a strictly Boolean endpoint, and only for the matching command.
    """
    def __init__(self, origin, command, command_id, endpoint, cause):
        self.origin = origin
        self.command = deepcopy(command)
        self.command_id = command_id
        self.endpoint = deepcopy(endpoint)
        self.cause = cause
        super().__init__('Native end-record durability unconfirmed: ' + str(cause))


class ReceivedExecutionError(RuntimeError):
    """An IO owner returned an outcome before later processing failed; never retry."""
    def __init__(self, execution, stage, cause):
        receipt = project_dispatch_feedback(execution, require_receipt=True)
        if receipt['native_dispatch_status'] != 'outcome_received':
            raise ValueError('Received execution error requires an acknowledged native outcome')
        self.execution = deepcopy(execution)
        self.stage = stage
        self.cause = cause
        self.execution['post_execution_processing_error'] = postprocessing_failure(stage, cause)
        super().__init__(str(cause))


def retain_after_failure(primary, feedback, stage, callback):
    """Attempt an evidence sink once; secondary retention must not replace primary."""
    retained = True
    try:
        callback()
    except BaseException as secondary:
        retained = False
        feedback.setdefault('failure_retention_errors', []).append(dict(stage=stage,
            exception_type=type(secondary).__name__, error=str(secondary),
            durability='unconfirmed_no_retry'))
    finally:
        primary.execution_failure_feedback = deepcopy(feedback)
        if isinstance(primary, NativeDispatchError) and 'failure_retention_errors' in feedback:
            primary.execution['failure_retention_errors'] = deepcopy(feedback['failure_retention_errors'])
    return retained


def postprocessing_failure(stage,error):
    return project_postprocessing_failure(dict(version=POSTPROCESSING_FAILURE_VERSION,
        stage=stage,error=str(error),exception_type=type(error).__name__))


def project_postprocessing_failure(value):
    if value is None:return None
    if (not isinstance(value,dict) or set(value)!={'version','stage','error','exception_type'}
            or any(not isinstance(v,str) for v in value.values())
            or value['version']!=POSTPROCESSING_FAILURE_VERSION
            or value['stage'] not in POSTPROCESSING_STAGES or not value['exception_type']):
        raise ValueError('Invalid post-execution processing failure')
    return deepcopy(value)

DISPATCH_FIELDS = ('native_dispatch_version', 'native_dispatch_status',
    'environment_command_executed', 'native_outcome_received', 'native_action_command_id')


def native_dispatch_feedback(status, command_id=None):
    """IO-owner receipt;native rejection is still a received native outcome."""
    states = {'not_sent': (False, False), 'outcome_received': (True, True),
              'outcome_unknown': (None, None)}
    if not isinstance(status, str) or status not in states or (command_id is not None and (not isinstance(command_id, str) or not command_id)):
        raise ValueError('Invalid native dispatch receipt')
    if status == 'not_sent' and command_id is not None:
        raise ValueError('An unsent action cannot have a native action command ID')
    sent, received = states[status]
    return {'native_dispatch_version': NATIVE_DISPATCH_VERSION, 'native_dispatch_status': status,
        'environment_command_executed': sent, 'native_outcome_received': received,
        'native_action_command_id': command_id}


def project_dispatch_feedback(result, *, require_receipt=False):
    """Validate/copy supplied facts only;never infer dispatch from success/audit growth."""
    if not isinstance(result, dict):
        raise ValueError('Execution feedback must be an object')
    version = result.get('native_dispatch_version')
    if require_receipt and version is None:
        raise ValueError('Current IO owner omitted its required native dispatch receipt')
    if version is not None:
        if version != NATIVE_DISPATCH_VERSION or any(key not in result for key in DISPATCH_FIELDS):
            raise ValueError('Incomplete or unsupported native dispatch receipt')
        expected = native_dispatch_feedback(result['native_dispatch_status'], result['native_action_command_id'])
        if any(type(result[key]) is not type(value) or result[key] != value for key, value in expected.items()):
            raise ValueError('Contradictory native dispatch receipt')
        timing = result.get('physical_time')
        ids = [result.get('command_id')]
        if isinstance(timing, dict): ids.append(timing.get('native_action_command_id'))
        if any(identity is not None and identity != expected['native_action_command_id'] for identity in ids):
            raise ValueError('Native dispatch receipt command identity disagrees with recorded endpoint')
        return expected
    # Older adapters/evidence retain only explicitly supplied booleans. Missing is unknown.
    values = {}
    for key in ('environment_command_executed', 'native_outcome_received'):
        value = result.get(key)
        if value is not None and type(value) is not bool:
            raise ValueError('Invalid legacy native dispatch field: ' + key)
        values[key] = value
    return values


def actor_selection_reference(action,scene,observation_id,actor_call_id):
    """Record the selected argument from this actor input; never certify intent."""
    alias=action.get('objectId',action.get('target_alias') if action.get('action')=='RequestReplan' else None)
    if not isinstance(alias,str):return None
    scene=scene if isinstance(scene,dict) else {}
    offered=next((r for r in scene.get('objects',[]) if r.get('objectId')==alias),None)
    return {'target_alias':alias,'offered_in_this_input':offered is not None,
        'object_type':offered.get('objectType') if offered is not None else None,
        'argument_role':('reported_candidate' if action.get('action')=='RequestReplan' else
                         'receiving_receptacle' if action.get('action')=='PutObject' else 'action_argument_target'),
        'held_alias_at_input':scene.get('heldObject'),'observation_id':observation_id,
        'actor_call_id':actor_call_id,'snapshot_role':'actor_input_at_recorded_call',
        'scope':'Native-assisted selected argument only;not planner intended identity,current state,execution or benefit'}


def execution_provenance(stage,source,feedback,*,before_observation_id=None,after_observation_id=None,selection=None):
    """One source contract for planner rejection, actor handling and stored history."""
    if stage not in {'planner_generation','planner_validation','actor_decoding','actor_handling'}:
        raise ValueError('Unknown decision/execution stage')
    if not isinstance(source,dict):raise ValueError('Execution provenance source must be an object')
    origin={k:deepcopy(source.get(k)) for k in ('hypothesis_id','planner_call_id','actor_call_id','intent_sha256')}
    if not isinstance(origin['planner_call_id'],str) or not origin['planner_call_id']:
        raise ValueError('Decision feedback requires its originating planner call')
    planner_stage=stage.startswith('planner_')
    if (planner_stage and origin['actor_call_id'] is not None
            or not planner_stage and not origin['actor_call_id']):
        raise ValueError('Decision stage disagrees with actor call provenance')
    if any(value is not None and (not isinstance(value,str) or not value)
           for value in (*origin.values(),before_observation_id,after_observation_id)):
        raise ValueError('Decision provenance IDs must be text or unknown')
    status=project_dispatch_feedback(feedback).get('native_dispatch_status')
    if (planner_stage or stage=='actor_decoding') and status!='not_sent':
        raise ValueError('Rejected/undecoded proposal cannot claim native dispatch')
    if selection is not None and (not isinstance(selection,dict) or stage!='actor_handling'
            or selection.get('actor_call_id')!=origin['actor_call_id']
            or selection.get('observation_id')!=before_observation_id):
        raise ValueError('Selected identity must belong to this actor input')
    return {'version':EXECUTION_PROVENANCE_VERSION,'stage':stage,'source':origin,
        'before_observation_id':before_observation_id,'after_observation_id':after_observation_id,
        'selected_argument':deepcopy(selection),
        'scope':'Controller-recorded decision stage and source only;observations are those available at feedback emission. '
            'Later operation observation records may add RGB without changing the received result. Planner generation/validation is not an actor attempt; '
            'actor decoding/report is not native execution. Read the sibling native dispatch receipt for '
            'sent/received/unknown status. Model step/outcome prose is unverified;acknowledgment is not benefit.'}


def project_execution_provenance(feedback):
    """Validate supplied provenance on every projection; legacy stays absent."""
    value=feedback.get('execution_provenance')
    if value is None:return None
    if not isinstance(value,dict) or value.get('version')!=EXECUTION_PROVENANCE_VERSION:
        raise ValueError('Unsupported execution provenance')
    expected=execution_provenance(value.get('stage'),value.get('source',{}),feedback,
        before_observation_id=value.get('before_observation_id'),after_observation_id=value.get('after_observation_id'),
        selection=value.get('selected_argument'))
    if value!=expected:raise ValueError('Contradictory execution provenance')
    return expected


def source_manifest(root):
    root = Path(root).resolve()
    files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
             for path in root.glob("*.py")}
    commit = None
    commit_source = 'no_local_repository_or_snapshot_identity'
    integrity_files = [(name, root/name) for name in ('frozen-source.json','source-integrity.json') if (root/name).is_file()]
    if integrity_files:
        runtime = dict(files)
        server = root/'tools/serve_inference.py'
        if server.is_file():runtime['tools/serve_inference.py']=hashlib.sha256(server.read_bytes()).hexdigest()
        for name,path in integrity_files:
            identity=json.loads(path.read_bytes())
            if identity.get('source_sha256')!=runtime:
                raise ValueError('Snapshot provenance runtime SHA mismatch: '+name)
        name,path=integrity_files[0]
        identity=json.loads(path.read_bytes())
        commit=identity.get('commit' if name=='frozen-source.json' else 'git_commit')
        if commit is not None and (not isinstance(commit,str) or not re.fullmatch(r'[0-9a-f]{40}',commit)):
            raise ValueError('Snapshot provenance has an invalid full commit')
        commit_source=name if commit is not None else name+'_without_full_commit'
    else:
        try:
            top = subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=root,
                                          stderr=subprocess.DEVNULL,text=True).strip()
            # A frozen directory nested inside a working checkout must never
            # inherit that parent checkout's changing HEAD.
            if Path(top).resolve()==root:
                commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,
                                               stderr=subprocess.DEVNULL,text=True).strip()
                commit_source='repository_root_HEAD'
        except (OSError,subprocess.CalledProcessError):
            pass
    versions = {}
    for name in ("ai2thor", "torch", "transformers", "accelerate", "numpy", "pillow", "opencv-python", "lm-format-enforcer", "pydantic"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"commit": commit, "commit_source":commit_source,"source_sha256": files, "packages": versions}


def verify_completion(metadata, conditions=None):
    """Truth is used for scoring only; no target or hidden state is returned to the policy."""
    if not conditions:
        if metadata.get("inventoryObjects"):
            return {"status": "incomplete", "reason": "An object is still held; no explicit holding goal was defined."}
        return {"status": "unknown", "reason": "No independent task postconditions were specified."}
    objects = {obj["objectId"]: obj for obj in metadata.get("objects", [])}
    failures = []
    for condition in conditions:
        if condition.get("field") == "inventoryEmpty":
            actual = not metadata.get("inventoryObjects", [])
        elif condition.get("field") == "onReceptacle":
            obj = objects.get(condition.get("objectId"))
            if obj is None or "parentReceptacles" not in obj:
                return {"status": "unknown", "reason": "Required placement relation is unavailable."}
            parents = obj["parentReceptacles"] or []
            if not isinstance(parents, list) or any(not isinstance(parent, str) for parent in parents):
                return {"status": "unknown", "reason": "Invalid native placement relation."}
            if condition.get("direct"):
                if any(parent not in objects or "parentReceptacles" not in objects[parent] for parent in parents):
                    return {"status": "unknown", "reason": "Direct receiver hierarchy is unavailable."}
                if any(objects[parent]["parentReceptacles"] is not None and
                       (not isinstance(objects[parent]["parentReceptacles"], list) or
                        any(not isinstance(value, str) for value in objects[parent]["parentReceptacles"])) for parent in parents):
                    return {"status": "unknown", "reason": "Invalid parent receiver hierarchy."}
                ancestors = {ancestor for parent in parents for ancestor in objects[parent]["parentReceptacles"] or []}
                direct = [parent for parent in parents if parent not in ancestors]
                if parents and not direct:
                    return {"status": "unknown", "reason": "Native receiver hierarchy has no direct candidate."}
                parents = direct
            actual = any(objects.get(parent, {}).get("objectType") == condition["objectType"] for parent in parents)
            if not actual and any(parent not in objects for parent in parents):
                return {"status": "unknown", "reason": "A parent receptacle is unavailable."}
        else:
            obj = objects.get(condition.get("objectId"))
            if obj is None or condition.get("field") not in obj:
                return {"status": "unknown", "reason": "A required object or state is unavailable."}
            actual = obj[condition["field"]]
        if actual != condition["equals"]:
            failures.append(condition)
    return {"status": "incomplete" if failures else "complete",
            "reason": "Task postconditions not met." if failures else "All specified postconditions hold.",
            "failed_conditions": failures}


def verify_preservation(baseline, metadata, allow_destructive=False):
    """Postconditions alone do not excuse new damage to the household."""
    if baseline is None:
        return {"status": "unknown", "changes": [], "reason": "No preserved task baseline."}
    before = {obj["objectId"]: obj for obj in baseline.get("objects", [])}
    changes = []
    for obj in metadata.get("objects", []):
        previous = before.get(obj["objectId"], {})
        for field in ("isBroken", "isDirty"):
            if obj.get(field) is True and previous.get(field) is not True:
                changes.append({"objectId": obj["objectId"], "field": field, "before": previous.get(field), "after": True})
    return {"status": "violation" if changes and not allow_destructive else "complete",
            "changes": changes, "destructive_tools_enabled": bool(allow_destructive),
            "reason": "New broken/dirty state requires explicitly enabled destructive tools." if changes and not allow_destructive
                      else "No prohibited broken/dirty state change detected; other effects are not certified."}


def verify_hand_motion(before, after, held_id):
    """Endpoint check for a bounded hand translation; no continuous-contact claim."""
    violations=[]; missing=[]
    old={row['objectId']:row for row in before.get('objects',[])}
    new={row['objectId']:row for row in after.get('objects',[])}
    if not old or held_id not in old or not new:
        missing.append('object metadata')
    if set(old)!=set(new): violations.append('object_set_changed')
    inventory=lambda state:[row.get('objectId') for row in state.get('inventoryObjects',[])]
    if inventory(before)!=[held_id] or inventory(after)!=[held_id]: violations.append('inventory_changed')
    if not before.get('agent') or not after.get('agent'): missing.append('camera metadata')
    elif before['agent']!=after['agent']: violations.append('camera_changed')
    fields=('isBroken','isDirty','isToggled','isOpen','isCooked','isSliced','isPickedUp','isFilledWithLiquid')
    for identity in old.keys() & new.keys():
        a,b=old[identity],new[identity]
        for field in fields:
            if type(a.get(field)) is bool and type(b.get(field)) is bool:
                if a[field]!=b[field]: violations.append({'objectId':identity,'field':field})
            elif field in a or field in b: missing.append({'objectId':identity,'field':field})
        for field in ('position','rotation'):
            if field=='position' and identity==held_id: continue
            try:
                values=[float(row[field][axis]) for row in (a,b) for axis in ('x','y','z')]
                if not all(math.isfinite(value) for value in values): raise ValueError('nonfinite pose')
                if field=='position':
                    delta=math.dist(values[:3],values[3:]); limit=0.01
                else:
                    delta=max(abs((y-x+180)%360-180) for x,y in zip(values[:3],values[3:])); limit=1
                if delta>limit: violations.append({'objectId':identity,'field':field,'delta':delta})
            except (KeyError,TypeError,ValueError): missing.append({'objectId':identity,'field':field})
        if identity!=held_id:
            if type(b.get('isMoving')) is not bool: missing.append({'objectId':identity,'field':'isMoving'})
            elif b['isMoving']: violations.append({'objectId':identity,'field':'isMoving'})
    return {'status':'violation' if violations else 'unknown' if missing else 'complete',
            'violations':violations,'missing':missing,
            'scope':'Native recorded endpoints only; not a continuous contact/collision or stability guarantee'}


def verify_task(metadata, conditions=None, baseline=None, allow_destructive=False):
    result = verify_completion(metadata, conditions)
    preservation = verify_preservation(baseline, metadata, allow_destructive)
    result["postconditions_status"] = result["status"]
    result["preservation"] = preservation
    if preservation["status"] == "violation":
        result.update(status="incomplete", reason=preservation["reason"])
    elif preservation["status"] == "unknown" and result["status"] == "complete":
        result.update(status="unknown", reason=preservation["reason"])
    return result


def load_conditions(path):
    conditions = json.loads(Path(path).read_text(encoding="utf-8"))
    fields = {"inventoryEmpty", "isOpen", "isToggled", "isPickedUp", "parentReceptacles", "onReceptacle"}
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("Completion conditions must be a nonempty JSON list")
    for condition in conditions:
        if not isinstance(condition, dict) or condition.get("field") not in fields or "equals" not in condition:
            raise ValueError("Invalid completion condition")
        if condition["field"] != "inventoryEmpty" and not isinstance(condition.get("objectId"), str):
            raise ValueError("Object-state conditions require an objectId")
        if condition["field"] == "onReceptacle" and (not isinstance(condition.get("objectType"), str)
                or not condition["objectType"] or type(condition["equals"]) is not bool):
            raise ValueError("Placement conditions require a receptacle objectType and Boolean equals")
        if "direct" in condition and (condition["field"] != "onReceptacle" or type(condition["direct"]) is not bool):
            raise ValueError("Direct placement is a Boolean option on onReceptacle only")
    return conditions
