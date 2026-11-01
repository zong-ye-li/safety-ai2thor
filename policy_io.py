"""Unambiguous policy input: task, observation and execution results have separate roles."""
import json
from copy import deepcopy
from input_references import enabled as shared_input_enabled,encode_input_packet,shared_input_text


WORKING_CONTEXT_VERSION='task-working-context-v2'


def working_context(request, *, actor=False):
    """Retain supplied observations and operations; exact sharing owns deduplication.

    Input-size reduction is not permission to remove known environmental facts.
    Rejected model drafts stay labeled as unexecuted claims, never action results.
    """
    packet=deepcopy(request)
    previous=packet.get('previous_execution')
    if isinstance(previous,dict):
        rejected=previous.get('rejected_proposal')
        if isinstance(rejected,dict):
            previous['rejected_draft']={
                'hypothesis_id':rejected.get('hypothesis_id'),
                'source_call_id':previous.get('proposal_call_id'),
                'status':'rejected_not_executed',
                'scope':'The retained draft is a model claim rejected before execution, not observed progress.'}
        if previous.get('environment_command_executed') is False:
            previous['event_scope']='No environment command executed in THIS event. Its success/failure is not a result for an earlier body action.'
        elif previous.get('environment_command_executed') is True:
            previous['event_scope']=('Result received for this event only; model progress claims remain unverified.'
                if previous.get('native_outcome_received') is True else
                'Command may have effects but its outcome is unknown; do not repeat it automatically.')
    if actor:
        packet['decision_discipline']=(
            'Decide only the next safe step from current observations and the planner intent. '
            'Do not rederive the whole task or repeatedly debate settled alternatives. '
            'An authorized observation must serve this current local objective, not a later plan step. '
            'If a necessary identity, location or prerequisite is unresolved, obtain supported evidence '
            'within this objective or report it with RequestReplan; never assume it away to act. '
            'If you judge the objective already satisfied or in conflict with current evidence, '
            'report that judgment to the planner rather than silently advancing the plan. '
            'Planner/reviewer claims and agreement are not observed facts. Check the actual evidence '
            'yourself and report disagreement rather than conforming to another role.')
        # The supplied explicit history is the working-memory owner. Do not add
        # another conversation-history copy; keep every already supplied fact.
        packet['independent_observation_context']=True
    packet['working_context']={'version':WORKING_CONTEXT_VERSION,
        'contents':'Supplied current and historical observations, task claims, operation records, outcomes and source references are retained. Claims remain unverified. Repeated equal values may use exact shared references.',
        'history':'Operation history is resident when enabled; its coverage is explicit. Older RGB and task text remain available through their existing retrieval interfaces. Raw model thinking is archived for review, not promised as model-retrievable task history.'}
    return packet


def compact_native_observation(observation):
    """Factor exactly equal observed fields across current objects, without defaults."""
    packet=deepcopy(observation)
    objects=packet.get('objects',[])
    if not isinstance(objects,list) or len(objects)<2:return packet
    if 'object_common_fields' in packet:raise ValueError('Reserved observation sharing field')
    def leaves(row,path=()):
        if isinstance(row,dict):
            for key,value in row.items():yield from leaves(value,path+(key,))
        else:yield path,row
    from input_references import canonical
    fields=dict(leaves(objects[0]));common=[]
    for path,value in fields.items():
        if path==('objectId',):continue
        equal=True
        for obj in objects[1:]:
            node=obj
            for key in path:
                if not isinstance(node,dict) or key not in node:equal=False;break
                node=node[key]
            if not equal or canonical(node)!=canonical(value):equal=False;break
        if equal:common.append((path,value))
    if not common:return packet
    for obj in objects:
        for path,_ in common:
            node=obj
            for key in path[:-1]:node=node[key]
            del node[path[-1]]
    packet['object_common_fields']=[{'path':list(path),'value':deepcopy(value)} for path,value in common]
    packet['object_field_sharing']='Each common field applies literally to EVERY listed current object. Missing fields outside this table stay unknown; no inferred defaults. Paths are relative to each object.'
    return packet


def expand_native_observation(observation):
    packet=deepcopy(observation)
    common=packet.pop('object_common_fields',[])
    if common:packet.pop('object_field_sharing')
    for obj in packet.get('objects',[]):
        for entry in common:
            node=obj
            for key in entry['path'][:-1]:node=node[key]
            key=entry['path'][-1]
            if key in node:raise ValueError('Common native observation field conflicts')
            node[key]=deepcopy(entry['value'])
    return packet


def actor_operation_scene(scene, native_objects=(), instance_masks=None, alias_for=None):
    """Current rendered native-assisted identities,without native range pruning.

    Never expose objects without current pixels or existing visible candidacy.
    This packet is ACT-only;RGB observe continues to use its privacy projection.
    """
    import numpy as np
    packet=deepcopy(scene)
    packet['operation_access']='capability-feasibility-v1'
    existing={row['objectId']:row for row in packet.get('objects',[])}
    flags=('pickupable','isPickedUp','receptacle','openable','isOpen','toggleable','isToggled',
           'sliceable','isSliced','cookable','isCooked','dirtyable','isDirty','breakable','isBroken',
           'canFillWithLiquid','isFilledWithLiquid','isInteractable')
    if instance_masks is not None and alias_for is not None:
        for obj in native_objects:
            mask=instance_masks.get(obj['objectId'])
            if mask is None:continue
            ys,xs=np.nonzero(mask)
            if not len(xs):continue
            alias=alias_for(obj['objectId'])
            row=existing.setdefault(alias,{'objectId':alias,'objectType':obj.get('objectType')})
            row.update({key:deepcopy(obj[key]) for key in flags if key in obj})
            row['nativeVisible']=obj.get('visible')
            if 'distance' in obj:row['distance']=round(float(obj['distance']),2)
            row['rendered_evidence']={'source':'current_instance_mask_native_assistance',
                'pixel_bounds_xyxy':[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)],
                'image_size_wh':[int(mask.shape[1]),int(mask.shape[0])],
                'scope':'Current rendered identity/extent only;not model recognition,intended target match,reachability or useful action'}
    packet['objects']=list(existing.values())
    packet['informationSource']='Current rendered/native-visible candidates and native affordances;native assistance,not pure RGB. Invocation authorization does not certify physical feasibility.'
    return packet


def image_identity_labels(request, image_count, key):
    """Repeat existing identity mapping only; never add visual interpretations."""
    if not isinstance(request,dict) or key not in request:
        return None
    views=request[key]
    from image_slots import enabled as shared_images,groups
    if shared_images(request):
        labels=[]
        for index,bindings in enumerate(groups(request,image_count,key)):
            text='RGB image_index='+str(index)+' shared_exact_raster; '
            for view in bindings:
                text+='observation_id='+json.dumps(view['observation_id'])+' current='+json.dumps(view.get('current',False))
                if 'visual_inspection' in view:
                    region=view['visual_inspection']
                    text+=' recorded_region_of='+json.dumps(region['source_observation_id'])+' bbox_xyxy='+json.dumps(region['bbox_xyxy'])+' NOT_A_NEW_CAPTURE'
                text+='; '
            labels.append(text)
        return labels
    if not isinstance(views,list) or len(views)!=image_count:
        raise ValueError('Image identity mapping must match actual image count')
    labels=[]
    for index,view in enumerate(views):
        if (not isinstance(view,dict) or type(view.get('image_index')) is not int
                or view['image_index']!=index or not isinstance(view.get('observation_id'),str)
                or not 1<=len(view['observation_id'])<=32):
            raise ValueError('Image identity mapping requires ordered index and observation_id')
        label='RGB image_index='+str(index)+' observation_id='+json.dumps(view['observation_id'])
        if 'visual_inspection' in view:
            region=view['visual_inspection']
            label+=' recorded_region_of='+json.dumps(region['source_observation_id'])+' bbox_xyxy='+json.dumps(region['bbox_xyxy'])+' NOT_A_NEW_CAPTURE'
        labels.append(label)
    return labels


def planner_authored_task_provenance(hypothesis_id, proposal_call_id, source_round):
    """Label actual planner authorship; never upgrade a goal to a human request."""
    if (not isinstance(hypothesis_id, str) or not hypothesis_id
            or not isinstance(proposal_call_id, str) or not proposal_call_id
            or type(source_round) is not int or source_round < 0):
        raise ValueError('Planner task provenance requires task/call/round identities')
    return {'source': 'model_authored_planner_proposal', 'hypothesis_id': hypothesis_id,
            'proposal_call_id': proposal_call_id, 'source_round': source_round,
            'assessment': 'unverified_model_objective',
            'scope': ('The task is the planner model\'s own proposed next objective, not a new human '
                      'instruction, system-certified target or observed fact. The actor may report '
                      'uncertainty; only the planner revises this objective.')}


def compact_check_feedback(request):
    """Lossless current-message reference; caller/report/history stay complete."""
    packet=deepcopy(request)
    # One reference protocol owns the delivered input. Do not replace a feedback
    # body with legacy same_as before the typed shared-input encoder can own it.
    if shared_input_enabled(packet):return packet
    previous=packet.get('previous_execution')
    prior=packet.get('hypothesis',{}).get('previous_execution',{}).get('feedback',{})
    if isinstance(previous,dict) and 'model_check_progress' in previous and previous==prior:
        packet['previous_execution']={'same_as':'hypothesis.previous_execution.feedback'}
    return packet


def execution_result(row):
    success=row.get('success')
    if success is not None and type(success) is not bool:raise ValueError('Execution success must be Boolean or unknown')
    result = {"origin": row["kind"],
              "category": row.get("category"),
              "action": row.get("action", {}), "success": success, "message": row.get("message",row.get("error", ""))}
    from runtime_support import project_dispatch_feedback,project_execution_provenance
    result.update(project_dispatch_feedback(row))
    from runtime_support import project_postprocessing_failure
    failure=project_postprocessing_failure(row.get('post_execution_processing_error'))
    if failure is not None:result['post_execution_processing_error']=failure
    if 'post_execution_observation_error' in row:
        if not isinstance(row['post_execution_observation_error'],str):
            raise ValueError('Observation error must remain text')
        result['post_execution_observation_error']=row['post_execution_observation_error']
    provenance=project_execution_provenance(row)
    if provenance is not None:result['execution_provenance']=provenance
    if row.get('actor_selection_report') is not None:
        from observation_memory import project_actor_selection
        result['actor_selection_report']=project_actor_selection(row['actor_selection_report'],
            source=provenance['source'] if provenance is not None else None,
            before_observation_id=provenance['before_observation_id'] if provenance is not None else None)
    if row.get("constraint"):
        result["constraint"] = row["constraint"]
    if row.get("effect"):
        result["effect"] = row["effect"]
    if row.get('physical_time'):
        result['physical_time'] = deepcopy(row['physical_time'])
    return result


def observation_packet(scene):
    if isinstance(scene, str):
        try:
            packet = json.loads(scene)
        except ValueError:
            packet = {"description": scene}
    else:
        packet = dict(scene)
    if not isinstance(packet, dict):
        raise ValueError("Policy observation must be an object")
    packet["recentFrames"] = [
        {key: view[key] for key in ("frame", "yaw", "horizon", "physical_time") if key in view}
        for view in packet.get("recentFrames", [])]
    packet["scope"] = "Current candidates describe the latest event; views are observations, never pending commands."
    return packet


def policy_request(task, previous=None, format_error=None, completion_managed=False, execution_history=None,
                   allow_destructive=False, task_contract=None, tool_scope="affordance"):
    request = {"task": task}
    request["completion_managed_by_controller"] = completion_managed
    request["allow_destructive_tools"] = bool(allow_destructive)
    if task_contract is not None:
        request["task_contract"] = task_contract
        request["tool_scope"] = tool_scope
    if execution_history is not None:
        request["execution_history"] = [execution_result(row)
            for row in execution_history if row["kind"] == "policy"]
    if previous is not None:
        request["previous_execution"] = execution_result(previous)
    if format_error:
        request["formatting_correction"] = {
            "error": format_error,
            "physical_state_changed": False,
            "instruction": "Return one valid action object. Preserve previous execution facts; omit objectId for non-object actions.",
        }
    return request


def actor_input_packet(request, observation):
    task_request = compact_check_feedback(request) if isinstance(request, dict) else {"task": request}
    # Legacy JSON clients already maintain their own bounded dialogue history.
    task_request = {key: value for key, value in task_request.items() if key != "execution_history"}
    scene=observation_packet(observation)
    if task_request.get('working_context',{}).get('version')==WORKING_CONTEXT_VERSION:
        scene['candidate_delivery']={
            'count':len(scene.get('objects',[])),
            'table':'observation.objects','identifier_field':'objectId',
            'scope':'These are the currently supplied candidates. A missing intended target is different from an absent table. Only the controller assigns aliases from observed environment identities; no role may invent one.'}
        if task_request.get('actor_tool_scope',{}).get('input_mode')=='recorded_rgb_own_body':
            scene['candidate_delivery']['scope']='RGB observation mode supplies no new native object/alias table or alias lookup result. This does not establish whether the intended object exists.'
        # The actual RGB bindings are image_views. A second native recentFrames
        # chronology is neither another image delivery nor current observation.
        scene.pop('recentFrames',None)
        compact=compact_native_observation(scene)
        from input_references import canonical
        if canonical(expand_native_observation(compact))!=canonical(scene):
            raise ValueError('Working observation changed native facts')
        if len(canonical(compact))<len(canonical(scene)):scene=compact
    return {**task_request,"observation":scene}


def model_input_projection(request,observation=None):
    packet=actor_input_packet(request,observation) if observation is not None else request
    return encode_input_packet(packet)


def policy_text(request, observation):
    packet=actor_input_packet(request,observation)
    if shared_input_enabled(packet):return shared_input_text(packet)[0]
    return json.dumps(packet,ensure_ascii=False,separators=(',',':'))
