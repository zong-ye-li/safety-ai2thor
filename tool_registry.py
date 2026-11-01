"""Observable affordances define tool availability; task answers are never consulted."""
import json
import math
from copy import deepcopy
from action_protocol import (ALLOWED_ACTIONS, OBJECT_ACTIONS, ProtocolError, validate_action,
                             HAND_AXES, HAND_MOVE_MAX_METRES, HAND_MOVE_MAX_ATTEMPTS,SELECTION_REPORT_SCHEMA)
from action_catalog import model_guide, action_interface_contract, literal_capability_semantics_match


def active_task_contract(request):
    if isinstance(request, dict) and request.get("tool_scope") == "task":
        return request.get("task_contract")
    return None


def available_tools(scene, completion_managed=False, allow_destructive=False, task_contract=None):
    capability_access = scene.get('operation_access') == 'capability-feasibility-v1' and task_contract is None
    objects = scene.get("objects", [])
    held = bool(scene.get("heldObject"))
    destination_type = None
    state_contract = None
    state_action = None
    if task_contract is not None:
        if isinstance(task_contract, dict) and task_contract.get("intent") == "set_state":
            if (set(task_contract) != {"intent", "target_alias", "object_type", "field", "equals"}
                    or type(task_contract["equals"]) is not bool or not isinstance(task_contract["target_alias"], str)):
                raise ProtocolError("Unsupported explicit state-restoration contract")
            # Explicit task restriction uses generic action mechanics, never an issue/type catalog.
            transitions = {("isOpen", True): "OpenObject", ("isOpen", False): "CloseObject",
                           ("isToggled", True): "ToggleObjectOn", ("isToggled", False): "ToggleObjectOff"}
            state_action = transitions.get((task_contract["field"], task_contract["equals"]))
            if state_action is None:
                raise ProtocolError("Unsupported explicit state-restoration predicate")
            state_contract = task_contract
        elif (not isinstance(task_contract, dict) or set(task_contract) != {"intent", "destination_type", "relation"}
                or task_contract["intent"] != "place_held" or task_contract["relation"] != "direct"
                or not isinstance(task_contract["destination_type"], str) or not task_contract["destination_type"].isidentifier()):
            raise ProtocolError("Unsupported explicit task contract")
        else:
            destination_type = task_contract["destination_type"]
    incompatible_receivers = {row.get("targetAlias") for row in scene.get("affordanceFeedback", [])
                              if row.get("source") == "observed_native_rejection"
                              and row.get("category") == "incompatible_target" and row.get("action") == "PutObject"
                              and row.get("heldAlias") == scene.get("heldObject")}
    tools = {}
    for name in sorted(ALLOWED_ACTIONS - OBJECT_ACTIONS):
        if name == 'RequestReplan' and scene.get('allow_planning_feedback') is not True:
            continue
        if name == "ThrowObject" and not allow_destructive:
            continue
        if name == "Done" and completion_managed:
            continue
        if name in {"DropHandObject", "ThrowObject"} and not held and not capability_access:
            continue
        if name == "MoveHeldObject" and not capability_access and (not held or scene.get("handMotion", {}).get("remaining_attempts", HAND_MOVE_MAX_ATTEMPTS) <= 0):
            continue
        if state_contract and name in {"DropHandObject", "ThrowObject"}:
            continue
        properties = {"action": {"type": "string", "enum": [name]}}
        if name in {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight", "ThrowObject"}:
            properties["moveMagnitude"] = {"type": "number", "exclusiveMinimum": 0, "maximum": 5}
        if name in {"RotateLeft", "RotateRight", "LookUp", "LookDown"}:
            properties["degrees"] = {"type": "number", "exclusiveMinimum": 0,
                                      "maximum": 90 if name in {"LookUp", "LookDown"} else 360}
        required = ["action"]
        if name == 'RequestReplan':
            properties['reason'] = {'type': 'string', 'minLength': 1, 'maxLength': 512}
            required.append('reason')
            if scene.get('allow_target_reports') is True:
                report_aliases = sorted({obj['objectId'] for obj in objects
                    if isinstance(obj.get('objectType'), str)})
                if report_aliases:
                    properties['target_alias'] = {'type': 'string', 'enum': report_aliases}
        if name == "MoveHeldObject":
            properties.update({key:{"type":"number","minimum":-HAND_MOVE_MAX_METRES,"maximum":HAND_MOVE_MAX_METRES} for key in HAND_AXES})
            required.extend(HAND_AXES)
        if name in {"LookUp", "LookDown"} and scene.get("camera") and not capability_access:
            camera = scene["camera"]
            remaining = (camera["horizon"] - camera["min_horizon"] if name == "LookUp"
                         else camera["max_horizon"] - camera["horizon"])
            maximum = min(90, round(remaining, 1))
            if maximum < camera["degree_increment"]:
                continue
            properties["degrees"].update(minimum=camera["degree_increment"], maximum=maximum, multipleOf=camera["degree_increment"])
            if maximum < 30:
                required.append("degrees")
        tools[name] = {"type": "object", "properties": properties,
                       "required": required, "additionalProperties": False}

    def supported(name, obj):
        if state_contract and (name != state_action or obj.get("objectId") != state_contract["target_alias"]
                               or obj.get("objectType") != state_contract["object_type"]):
            return False
        if name == "PickupObject": return not held and obj.get("pickupable") and not obj.get("isPickedUp")
        if name == "PutObject": return (held and obj.get("objectId") != scene.get("heldObject")
                                        and (destination_type is None or obj.get("objectType") == destination_type)
                                        and obj.get("objectId") not in incompatible_receivers
                                        and not obj.get("isPickedUp") and obj.get("receptacle")
                                        and (not obj.get("openable") or obj.get("isOpen")))
        if name == "OpenObject": return obj.get("openable") and not obj.get("isOpen")
        if name == "CloseObject": return obj.get("openable") and obj.get("isOpen")
        if name == "ToggleObjectOn": return obj.get("toggleable") and not obj.get("isToggled")
        if name == "ToggleObjectOff": return obj.get("toggleable") and obj.get("isToggled")
        if name == "SliceObject": return obj.get("sliceable") and not obj.get("isSliced")
        if name == "CookObject": return obj.get("cookable") and not obj.get("isCooked")
        if name == "CleanObject": return obj.get("dirtyable") and obj.get("isDirty")
        if name == "DirtyObject": return obj.get("dirtyable") and not obj.get("isDirty")
        if name == "BreakObject": return obj.get("breakable") and not obj.get("isBroken")
        if name == "FillObjectWithLiquid": return obj.get("canFillWithLiquid") and not obj.get("isFilledWithLiquid")
        return False

    for name in sorted(OBJECT_ACTIONS):
        if name in {"BreakObject", "DirtyObject"} and not allow_destructive:
            continue
        targets = sorted({obj["objectId"] for obj in objects if capability_access or supported(name, obj)})
        if not targets and not capability_access:
            continue
        properties = {"action": {"type": "string", "enum": [name]},
                      "objectId": ({"type": "string", "enum": targets} if targets else
                                   {"type": "string", "minLength": 1, "maxLength": 32})}
        required = ["action", "objectId"]
        if name == "FillObjectWithLiquid":
            properties["fillLiquid"] = {"type": "string", "enum": ["water", "coffee", "wine"]}
            required.append("fillLiquid")
        tools[name] = {"type": "object", "properties": properties, "required": required,
                       "additionalProperties": False}
    if scene.get('allow_target_correspondence') is True:
        for name,specification in tools.items():
            specification['properties']['selection_report']=deepcopy(SELECTION_REPORT_SCHEMA)
            if name in OBJECT_ACTIONS:specification['required'].append('selection_report')
    allowed = scene.get("allowed_action_names")
    if allowed is not None:
        tools = {name: spec for name, spec in tools.items() if name in allowed}
    return tools


def response_schema(tools):
    return {"type": "object", "properties": {"action": {"anyOf": list(tools.values())}},
            "required": ["action"], "additionalProperties": False}


def validate_available_action(action, tools):
    action = validate_action(action)
    specification = tools.get(action["action"])
    if specification is None:
        raise ProtocolError("Tool unavailable under the current observable preconditions")
    missing = set(specification["required"]) - set(action)
    if missing:
        raise ProtocolError("Current tool requires explicit parameters: " + ", ".join(sorted(missing)))
    if set(action) - set(specification['properties']):
        raise ProtocolError('Parameter not offered in this prepared tool contract')
    for key, value in action.items():
        spec = specification["properties"][key]
        if key == 'target_alias' and value not in spec.get('enum', []):
            raise ProtocolError('Target report alias was not present in this actor input')
        if spec.get("type") == "number":
            if value < spec.get("minimum", -math.inf):
                raise ProtocolError(key + " is below the currently available range")
            if value > spec["maximum"]:
                raise ProtocolError(key + " exceeds the currently available range")
            step = spec.get("multipleOf")
            if step and not math.isclose(value / step, round(value / step), abs_tol=1e-6):
                raise ProtocolError(key + " must be a multiple of " + str(step))
    if "objectId" in action and 'enum' not in specification["properties"]["objectId"]:
        raise ProtocolError('No current grounded target alias was supplied; no native command. The operation capability still exists; inspect or report uncertainty.')
    if "objectId" in action and action["objectId"] not in specification["properties"]["objectId"].get("enum", []):
        raise ProtocolError("Target unavailable under current affordances, observed rejections or explicit task constraints")
    return action


def numeric_tool_contract(tools):
    """Own API parameter domains only; never object target enums or native state."""
    return {name: {key: {**deepcopy(spec), 'required': key in schema['required']}
                   for key, spec in schema['properties'].items() if spec['type'] == 'number'}
            for name, schema in tools.items()
            if any(spec['type'] == 'number' for spec in schema['properties'].values())}


def actor_offer_contract(tools, planner_decision, input_mode, *, provenance=None, scene=None):
    """Describe the same prepared offer used to validate the eventual action."""
    packet = {'planner_decision': planner_decision, 'input_mode': input_mode,
            'restriction': 'observation_only' if planner_decision == 'observe' else 'current_affordances',
            'available_action_names': sorted(tools),
            'numeric_parameters': numeric_tool_contract(tools),
            'interface_contract': action_interface_contract(),
            'scope': 'Tools actually offered for this call. Observation-only restriction follows the planner decision; current absence is not permanent incapability. No target, action, remedy or outcome is selected.'}
    if provenance is not None:
        required = {'proposal_call_id', 'actor_call_id', 'observation_id'}
        if (not isinstance(provenance, dict) or set(provenance) != required
                or any(not isinstance(value, str) or not value for value in provenance.values())):
            raise ValueError('Offer provenance requires exact proposal/actor/observation identities')
        packet['provenance'] = deepcopy(provenance)
        packet['restriction_source'] = ('model_planner_observe_selection' if planner_decision == 'observe'
                                        else 'current_actor_observation_affordances')
        packet['availability_scope'] = 'this_prepared_offer_only_not_global_capability'
    if scene is not None and scene.get('operation_access') == 'capability-feasibility-v1':
        packet.update(version='capability-feasibility-v1',
            restriction_source='model_planner_observe_selection' if planner_decision=='observe' else 'installed_configuration_authorization',
            restriction='observation_mode_authorization' if planner_decision=='observe' else 'configuration_authorization',
            availability_scope='Invocation authorization only;current physical feasibility is separate',
            current_feasibility=operation_feasibility(scene, tools),
            scope='Operations are callable under this mode/configuration;their presence does not predict physical success. Empty candidates do not remove a capability. Use current aliases only;native results decide effects.')
    return packet


def operation_feasibility(scene, tools):
    """Explicit native-assisted conditions,not an action gate or success oracle."""
    held=bool(scene.get('heldObject')); result={}
    capability={'PickupObject':'pickupable','PutObject':'receptacle','OpenObject':'openable',
        'CloseObject':'openable','ToggleObjectOn':'toggleable','ToggleObjectOff':'toggleable',
        'SliceObject':'sliceable','CookObject':'cookable','CleanObject':'dirtyable',
        'DirtyObject':'dirtyable','BreakObject':'breakable','FillObjectWithLiquid':'canFillWithLiquid'}
    desired={'PickupObject':('isPickedUp',False),'OpenObject':('isOpen',False),
        'CloseObject':('isOpen',True),'ToggleObjectOn':('isToggled',False),
        'ToggleObjectOff':('isToggled',True),'CookObject':('isCooked',False),
        'CleanObject':('isDirty',True),'DirtyObject':('isDirty',False),
        'BreakObject':('isBroken',False),'SliceObject':('isSliced',False),
        'FillObjectWithLiquid':('isFilledWithLiquid',False)}
    for name in tools:
        conditions=[];targets=[]
        if name in {'DropHandObject','ThrowObject','MoveHeldObject','PutObject'} and not held:
            conditions.append('no_held_object')
        if name=='PickupObject' and held:conditions.append('hand_occupied')
        if name=='MoveHeldObject' and scene.get('handMotion',{}).get('remaining_attempts',HAND_MOVE_MAX_ATTEMPTS)<=0:
            conditions.append('hand_attempt_limit')
        if name in OBJECT_ACTIONS:
            if not scene.get('objects'):conditions.append('no_current_grounded_target_alias')
            for obj in scene.get('objects',[]):
                if obj.get(capability[name]) is not True:
                    continue
                known=[];unknown=[]
                flag=capability[name]
                if obj.get(flag) is not True:known.append('target_lacks_'+flag)
                if name in desired:
                    field,value=desired[name]
                    if field not in obj:unknown.append(field)
                    elif obj[field]!=value:known.append('current_state_'+field)
                if obj.get('nativeVisible') is False:known.append('native_visibility_precondition_false')
                if obj.get('isInteractable') is False:known.append('native_interactability_false')
                if name=='PutObject' and obj.get('openable') and obj.get('isOpen') is not True:known.append('receiver_not_open')
                targets.append({'target_alias':obj['objectId'],'known_conditions_not_met':known,
                                'unknown_conditions':unknown,'physical_success':'unverified_native_attempt_required'})
            if scene.get('objects') and not targets:
                conditions.append('no_supported_affordance_observed')
        result[name]={'invocation_authorized':True,'known_conditions_not_met':conditions,
                      'targets':targets,'scope':'Advisory current native-assisted state;not a global capability restriction or certified impossibility/success'}
    return result


def registry_prompt(task, scene, tools, completion_managed, guide_mode="contextual", *, capability_contract=None):
    target_map = {name: specification["properties"]["objectId"].get("enum", [])
                  for name, specification in tools.items() if "objectId" in specification["properties"]}
    completion = ("The controller handles task completion. Select an available physical tool; Done is not a policy tool in this episode."
                  if completion_managed else "Done requests independent evaluation; it is not proof of task success.")
    numeric = numeric_tool_contract(tools)
    role = ''
    if scene.get('allow_planning_feedback') is True:
        role = ('CURRENT OBJECTIVE AND HISTORICAL FEEDBACK\n'
            'task is the current planner-issued local objective. task_goal and plan_steps '
            'are context, not permission to advance to a later step. Previous execution '
            'belongs to its recorded earlier objective, even when its text or tool matches '
            'the current task. A successful motion does not establish the intended observation '
            'or task benefit. Assess the current objective against current evidence yourself. '
            'Choose a supported means within this objective; do not silently replace it with '
            'a later objective. If you judge it already satisfied, ambiguous, unsafe or unsupported, '
            'use RequestReplan with your evidence and disagreement so the planner can reassess. '
            'Do not repeat a motion merely because the task text repeats. RequestReplan is a '
            'valid selection with no physical effect; no physical tool is required on this call.\n')
    if scene.get('allow_target_reports') is True:
        role += ('ROLE AND TARGET CORRESPONDENCE\n'
            'The task is the planner model\'s unverified objective, not a human command or '
            'system-certified identity. Keep its intended benefit distinct from a named alias. '
            'Use current observation identity and affordances to assess the named item; '
            'an available alias does not make it the intended object. If identity conflicts '
            'or a prerequisite is unsupported, report it to the planner instead of substituting '
            'a different object. RequestReplan(reason,target_alias) optionally names ONE typed '
            'candidate you actually see in this input. Its recorded identity and affordances '
            'return with your reason without a native command. Omit target_alias when no '
            'candidate is supported. The planner chooses reinspection, revision or withdrawal. '
            'Distinguish an absent intended object from an absent candidate table. '
            'If observation.objects contains aliases, do not report that no aliases were supplied. '
            'A visible shape that conflicts with supplied object identities is an unresolved visual '
            'interpretation, not proof that the controller omitted a required alias. '
            'Drop releases into physics; it does not restore the item to its prior surface.\n')
    if scene.get('allow_target_correspondence') is True:
        role+=('PRE-ACTION SELECTION REPORT\n'
            'For an objectId action include selection_report:{intent_path,assessment,reason,evidence}. '
            'intent_path cites your delivered planner intent using actor_selection_contract; evidence uses '
            'actual delivered observation IDs and cues. matches_intent is your unverified assessment, not '
            'a controller certificate. Object-directed execution requires a current-view citation; if '
            'uncertain/different, RequestReplan can return your reasoning without a native effect. '
            'The controller checks sources, never chooses an alias or repairs your plan. selection_report '
            'is controller metadata and is removed before native dispatch.\n')
    return (role + "Select ONE available tool using the current task, observation and execution result. "
            "Return only {\"action\":{\"action\":\"TOOL_NAME\",...parameters}}. No thought or prose. "
            "Body movement magnitudes and rotation angles must be positive. MoveHeldObject uses signed camera-relative axes; its whole vector must be nonzero and <=0.25m. Omit an optional numeric parameter to use its documented default. "
            + completion + "\n" + model_guide(task, scene, guide_mode, available_names=set(tools),
                semantics_in_input=literal_capability_semantics_match(capability_contract))
            + "\nCurrent numeric constraints override generic defaults; required parameters must be explicit: " + json.dumps(numeric, separators=(",", ":"))
            + ("\nCurrent target aliases by tool (identity only,not success): " if scene.get('operation_access')=='capability-feasibility-v1'
               else "\nCurrent eligible target aliases by tool: ") + json.dumps(target_map, separators=(",", ":"))
            + ("\nInvocation authorization and physical feasibility are separate. Use current observed affordances and state; no success is promised. Never invent an alias when the target list is empty."
               if scene.get('operation_access')=='capability-feasibility-v1' else ''))
