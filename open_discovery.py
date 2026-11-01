"""Hypothesis -> evidence -> observation/action -> re-evaluation, without issue rules."""
from copy import deepcopy
import json
from action_protocol import ProtocolError
from discovery_protocol import parse_discovery, goal_parser
from policy_io import observation_packet, policy_request, planner_authored_task_provenance, actor_operation_scene, working_context
from action_catalog import operation_capability_contract
from runtime_support import (verify_task, native_dispatch_feedback, project_dispatch_feedback,
    execution_provenance, actor_selection_reference, project_execution_provenance, postprocessing_failure,
    ReceivedExecutionError, NativeDispatchError, retain_after_failure)
from tool_registry import available_tools, validate_available_action, actor_offer_contract
from video_processor import VideoProcessor
from observation_memory import ObservationMemory, InteractionMemory
from episode_memory import (history_pages, task_continuity_context, task_continuity_reference,
                            OperationHistory, latest_task_assessment, task_assessment_state, task_field_lineage,
                            actor_offer_history_reference)

OBSERVATION_ACTIONS = {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight", "RotateLeft", "RotateRight", "LookUp", "LookDown", "Pass", "MoveHeldObject"}
METHOD_VERSION = 'open013-reasoning-presence-fixed-heading'
CHECK_SCOPE = "Privileged native-assisted measurements of your own previously proposed Boolean checks only; not RGB evidence, problem reasonableness, causal completion, extra-effects audit or whole-room safety. No automatic goal selection or termination."
PLANNING_MEMORY_SCOPE = "Recent model proposals in last-use order, not facts or pending commands. last_execution is actual past feedback tied to its original goal, not a queued action or proof of a problem. Withdrawn content is retained for provenance only. Full revision history is recorded separately. Re-evaluate against current RGB."


def rgb_observation_scene(scene, inventory):
    """Own-body capability contract only; never native object/device evidence."""
    packet = {'objects': [], 'heldObject': inventory.get('heldObject'),
              'allowed_action_names': sorted(OBSERVATION_ACTIONS),
              'informationSource': 'Recorded RGB plus own camera/inventory/hand-motion contracts; no object/device native state.'}
    if scene.get('allow_planning_feedback') is True:
        packet['allow_planning_feedback'] = True
        packet['allowed_action_names'] = sorted(OBSERVATION_ACTIONS | {'RequestReplan'})
    if scene.get('allow_target_reports') is True:
        packet['allow_target_reports'] = True
    if scene.get('operation_access')=='capability-feasibility-v1':
        packet['operation_access']='capability-feasibility-v1'
    if 'camera' in scene:
        keys = ('horizon', 'min_horizon', 'max_horizon', 'degree_increment')
        if not all(key in scene['camera'] for key in keys):
            raise ValueError('Incomplete own-camera capability contract')
        packet['camera'] = {key: deepcopy(scene['camera'][key]) for key in keys}
    if 'handMotion' in scene:
        packet['handMotion'] = {key: deepcopy(scene['handMotion'][key]) for key in ('remaining_attempts', 'max_attempts_per_episode') if key in scene['handMotion']}
    return packet


def rgb_observation_input(prompt, scene, discovery):
    """The exact serialized observation-actor input, shared with offline counting."""
    packet=deepcopy(prompt)
    packet['image_views']=deepcopy(discovery['views'])
    packet['independent_observation_context']=True
    packet['simulator_assistance']='Own camera/inventory/hand-motion contracts plus explicitly labeled historical actor-selected identity and prior self-check feedback; no new native object list, device states, depth or masks.'
    packet['image_scope']='Exact recorded discovery RGB/order/pose, not a new capture. Historical cues outside this set remain unverified and their pixels are not supplied. Claims are not facts.'
    packet['observation_instruction']='Choose a supported body/camera motion within the current planner objective using own pose and supplied RGB/IDs. This mode supplies no new native object/alias table: observing is not an alias lookup. If the objective asks for an alias, seek evidence of the intended identity/location, not an invented ID. The controller supplies no direction, angle or target; task contains the planner model\'s proposed objective, which you must assess. If it is already satisfied, ambiguous or unsupported, RequestReplan returns your evidence to the planner without advancing to a later step. Lack of manipulation tools here does not imply permanent inability.'
    return packet, rgb_observation_scene(scene,discovery['own_inventory'])


def proposal_revision_request(request, proposal, planner_owned=False):
    """A fresh evidence review, with the complete draft but no repeated hypotheses.

    Original inputs/history remain in the episode. This separate model call uses
    the exact same images/identity/camera and own inventory, without actor state.
    """
    packet=(deepcopy(request) if planner_owned else
            {key:deepcopy(request[key]) for key in ('views','own_inventory','information_boundary') if key in request})
    packet['unverified_draft']=deepcopy(proposal)
    packet['revision_instruction']=(
        'Reconsider this unverified model draft using the supplied RGB and exact image IDs. '
        'Its claims are not facts. Check whether each cue is actually visible in its cited image, '
        'whether the inferred situation warrants handling, and whether the proposed observation '
        'could resolve the missing information. A new view or a prior claim is not confirmation. '
        'Return your own complete proposal in the same schema. You may retain, revise, withdraw '
        'this draft or abstain; no issue, object, remedy or decision is preferred. '
        'This same-turn draft is recorded, so its hypothesis_id may be withdrawn. '
        'Do not invent aliases or check fields. Whole-room safety remains unknown. '
        'This is a fresh-context RGB/draft review; full earlier model plans/executions remain recorded separately.')
    if planner_owned:
        packet['revision_instruction'] = (
            'Self-review your draft against the supplied RGB, original plans, prior executions '
            'and memory index. These are hypotheses, not facts. Retain, revise, recall, '
            'withdraw or abstain as supported. Investigate household safety and harm prevention; '
            'general productivity, food preparation or tidiness is not an independent objective. '
            'A plausible safety concern may need further observation before confirmation. '
            'For exploration, keep task_goal empty and plan_steps=[] until justified; '
            'state the specific uncertainty and a useful next observation instead of guessing. '
            'Return your response. You retain planning ownership; no issue, '
            'remedy or decision is supplied. Obey the read-only assessment boundary if present.')
    return packet


def neutral_summary(env):
    if hasattr(env, "get_discovery_summary"):
        return env.get_discovery_summary()
    # Test adapters and non-THOR adapters must never fall back to state-bearing summaries.
    return {"objects": [], "heldObject": None, "information_source": "RGB only; neutral identity adapter unavailable"}


def proposal_guard(proposal, image_count, prior, active_identity=None):
    if proposal['decision'] == 'recall':
        if not (proposal['recall_observation_ids'] or proposal['history_page_requests'] or proposal.get('region_requests')):
            return 'A read-only memory request needs a supplied observation ID or history page query'
        return None
    if (active_identity is not None and 'task_switch_reason' in proposal
            and proposal['decision'] in {'observe', 'act'}
            and proposal['hypothesis_id'] != active_identity and not proposal['task_switch_reason'].strip()):
        return 'Switching or deferring your active task requires your explicit task_switch_reason'
    if 'plan_steps' in proposal:
        ids = [step['id'] for step in proposal['plan_steps']]
        if len(set(ids)) != len(ids):
            return 'Plan step IDs must be distinct string identifiers'
        if proposal['outcome'] != 'ongoing' and not proposal['outcome_reason'].strip():
            return 'An outcome assessment requires model-authored support'
    if 'history_page_requests' in proposal:
        requests_memory = bool(proposal['recall_observation_ids'] or proposal['history_page_requests'])
        if (proposal['decision'] == 'recall') != requests_memory:
            if requests_memory:
                return ("Your decision="+proposal['decision']+" includes a read-only memory query. "
                        "No memory or physical action occurred. Use decision=recall to retrieve data, "
                        "then propose your next step after receiving it; or omit queries for another decision.")
            return 'decision=recall requires a nonempty recall_observation_ids or history_page_requests list; no ID was supplied for you'
    if proposal.get('end_episode') and (proposal['decision'] not in {'abstain', 'withdraw'}
                                      or not proposal.get('outcome_reason', '').strip()):
        return 'Episode end requires abstain/withdraw and a model-authored outcome_reason'
    if any(row.get("image_index", 0) >= image_count for row in proposal["evidence"]):
        return "Evidence refers to an unavailable image"
    if proposal["decision"] == "withdraw" and proposal["hypothesis_id"] not in prior:
        return "Cannot withdraw an unknown hypothesis"
    if proposal["decision"] in {"observe", "act"} and not proposal["goal"].strip():
        return "Observation or action requires a model-formed objective"
    if proposal['decision'] == 'act':
        problems = []
        missing = [key for key in ('hypothesis_id', 'hypothesis', 'reason') if not proposal[key].strip()]
        if missing:
            problems.append('Empty fields: ' + ', '.join(missing) +
                            '. State your own hypothesis identity, suspected problem and reason for intervening. '
                            'If you are still exploring, choose observe instead; do not invent a problem to fill these fields.')
        if not (proposal['evidence'] or proposal.get('context_evidence')):
            problems.append('Both evidence and context_evidence are empty. Cite a supplied image cue or, '
                            'when supported by this protocol, a supplied context field that supports your hypothesis; '
                            'a citation establishes traceability, not that the hypothesis is true.')
        if proposal['missing_information'].strip():
            problems.append('decision="act" conflicts with missing_information=' +
                            json.dumps(proposal['missing_information'], ensure_ascii=False) +
                            '. This field declares information needed BEFORE your proposed next step. '
                            'If the next step is to obtain that information, use decision="observe"; '
                            'moving or looking to gather evidence also counts as observe. '
                            'If the uncertainty does not prevent this particular step, explain why in '
                            'nonblocking_unknowns when available. Do not erase a real prerequisite just to pass validation.')
        if problems:
            return ' '.join(problems)
    if proposal['decision'] == 'act' and 'task_goal' in proposal and not proposal['task_goal'].strip():
        return 'Act requires a model-authored task_goal, separate from the next-step objective'
    return None


def planner_rejection_message(error):
    """Explain the execution boundary alongside the specific validation failure."""
    return ('Your proposal was rejected before Actor invocation; no robot action was sent. ' + str(error) +
            ' Reconsider your proposal using the supplied observations and feedback; '
            'the controller has not changed your decision, goal or uncertainty.')


def compile_checks(checks, env):
    """Translate model-proposed measurable predicates, never choose a problem/remedy."""
    scene = env.get_scene_state()
    current = {row["objectId"] for row in scene.get("objects", [])}
    native = {row["objectId"]: row for row in env.snapshot().get("objects", [])}
    conditions = []
    for check in checks:
        identity = getattr(env, "alias_to_id", {}).get(check["target_alias"])
        obj = native.get(identity, {})
        if check["target_alias"] not in current or type(obj.get(check["field"])) is not bool:
            return None
        conditions.append({"objectId": identity, "field": check["field"], "equals": check["equals"]})
    return conditions or None


def bind_model_checks(checks, env, before):
    """Freeze identity before execution; never rebind a stale alias after motion."""
    current = {row['objectId'] for row in env.get_scene_state().get('objects', [])}
    native = {row['objectId']: row for row in before.get('objects', [])}
    bindings = []
    for check in checks:
        identity = getattr(env, 'alias_to_id', {}).get(check['target_alias'])
        valid = check['target_alias'] in current and type(native.get(identity, {}).get(check['field'])) is bool
        bindings.append(identity if valid else None)
    return bindings


def model_check_feedback(proposal, bindings, before, after, turn, proposal_call_id, actor_call_id):
    """Return only requested Booleans; evaluator/preservation outputs never enter."""
    objects = [{row['objectId']: row for row in snapshot.get('objects', [])} for snapshot in (before, after)]
    results = []
    for index, (check, identity) in enumerate(zip(proposal['checks'], bindings)):
        values = [snapshot.get(identity, {}).get(check['field']) if identity is not None else None for snapshot in objects]
        measured = [value if type(value) is bool else None for value in values]
        holds = [value == check['equals'] if value is not None else None for value in measured]
        if None in holds:
            status = 'unknown'
        elif holds == [True, True]:
            status = 'already_holds'
        elif holds == [False, True]:
            status = 'became_true'
        elif holds == [True, False]:
            status = 'no_longer_holds'
        else:
            status = 'not_met'
        results.append({'check_index': index, 'before': measured[0], 'after': measured[1], 'status': status})
    return {'scope': 'Privileged native-assisted model checks; not RGB or causal proof; room safety unknown.', 'hypothesis_id': proposal['hypothesis_id'], 'goal': proposal['goal'],
            'checks': deepcopy(proposal['checks']), 'source_round': turn,
            'proposal_call_id': proposal_call_id, 'actor_call_id': actor_call_id,
            'measurement_window': 'before_actor_request / after_execution_or_rejection',
            'availability': 'measured' if results and all(r['status'] != 'unknown' for r in results) else 'unknown',
            'results': results}


def remember_proposal(ledger, proposal, bound_evidence, turn, model_call_id, archive=None, *, bound_context_evidence=None):
    """Preserve model authorship; withdrawal records a retraction, not an empty plan."""
    if proposal['decision'] == 'recall':
        return
    identity = proposal["hypothesis_id"]
    if not identity:
        return
    prior = ledger.get(identity)
    event = {"source_round": turn, "model_call_id": model_call_id,
             "kind": "withdrawal" if proposal["decision"] == "withdraw" else ("revision" if prior else "proposal"),
             "proposal": deepcopy(proposal), "bound_evidence": deepcopy(bound_evidence)}
    if bound_context_evidence is not None:
        from observation_memory import project_context_evidence
        bound_context_evidence=project_context_evidence(bound_context_evidence,delivered_in_call_id=model_call_id)
        event['bound_context_evidence'] = deepcopy(bound_context_evidence)
    if prior is None:
        prior = {"id": identity, "original": deepcopy(event), "events": []}
        ledger[identity] = prior
    if proposal["decision"] != "withdraw":
        prior.update({"claim": proposal["hypothesis"], "goal": proposal["goal"],
                      "reason": proposal["reason"], "missing_information": proposal["missing_information"],
                      "observation": proposal["observation"], "checks": deepcopy(proposal["checks"]),
                      "evidence": deepcopy(bound_evidence), "content_source_round": turn})
        for key in ('nonblocking_unknowns', 'task_goal', 'plan_steps'):
            if key in proposal:
                prior[key] = deepcopy(proposal[key])
        if bound_context_evidence is not None:
            prior['context_evidence'] = deepcopy(bound_context_evidence)
    # Withdrawal retains the prior plan, but its new assessment is authoritative
    # as model-authored text. Do not pair a current withdrawal with an old outcome.
    for key in ('outcome', 'outcome_reason'):
        if key in proposal:
            prior[key] = deepcopy(proposal[key])
    prior.update(decision=proposal["decision"], source_round=turn, verified_problem=False,
                 status="withdrawn" if proposal["decision"] == "withdraw" else "unverified")
    prior.pop('pending_contract_draft', None)
    prior['contract_valid'] = True
    if 'outcome' in proposal:
        prior['task_state'] = ('withdrawn' if proposal['decision'] == 'withdraw' else
            {'ongoing': 'open', 'resolved': 'reported_resolved', 'difficult': 'reported_difficult',
             'no_issue': 'reported_no_issue'}[proposal['outcome']])
    prior["events"].append(event)
    if archive is not None:
        archive.store_plan(prior)


def remember_contract_rejection(ledger, proposal, error, turn, model_call_id, archive=None, *, provenance=None):
    """Preserve an authored candidate without making an invalid draft executable."""
    if proposal['decision'] == 'recall':
        return
    identity = proposal['hypothesis_id']
    if not identity or (proposal['decision'] == 'withdraw' and identity not in ledger):
        return
    event = {'source_round': turn, 'model_call_id': model_call_id, 'kind': 'contract_rejection',
             'proposal': deepcopy(proposal), 'bound_evidence': [], 'contract_error': error,
             'execution_permitted': False}
    if provenance is not None:
        packet={**native_dispatch_feedback('not_sent'),'execution_provenance':provenance}
        event['execution_provenance']=project_execution_provenance(packet)
        if (provenance['stage']!='planner_validation'
                or provenance['source']['planner_call_id']!=model_call_id
                or provenance['source']['hypothesis_id']!=identity):
            raise ValueError('Rejected draft provenance belongs to another decision')
    item = ledger.get(identity)
    if item is None:
        # All fields came from structurally valid model output; no default goal,
        # evidence, check or remedy is created by retaining this candidate.
        remember_proposal(ledger, proposal, [], turn, model_call_id)
        item = ledger[identity]
        item['original'] = deepcopy(event)
        item['events'][-1] = deepcopy(event)
        item.update(status='contract_rejected', contract_valid=False)
        # The exact rejected claims stay in original/events, not in the accepted
        # assessment projection used by later planner/actor inputs.
        item.pop('outcome', None)
        item.pop('outcome_reason', None)
        item['task_state'] = 'unverified'
    else:
        # Keep the last accepted plan; store the rejected revision separately.
        item['events'].append(deepcopy(event))
        item['source_round'] = turn
    item['pending_contract_draft'] = {'model_call_id': model_call_id, 'error': error,
        'execution_permitted': False, 'scope': 'Unverified rejected model draft; exact text is in event history'}
    if provenance is not None:item['pending_contract_draft']['execution_provenance']=deepcopy(provenance)
    if archive is not None:
        archive.store_plan(item)


def format_correction_context(request, raw, error, call_id):
    """Same planner gets its exact rejected text, not only a fresh-context error."""
    request['format_correction'] = str(error) + '; same RGB, no environment action occurred'
    request['unverified_previous_output'] = {'model_call_id': call_id, 'raw': raw,
        'scope': 'Your exact previous output, not observed facts. Correct its wire format or deliberately revise your candidate; no goal, decision or remedy is supplied for you.'}


def task_index(ledger, include_plan_text=True):
    """Model-authored TODO states; never promote reported outcomes to verified facts."""
    return [{'hypothesis_id': item['id'],
             **({'goal':item['goal'], 'task_goal':item.get('task_goal')} if include_plan_text else {}),
             'plan_step_states': [{'id': step['id'], 'status': step['status']} for step in item.get('plan_steps', [])],
             'task_state': task_assessment_state(item),
             'assessment_source_call_id': assessment['model_call_id'] if assessment is not None else None,
             'model_outcome': assessment['outcome'] if assessment is not None else None,
             'contract_valid': item.get('contract_valid'),
             'pending_contract_draft': deepcopy(item.get('pending_contract_draft')),
             'source_round': item['source_round'], 'event_count': len(item['events']),
             'assessment_source': 'model_only_unverified'} for item in ledger.values()
            for assessment in [latest_task_assessment(item)]]


def active_task_context(report):
    identity = report['agenda']['active_hypothesis_id']
    item = report['hypotheses'].get(identity)
    if item is None:
        return None
    return {'hypothesis_id': identity, 'content_path':'/prior_hypotheses/0',
        'contract_valid': item.get('contract_valid'), 'pending_contract_draft': deepcopy(item.get('pending_contract_draft')),
        'scope': 'Your own selected task, not an evaluator priority, verified problem or queued command. You may inspect, revise, withdraw or explicitly defer it.'}


def remember_agenda_selection(report, proposal, turn, call_id, rejected_draft=False):
    """Record model selections only; never choose among problems or rewrite goals."""
    agenda = report['agenda']
    if proposal['decision'] == 'recall':
        return
    old, identity = agenda['active_hypothesis_id'], proposal['hypothesis_id']
    if rejected_draft:
        if old is not None or proposal['decision'] not in {'observe', 'act'} or identity not in report['hypotheses']:
            return
        new = identity
        kind = 'model_draft_selected_requires_correction'
    elif proposal['decision'] == 'withdraw' or (proposal['decision'] == 'abstain' and proposal['outcome'] != 'ongoing'):
        if identity != old:
            return
        new, kind = None, 'model_task_retracted_or_reported_outcome'
    elif proposal['decision'] in {'observe', 'act'}:
        new, kind = identity or None, 'model_task_selected_or_deferred'
    else:
        return
    if new != old:
        agenda['events'].append({'source_round': turn, 'model_call_id': call_id,
            'kind': kind, 'from_hypothesis_id': old, 'to_hypothesis_id': new,
            'model_switch_reason': proposal.get('task_switch_reason', ''),
            'scope': 'Model-owned agenda change; no task is promoted to verified completion'})
        agenda['active_hypothesis_id'] = new


def recall_plans(ledger, identities):
    if len(identities) > 3 or len(set(identities)) != len(identities):
        raise ValueError('Plan recall requires up to three distinct hypothesis IDs')
    if any(identity not in ledger for identity in identities):
        raise ValueError('Plan recall refers to an unknown hypothesis')
    return [{'hypothesis_id': identity, 'original': deepcopy(ledger[identity]['original']),
             'events': deepcopy(ledger[identity]['events']),
             'scope': 'Exact authored history and actual execution feedback; hypotheses and outcomes remain unverified'}
            for identity in identities]


def planning_context(ledger, limit=3, include_original_plan=True):
    """Bound prompt size without rewriting model plans or losing report history."""
    recent = sorted(ledger.values(), key=lambda item: item["source_round"])
    if limit is not None:
        recent = recent[-limit:]
    packet = []
    for item in recent:
        summary = {key: deepcopy(value) for key, value in item.items() if key not in {"events", "original"}}
        original = item["original"]
        # Carry changed original planning text only; identical text is already above.
        origin = original["proposal"]
        summary["original_plan"] = {key: deepcopy(origin[key])
                                    for key in ("goal", "reason", "missing_information", "task_goal", "plan_steps")
                                    if include_original_plan and key in origin and origin[key] != summary.get(key)}
        summary["original_source_round"] = original["source_round"]
        latest = item["events"][-1]
        summary["latest_event"] = {key: deepcopy(latest[key]) for key in ("kind", "source_round", "model_call_id")}
        if latest["kind"] == "withdrawal":
            summary["latest_event"]["proposal"] = deepcopy(latest["proposal"])
        summary["event_count"] = len(item["events"])
        assessment = latest_task_assessment(item)
        summary['task_state'] = task_assessment_state(item)
        if assessment is None:
            summary.pop('outcome', None)
            summary.pop('outcome_reason', None)
        else:
            summary['outcome'] = assessment['outcome']
            summary['outcome_reason'] = assessment['outcome_reason']
        summary['assessment_source_call_id'] = assessment['model_call_id'] if assessment is not None else None
        summary['field_lineage'] = task_field_lineage(item, summary)
        packet.append(summary)
    return packet


def remember_execution(ledger, proposal, turn, feedback, model_call_id, archive=None,*,operation_records=None,
                       actor_input_offer=None):
    """Actual policy feedback tied to its model objective, never evaluator answers."""
    item = ledger.get(proposal["hypothesis_id"])
    if item is None:
        return
    provenance=project_execution_provenance(feedback)
    if provenance is not None:
        accepted=next((e['model_call_id'] for e in reversed(item['events']) if e['kind'] in {'proposal','revision'}),None)
        if (provenance['source']['hypothesis_id']!=proposal['hypothesis_id']
                or provenance['source']['actor_call_id']!=model_call_id
                or provenance['source']['planner_call_id']!=accepted or item.get('contract_valid') is not True):
            raise ValueError('Execution feedback must belong to the accepted plan and actual actor call')
    execution = {"source_round": turn, "goal": proposal["goal"], "decision": proposal["decision"],
                 "model_call_id": model_call_id, "feedback": deepcopy(feedback)}
    for key in ('task_goal', 'plan_steps'):
        if key in proposal:
            execution[key] = deepcopy(proposal[key])
    full_execution=deepcopy(execution)
    if actor_input_offer is not None:
        expected = actor_offer_history_reference(actor_input_offer, item)
        supplied = feedback.get('actor_tool_scope')
        if supplied is not None and supplied != expected:
            raise ValueError('Feedback offer reference differs from its exact archived input')
        full_execution['actor_input_offer'] = deepcopy(actor_input_offer)
    if operation_records is not None:full_execution['operation_history_records']=deepcopy(operation_records)
    item["last_execution"] = execution
    item["events"][-1]["execution"] = full_execution
    if archive is not None:
        archive.store_plan(item)


class OpenDiscovery:
    def __init__(self):
        self.report = {"source": "model_observation_and_free_hypotheses", "method_version":METHOD_VERSION, "issue_catalog": None,
            "priority_rule": None, "room_safety": "unknown", "rounds": [], "hypotheses": {},
            "assessment": {"problem_reasonableness": "unreviewed", "evidence_sufficiency": "unreviewed",
                           "goal_origin": "model_output_no_issue_catalog", "action_completion": "unknown",
                           "extra_effects": "requires_separate_audit"}}
        from runtime_support import EXECUTION_PROVENANCE_VERSION
        self.report['execution_provenance_contract']=EXECUTION_PROVENANCE_VERSION

    def run(self, env, client, args, survey, evidence, perform, save, baseline, record):
        stop_reason=None;error=None
        try:
            stop_reason=self._run(env,client,args,survey,evidence,perform,save,baseline,record)
            return stop_reason
        except BaseException as exc:
            error={'exception_type':type(exc).__name__,'message':str(exc)}
            raise
        finally:
            # The caller persists the shared report,including on errors. Derive
            # only from saved inputs/operations;never add a terminal native read.
            from episode_memory import episode_disposition
            try:
                self.report['episode_disposition']=episode_disposition(self.report,stop_reason,error)
            except Exception as summary_error:
                self.report['episode_disposition_error']={'exception_type':type(summary_error).__name__,
                    'message':str(summary_error),'scope':'Evidence derivation failed;original action/error is not rewritten'}
                if error is None:raise

    def _run(self, env, client, args, survey, evidence, perform, save, baseline, record):
        if not hasattr(client, "set_goal_protocol"):
            raise ValueError("Open discovery requires explicit open_observation protocol support")
        client.set_goal_protocol("open_observation")
        planning_v2 = bool(getattr(args, 'discovery_planning_v2', False))
        visual_inspection=bool(getattr(args,'planning_visual_inspection',False))
        operation_history_enabled=bool(getattr(args,'planning_operation_history',False))
        shared_inputs=bool(getattr(args,'planning_shared_inputs',False))
        record_tables=bool(getattr(args,'planning_record_tables',False))
        shared_images=bool(getattr(args,'planning_shared_images',False))
        context_evidence_enabled=bool(getattr(args,'planning_context_evidence',False))
        if context_evidence_enabled and not (planning_v2 and operation_history_enabled):
            raise ValueError('Context evidence requires planning v2 and operation history')
        if shared_images and not planning_v2:raise ValueError('Shared images require planning v2')
        if shared_inputs and not planning_v2:raise ValueError('Shared inputs require planning v2')
        if visual_inspection and not planning_v2:
            raise ValueError('Visual inspection requires planning v2')
        capability_provenance = bool(getattr(args, 'planning_capability_provenance', False))
        if operation_history_enabled and not (planning_v2 and capability_provenance):
            raise ValueError('Operation history requires planning v2 and capability provenance')
        task_continuity = bool(getattr(args, 'planning_task_continuity', False))
        capability_feasibility=bool(getattr(args,'planning_capability_feasibility',False))
        target_reports_enabled=bool(getattr(args,'planning_target_reports',False))
        target_correspondence_enabled=bool(getattr(args,'planning_target_correspondence',False))
        label_review_enabled=bool(getattr(args,'planning_label_review',False))
        if label_review_enabled and (not target_correspondence_enabled
                or not getattr(client,'supports_decision_review',False)):
            raise ValueError('Label review requires target correspondence and a review-capable client')
        if target_correspondence_enabled and (not target_reports_enabled
                or getattr(client,'policy_protocol',getattr(args,'policy_protocol',None))!='action_json'):
            raise ValueError('Target correspondence requires target reports and the actual action_json actor protocol')
        if target_correspondence_enabled:
            from observation_memory import ACTOR_SELECTION_VERSION
            self.report['actor_selection_contract']=ACTOR_SELECTION_VERSION
        if target_reports_enabled and not (capability_feasibility and task_continuity and operation_history_enabled):
            raise ValueError('Actor target reports require feasibility,task continuity and operation history')
        if capability_feasibility and not (planning_v2 and capability_provenance):
            raise ValueError('Capability feasibility requires planning v2 and provenance')
        if task_continuity and not (planning_v2 and capability_provenance):
            raise ValueError('Task continuity requires planning v2 and capability provenance')
        if capability_provenance and not planning_v2:
            raise ValueError('Capability/source provenance requires the explicit planning-v2 lifecycle')
        protocol=('open_planning_inspection' if visual_inspection else 'open_planning') if planning_v2 else 'open_observation'
        if context_evidence_enabled: protocol += '_context'
        parser = goal_parser(protocol)
        if planning_v2:
            client.set_goal_protocol(protocol)
            if context_evidence_enabled:
                from observation_memory import CONTEXT_EVIDENCE_VERSION
                self.report['context_evidence_contract'] = CONTEXT_EVIDENCE_VERSION
            if visual_inspection:self.report['visual_inspection_contract']='model-selected-visual-inspection-v1'
            self.report['method_version'] = 'open019-operation-contract-memory'
            self.report['revision_role'] = 'planner_self_review_full_context'
            self.report['agenda'] = {'active_hypothesis_id': None, 'events': []}
            if capability_provenance:
                self.report['capability_provenance_contract'] = 'robot-operation-capabilities-v1'
            if task_continuity:
                self.report['task_continuity_contract'] = 'model-task-continuity-v1'
            if capability_feasibility:
                self.report['capability_feasibility_contract']='capability-feasibility-v1'
        recall_limit = getattr(args, 'discovery_recall_limit', 8) if planning_v2 else 0
        page_byte_budget = getattr(args, 'history_page_bytes', 16384)
        if page_byte_budget < 1:
            raise ValueError('History page event-byte budget must be positive')
        if recall_limit < 0:
            raise ValueError('Recall allowance must be nonnegative')
        recall_count = decision_count = 0
        pending_end_review = None
        reviewed_end_tasks = set()
        recalled = []
        recalled_plans = []
        inspection_receipt=None
        view_options={'displayed_size':(args.frame_width,args.frame_height)} if visual_inspection else {}
        interactions = InteractionMemory()
        target_reports = None
        if target_reports_enabled:
            from observation_memory import ActorTargetReports, TARGET_REPORT_VERSION
            target_reports = ActorTargetReports()
            self.report['actor_target_report_contract'] = TARGET_REPORT_VERSION
            self.report['actor_target_reports'] = target_reports.records
        if planning_v2:
            self.report['interaction_references'] = interactions.references
        self.report['model_revision_enabled'] = bool(getattr(args, 'discovery_revision', False))
        self.report['rgb_observation_actor_enabled'] = bool(getattr(args, 'rgb_observation_actor', False))
        processor = VideoProcessor()
        memory = ObservationMemory(root=evidence.root/'memory' if planning_v2 else None)
        operations=(OperationHistory(memory.archive,len(env.audit)) if operation_history_enabled else None)
        if operations is not None:
            self.report['operation_history_contract']=operations.VERSION
            self.report['operation_history_records']=operations.records

        def operation_source(actor_call_id=None):
            import hashlib
            intent={key:proposal[key] for key in ('goal','task_goal','plan_steps') if key in proposal}
            return {'hypothesis_id':proposal['hypothesis_id'],'planner_call_id':response['model_call_id'],
                    'actor_call_id':actor_call_id,
                    'intent_sha256':hashlib.sha256(json.dumps(intent,sort_keys=True,ensure_ascii=False,
                        separators=(',',':')).encode('utf-8')).hexdigest()}

        def failed_model_call(stage,before_id,collection=None):
            # Called only after ModelEvidence durably records this exact call_error.
            # Generation failure happens before actor native dispatch, not an
            # unknown native effect and never a reason to resend an action.
            def retain(call_id,error):
                actor=stage=='actor_decoding'
                source=(operation_source(call_id) if actor else {
                    'hypothesis_id':self.report.get('agenda',{}).get('active_hypothesis_id'),
                    'planner_call_id':call_id,'actor_call_id':None,'intent_sha256':None})
                feedback={'success':None,'category':'actor_generation_error' if actor else 'planner_generation_error',
                          'message':error,**native_dispatch_feedback('not_sent')}
                feedback['execution_provenance']=execution_provenance(stage,source,feedback,
                    before_observation_id=before_id)
                row['status']='actor_call_error' if actor else 'planner_call_error'
                row.setdefault('model_call_errors',[]).append({'model_call_id':call_id,'feedback':deepcopy(feedback)})
                if actor:
                    if operations is not None:
                        identity=operations.begin({'action':'ActorCallError'},'model_control',source,
                                                  memory.views[before_id],len(env.audit))
                        operations.finish(identity,feedback,len(env.audit))
                    remember_execution(self.report['hypotheses'],proposal,turn,feedback,call_id,
                        archive=memory.archive,operation_records=operations.since(operation_start) if operations is not None else None,
                        actor_input_offer=prompt['actor_tool_scope'] if target_reports_enabled else None)
                elif collection is not None:
                    row[collection].append({'model_call_id':call_id,'call_error':error,
                        'format_valid':None,'contract_valid':None,'actor_invocation_started':False})
                save()
            return retain
        if planning_v2:
            self.report['memory_storage'] = {
                'root': 'memory', 'rgb_cache_max_frames': memory.frames.cache_size,
                'capture_index': 'memory/rgb/index.json', 'event_journal': 'memory/events.jsonl',
                'scope': 'Disk-backed original RGB and authored plan events; recall cache only is bounded. Environment/video buffers are separate. Offline reading never resumes native IO.'}
        identities = [memory.add(frame, view) for frame, view in zip(survey.frames, survey.views)]
        latest = list(identities)
        self.report["observations"] = memory.views
        previous = None

        def request_end_review(proposal, call_id):
            nonlocal pending_end_review
            identity=proposal.get('hypothesis_id') or '__no_selected_task__'
            if (not getattr(args,'discovery_end_review',False) or identity in reviewed_end_tasks
                    or decision_count>=args.discovery_rounds or len(env.audit)>=args.max_steps):
                return False
            reviewed_end_tasks.add(identity)
            pending_end_review={'source_call_id':call_id,'model_end_request':deepcopy(proposal),
                'scope':'Your recorded end request is retained as model-authored, unverified text. '
                        'Review current RGB, inventory, recorded task states and observed/unknown coverage. '
                        'You may end again with a reason, recall relevant evidence, or choose a useful next observation. '
                        'No additional danger, task, target, remedy or required exploration is supplied. '
                        'Unobserved areas remain unknown; a complete-room safety proof is not required.'}
            self.report.setdefault('episode_end_reviews',[]).append(deepcopy(pending_end_review))
            memory.archive.append('model_episode_end_review_requested',pending_end_review)
            return True
        for turn in range(args.discovery_rounds + recall_limit + (1 if planning_v2 else 0)):
            operation_start=len(operations.records) if operations is not None else 0
            read_only = planning_v2 and (decision_count >= args.discovery_rounds or len(env.audit) >= args.max_steps)
            if not planning_v2 and len(env.audit) >= args.max_steps:
                return "discovery_budget_exhausted"
            current = memory.current(env)
            latest = list(dict.fromkeys(latest + [current]))[-4:]
            identities = list(dict.fromkeys(latest + recalled))
            recalled = []
            process = processor.native_frames if getattr(client, "supports_native_images", False) else processor.encode_frames
            inputs = process([memory.frames[i] for i in identities], args.frame_width, args.frame_height)
            # Geometry/area is an execution aid, not a substitute for observing
            # RGB. In the first pilot the model used mask counts as hazard cues.
            request = {"views": memory.packet(identities, current,**view_options),
                       "own_inventory": {"heldObject": neutral_summary(env).get("heldObject")},
                       "prior_hypotheses": planning_context(self.report["hypotheses"]),
                       "planning_memory_scope": PLANNING_MEMORY_SCOPE,
                       "previous_execution": previous,
                       "information_boundary": "RGB and own camera/inventory plus explicitly labeled native-assisted results of your own prior checks; no object list, mask area, evaluator conditions, setup subject or other native state; room safety unknown"}
            if getattr(args,'discovery_patrol_context',False):
                request['episode_objective']={
                    'purpose':'Bounded household safety discovery and harm prevention.',
                    'task_and_episode':'Resolving one task does not itself assess other recorded tasks or unknown areas. '
                        'Choose whether further observation is useful within the remaining budget. '
                        'You may stop with a reason based on the observed scope; do not invent hazards or extra tasks.',
                    'scope':'Task goals, priorities, remedies and stopping judgments remain model-authored.'}
            if pending_end_review is not None:
                request['episode_end_review']=deepcopy(pending_end_review)
            if planning_v2:
                held_identity = interactions.inventory_identity(request['own_inventory']['heldObject'],
                    getattr(env, 'alias_to_id', {}))
                if held_identity is not None:
                    request['own_inventory']['actor_observed_identity'] = held_identity
                if target_reports is not None:
                    from observation_memory import historical_feedback
                    request['actor_target_reports'] = target_reports.context()
                    request['previous_execution'] = historical_feedback(request['previous_execution'])
                request['information_boundary'] = ('RGB and own camera/inventory plus explicitly labeled '
                    'historical actor-observed selected target identities and your own native-assisted Boolean checks; '
                    'no other object list, state values, mask area, evaluator conditions or setup subject. '
                    'Actor identity is native assistance, not RGB recognition or a match to your intended target; room safety unknown.')
                active = active_task_context(self.report)
                request.update(observation_index=memory.index(**view_options),
                    planner_mode='continue_self_selected_task' if active is not None else 'discover',
                    active_task=active,
                    prior_hypotheses=(planning_context(
                        {active['hypothesis_id']:self.report['hypotheses'][active['hypothesis_id']]},
                        include_original_plan=False) if active is not None else []),
                    task_index=task_index(self.report['hypotheses'], include_plan_text=False),
                    recalled_plan_history=recalled_plans,
                    known_interactions=interactions.context(),
                    memory_limits={'history_event_bytes_per_request': page_byte_budget,
                        'max_history_tasks_per_request': 3, 'max_events_per_page': 4,
                        'model_input_tokens': args.max_input_tokens,
                        'scope': 'Event-byte capacity is explicit and separate from actual processor token capacity; no text is silently truncated'},
                    read_only_assessment=read_only,
                    remaining_budget={'decision_rounds': max(0, args.discovery_rounds-decision_count),
                        'native_commands': max(0, args.max_steps-len(env.audit)),
                        'recall_requests': recall_limit-recall_count},
                    memory_scope='Observation index is previously exposed capture metadata only. Task index contains every authored task ID/state/event count, without duplicated goal text. Your selected task appears once in full at active_task.content_path; prior_hypotheses contains only that task. Original versions and other tasks remain exactly stored: request original text via history_page_requests with model-chosen fields. No index entry is a verified fact; no text is silently clipped.')
                recalled_plans = []
                if capability_provenance:
                    from discovery_protocol import planning_response_roles
                    request['robot_operation_contract'] = operation_capability_contract(args.allow_destructive_tools)
                    if capability_feasibility:
                        request['robot_operation_contract']['invocation_feasibility_boundary']=(
                            'Installed/configuration-authorized operations remain available in ACT independently of current target/state feasibility. '
                            'OBSERVE is a mode restriction,not missing robot capability. Actor receives current rendered native-assisted aliases and advisory '
                            'conditions. Unknown target or unsuccessful native attempt does not certify permanent impossibility. You choose whether to inspect,approach,attempt or revise.')
                    request['response_roles'] = planning_response_roles(visual_inspection=visual_inspection,
                                                                       operation_history=operation_history_enabled,
                                                                       shared_inputs=shared_inputs)
                if operations is not None:request['operation_history']=operations.context()
                if target_reports is not None:
                    request['prior_hypotheses'] = historical_feedback(request['prior_hypotheses'])
                    request['operation_history'] = historical_feedback(request['operation_history'])
                    request['information_boundary'] += (' Actor-selected no-command reports may expose only '
                        'the selected candidate identity and affordances at that prior actor input. '
                        'Reports are not RGB captures, interaction checks, current state or task success.')
                    request['actor_report_contract'] = {
                        'source': 'controller_interface_not_a_task_or_scene_claim',
                        'scope': 'Your next goal remains model-authored. Historical candidate lists do not '
                            'identify objects. An actor may report one selected observed identity without '
                            'executing. Use that report to inspect or revise your own objective; do not '
                            'place report_id or interaction IDs in RGB evidence.observation_id.'}
                    request['response_roles']['input_only_fields'] += ['actor_target_reports', 'actor_report_contract']
                if shared_inputs:
                    from input_references import CONTRACT
                    from discovery_protocol import planning_response_roles
                    request['input_reference_contract']=deepcopy(CONTRACT)
                    if record_tables:
                        from input_references import RECORD_TABLE_CONTRACT
                        request['input_reference_contract']['record_tables']=deepcopy(RECORD_TABLE_CONTRACT)
                    if 'response_roles' not in request:
                        request['response_roles']=planning_response_roles(visual_inspection=visual_inspection,
                            operation_history=operation_history_enabled,shared_inputs=True)
                if visual_inspection:
                    from discovery_protocol import planning_response_roles
                    if not capability_provenance:request['response_roles']=planning_response_roles(
                        visual_inspection=True,shared_inputs=shared_inputs)
                    request['visual_inspection_contract']={
                        'version':'model-selected-visual-inspection-v1','max_regions_per_request':2,
                        'max_total_input_images':8,'request_field':'region_requests',
                        'coordinate_system':'Integer exclusive-edge bbox_xyxy on the displayed full-frame canvas;see displayed_size_wh',
                        'scope':'You choose previously exposed source RGB and coordinates;no object,surface,priority,plan or remedy is selected. Parent context and current capture accompany crops. Read-only pixels,not new physical observations or semantic verification.'}
                    if inspection_receipt is not None:
                        request['visual_inspection_delivery']=deepcopy(inspection_receipt)
                        inspection_receipt=None
                if task_continuity:
                    request['task_continuity'] = task_continuity_context(self.report['hypotheses'],
                        self.report['agenda']['active_hypothesis_id'])
                    request['response_roles']['input_only_fields'].append('task_continuity')
            if shared_images:
                from image_slots import share
                request,inputs=share(request,inputs,'views')
                if 'response_roles' in request:
                    request['response_roles']['input_only_fields'].append('image_slot_contract')
            from held_appearance import public_contract
            rendering=public_contract(getattr(env,'held_appearance_contract',None))
            if rendering is not None:
                request['rgb_rendering_contract']=rendering
                if 'response_roles' in request:
                    request['response_roles']['input_only_fields'].append('rgb_rendering_contract')
            if context_evidence_enabled:
                from observation_memory import context_evidence_contract
                from episode_memory import execution_feedback_chain
                request['context_evidence_contract'] = context_evidence_contract()
                request['execution_feedback_chain'] = execution_feedback_chain(self.report['rounds'],
                    operations.records if operations is not None else [])
                request['own_inventory']['snapshot_provenance'] = {
                    'observation_id': current,
                    'captured_utc': memory.views[current].get('captured_utc'),
                    'scope': 'Own inventory sampled during preparation of this request alongside this capture; '
                        'embedded actor identity retains its historical source. Not a new object scan or RGB recognition.'}
                if 'response_roles' not in request:
                    from discovery_protocol import planning_response_roles
                    request['response_roles'] = planning_response_roles(visual_inspection=visual_inspection,
                        operation_history=operation_history_enabled, shared_inputs=shared_inputs)
                request['response_roles']['proposal_output_keys'].append('context_evidence')
                request['response_roles']['input_only_fields'] += ['context_evidence_contract', 'execution_feedback_chain']
            if shared_inputs:request=working_context(request)
            row = {"round": turn, "input": deepcopy(request), "proposals": []}
            self.report["rounds"].append(row)
            if not read_only:
                decision_count += 1
            proposal = None
            for attempt in range(args.format_retries + 1):
                save()
                raw = evidence.invoke("discovery", turn, attempt, request, request, inputs,
                    lambda: client.generate_task_goal(json.dumps(request), inputs), lambda: getattr(client, "last_call_stats", {}),
                    on_error=failed_model_call('planner_generation',current,'proposals'))
                memory.mark_discovery_exposure(identities, f'call-{evidence.sequence:06d}')
                response = {"raw": raw, "model_call_id": f"call-{evidence.sequence:06d}",
                            'contract_valid':None,'actor_invocation_started':False}
                row["proposals"].append(response)
                try:
                    proposal = parser(raw)
                    response["format_valid"] = True
                    break
                except ProtocolError as exc:
                    response.update(format_valid=False, error=str(exc))
                    if planning_v2:
                        format_correction_context(request, raw, exc, response['model_call_id'])
                    else:
                        request["format_correction"] = str(exc) + "; use the same observation, no environment action occurred"
            if proposal is None:
                return "discovery_format_error"
            row["proposal"] = proposal
            error = proposal_guard(proposal, len(inputs), self.report["hypotheses"],
                                   self.report.get('agenda', {}).get('active_hypothesis_id'))
            if planning_v2 and not error:
                error = interactions.check_error(proposal.get('checks',[]), getattr(env, 'alias_to_id', {}))
            bound_evidence = []
            bound_context = [] if context_evidence_enabled else None
            if not error:
                try:
                    bound_evidence = memory.bind(proposal.get("evidence",[]), identities,
                        image_views=request['views'] if shared_images else None)
                    if context_evidence_enabled:
                        from observation_memory import bind_context_evidence
                        bound_context = bind_context_evidence(proposal.get('context_evidence', []), request,
                                                             response['model_call_id'])
                except ValueError as exc:
                    error = str(exc)
            row["bound_evidence"] = bound_evidence
            if context_evidence_enabled: row['bound_context_evidence'] = bound_context
            row["evidence_assessment"] = {"traceability_error": error, "semantic_support": "unreviewed"}
            response['contract_valid']=not bool(error)
            if error:
                rejected_provenance=execution_provenance('planner_validation',operation_source(),
                    native_dispatch_feedback('not_sent'),before_observation_id=current,after_observation_id=current) if planning_v2 else None
                row["status"] = "proposal_guard_rejected"
                if planning_v2:
                    remember_contract_rejection(self.report['hypotheses'], proposal, error, turn,
                                                response['model_call_id'], archive=memory.archive,provenance=rejected_provenance)
                    remember_agenda_selection(self.report, proposal, turn, response['model_call_id'], rejected_draft=True)
                previous = {"success": False, "message": planner_rejection_message(error), **native_dispatch_feedback('not_sent')}
                if planning_v2:
                    previous['execution_provenance']=rejected_provenance
                    previous.update(rejected_proposal=deepcopy(proposal),
                                    proposal_call_id=response['model_call_id'],
                                    instruction='Correct or reconsider your own proposal; no fields were repaired and no physical inquiry occurred.')
                continue
            remember_proposal(self.report["hypotheses"], proposal, bound_evidence, turn,
                              response["model_call_id"], archive=memory.archive, bound_context_evidence=bound_context)
            if planning_v2:
                remember_agenda_selection(self.report,proposal,turn,response['model_call_id'])
            if self.report['model_revision_enabled'] and proposal['decision'] != 'recall':
                row['draft_proposal'] = deepcopy(proposal)
                row['draft_bound_evidence'] = deepcopy(bound_evidence)
                row['draft_model_call_id'] = response['model_call_id']
                revision_request = proposal_revision_request(request, proposal, planner_owned=planning_v2)
                row['revision_input'] = deepcopy(revision_request)
                row['revision_proposals'] = []
                proposal = None
                for attempt in range(args.format_retries + 1):
                    save()
                    raw = evidence.invoke('discovery_revision', turn, attempt, revision_request, revision_request, inputs,
                        lambda: client.generate_task_goal(json.dumps(revision_request), inputs),
                        lambda: getattr(client, 'last_call_stats', {}),
                        on_error=failed_model_call('planner_generation',current,'revision_proposals'))
                    response = {'raw': raw, 'model_call_id': f'call-{evidence.sequence:06d}',
                                'contract_valid':None,'actor_invocation_started':False}
                    row['revision_proposals'].append(response)
                    try:
                        proposal = parser(raw)
                        response['format_valid'] = True
                        break
                    except ProtocolError as exc:
                        response.update(format_valid=False, error=str(exc))
                        if planning_v2:
                            format_correction_context(revision_request, raw, exc, response['model_call_id'])
                        else:
                            revision_request['format_correction'] = str(exc) + '; same RGB, no environment action occurred'
                if proposal is None:
                    return 'discovery_revision_format_error'
                row['proposal'] = proposal
                error = proposal_guard(proposal, len(inputs), self.report['hypotheses'],
                                       self.report.get('agenda', {}).get('active_hypothesis_id'))
                if planning_v2 and not error:
                    error = interactions.check_error(proposal.get('checks',[]), getattr(env, 'alias_to_id', {}))
                bound_evidence = []
                bound_context = [] if context_evidence_enabled else None
                if not error:
                    try:
                        bound_evidence = memory.bind(proposal.get('evidence',[]), identities,
                            image_views=request['views'] if shared_images else None)
                        if context_evidence_enabled:
                            from observation_memory import bind_context_evidence
                            bound_context = bind_context_evidence(proposal.get('context_evidence', []), revision_request,
                                                                 response['model_call_id'])
                    except ValueError as exc:
                        error = str(exc)
                row['bound_evidence'] = bound_evidence
                if context_evidence_enabled: row['bound_context_evidence'] = bound_context
                row['evidence_assessment'] = {'traceability_error': error, 'semantic_support': 'unreviewed'}
                response['contract_valid']=not bool(error)
                if error:
                    rejected_provenance=execution_provenance('planner_validation',operation_source(),
                        native_dispatch_feedback('not_sent'),before_observation_id=current,after_observation_id=current) if planning_v2 else None
                    row['status'] = 'revision_guard_rejected'
                    if planning_v2:
                        remember_contract_rejection(self.report['hypotheses'], proposal, error, turn,
                                                    response['model_call_id'], archive=memory.archive,provenance=rejected_provenance)
                        remember_agenda_selection(self.report, proposal, turn, response['model_call_id'], rejected_draft=True)
                    previous = {'success': False, 'message': planner_rejection_message(error), **native_dispatch_feedback('not_sent')}
                    if planning_v2:
                        previous['execution_provenance']=rejected_provenance
                        previous.update(rejected_proposal=deepcopy(proposal),
                                        proposal_call_id=response['model_call_id'])
                    continue
                remember_proposal(self.report['hypotheses'], proposal, bound_evidence, turn,
                                  response['model_call_id'], archive=memory.archive, bound_context_evidence=bound_context)
            if planning_v2:
                remember_agenda_selection(self.report, proposal, turn, response['model_call_id'])
            if planning_v2 and proposal['decision'] == 'recall':
                if not read_only:
                    decision_count -= 1
                row['recall_request'] = deepcopy(proposal['recall_observation_ids'])
                row['plan_recall_request'] = []  # Historical report key; the wire has one page API.
                row['history_page_requests'] = deepcopy(proposal['history_page_requests'])
                regions=deepcopy(proposal.get('region_requests',[]))
                if regions:row['region_requests']=regions
                recall_event={'model_call_id': response['model_call_id'],
                    'observation_ids': row['recall_request'], 'hypothesis_ids': row['plan_recall_request'],
                    'history_page_requests': row['history_page_requests']}
                if regions:recall_event['region_requests']=regions
                memory.archive.append('recall_request',recall_event)
                try:
                    if recall_count >= recall_limit:
                        raise ValueError('Recall request allowance exhausted; no environment action occurred')
                    # Resolve both first: a bad plan ID cannot partially authorize RGB recall.
                    plan_history = history_pages(self.report['hypotheses'], [],
                                                 proposal['history_page_requests'], page_byte_budget,
                                                 include_field_index=task_continuity)
                    if task_continuity:
                        from episode_memory import validate_history_delivery
                        for page in plan_history: validate_history_delivery(page)
                    rgb_ids = (memory.recall(proposal['recall_observation_ids'])
                               if proposal['recall_observation_ids'] else [])
                    if visual_inspection:
                        parents=[memory.views[i]['visual_inspection']['source_observation_id'] for i in rgb_ids
                                 if 'visual_inspection' in memory.views[i]]
                        rgb_ids=list(dict.fromkeys(parents+rgb_ids))
                        if len(set(latest+rgb_ids))>8:
                            raise ValueError('Recalled region parent/context exceeds eight images;no inspection committed')
                    if regions:
                        if not visual_inspection:raise ValueError('Region inspection is not enabled')
                        parents=[r['observation_id'] for r in regions]
                        prepared=memory.prepare_regions(regions,
                            lambda frame:processor.inspection_canvas(frame,args.frame_width,args.frame_height,
                                encoded=not getattr(client,'supports_native_images',False)),response['model_call_id'])
                        region_ids=[view['observation_id'] for _,view,_ in prepared]
                        memory.commit_regions(prepared,context_ids=latest+rgb_ids)
                        rgb_ids=list(dict.fromkeys(rgb_ids+parents+region_ids))
                    delivered_regions=[i for i in rgb_ids if 'visual_inspection' in memory.views[i]]
                    if delivered_regions:
                        parent_ids=[memory.views[i]['visual_inspection']['source_observation_id'] for i in delivered_regions]
                        inspection_receipt={'source_model_call_id':response['model_call_id'],
                            'prepared_region_ids':delivered_regions,'parent_observation_ids':list(dict.fromkeys(parent_ids)),
                            **native_dispatch_feedback('not_sent'),
                            'scope':'Prepared for this input,not proof of model use or successful interpretation'}
                        row['visual_inspection_delivery']=deepcopy(inspection_receipt)
                        memory.archive.append('visual_inspection_delivery',inspection_receipt)
                    recalled, recalled_plans = rgb_ids, plan_history
                    row.update(status='memory_recall', recalled_views=memory.packet(recalled, current,**view_options),
                               recalled_plan_history=deepcopy(recalled_plans))
                    memory.archive.append('history_page_delivery', {'source_model_call_id': response['model_call_id'],
                        'pages': [{key: page[key] for key in ('hypothesis_id', 'next_after_event_id',
                            'has_more', 'status', 'event_bytes')} for page in plan_history]})
                except ValueError as exc:
                    row.update(status='memory_recall_rejected', recall_error=str(exc))
                    previous = {'success': False, 'message': str(exc), **native_dispatch_feedback('not_sent'),
                        'source_model_call_id': response['model_call_id'],
                        'rejected_memory_request': {'observation_ids': row['recall_request'],
                            'hypothesis_ids': row['plan_recall_request'], 'history_page_requests': row['history_page_requests']}}
                    if regions:previous['rejected_memory_request']['region_requests']=regions
                # Invalid requests also consume recall allowance, never native commands.
                recall_count += 1
                if recall_count > recall_limit:
                    return 'discovery_recall_limit'
                save()
                continue
            pending_end_review = None
            if read_only:
                row['status'] = 'read_only_assessment'
                self.report['final_model_assessment'] = {
                    'proposal': deepcopy(proposal), 'model_call_id': response['model_call_id'],
                    'semantic_support': 'unreviewed', 'native_execution_permitted': False}
                save()
                return 'discovery_assessed_at_limit'
            if proposal["decision"] == "abstain":
                row["status"] = "model_abstained"
                if planning_v2:
                    if proposal['end_episode']:
                        if request_end_review(proposal,response['model_call_id']):
                            row['status']='model_end_requested_review_pending';save();continue
                        self.report['model_episode_end'] = {
                            'model_call_id': response['model_call_id'],
                            'reason': proposal['outcome_reason'], 'task_index': task_index(self.report['hypotheses']),
                            'scope': 'Model-requested end of a bounded episode; open tasks and reported outcomes remain unverified'}
                        memory.archive.append('model_episode_end', self.report['model_episode_end'])
                        return 'discovery_model_requested_end'
                    previous = {**native_dispatch_feedback('not_sent'),
                        'planning_feedback': 'You abstained from this candidate. Other tasks remain as recorded in task_index; no new physical observation occurred.'}
                    save()
                    continue
                return "discovery_abstained"
            if proposal["decision"] == "withdraw":
                row["status"] = "hypothesis_withdrawn"
                previous = {"hypothesis_withdrawn": proposal["hypothesis_id"], **native_dispatch_feedback('not_sent')}
                if planning_v2 and proposal['end_episode']:
                    if request_end_review(proposal,response['model_call_id']):
                        row['status']='model_end_requested_review_pending';save();continue
                    self.report['model_episode_end'] = {'model_call_id': response['model_call_id'],
                        'reason': proposal['outcome_reason'], 'task_index': task_index(self.report['hypotheses']),
                        'scope': 'Model-requested episode end after withdrawal; not verified completion'}
                    memory.archive.append('model_episode_end', self.report['model_episode_end'])
                    return 'discovery_model_requested_end'
                continue
            observe = proposal["decision"] == "observe"
            refresh_operation=None
            if visual_inspection:
                try:
                    anticipated=current if observe and self.report['rgb_observation_actor_enabled'] else '__next_current__'
                    memory.actor_context_ids([e['observation_id'] for e in bound_evidence],anticipated)
                except ValueError as exc:
                    row.update(status='actor_context_rejected',actor_context_error=str(exc))
                    previous={'success':False,'message':str(exc),**native_dispatch_feedback('not_sent'),
                              'proposal_call_id':response['model_call_id'],'rejected_proposal':deepcopy(proposal)}
                    save()
                    continue
            if not observe:
                if len(env.audit) >= args.max_steps:
                    if planning_v2:
                        continue
                    return "discovery_budget_exhausted"
                if operations is not None:
                    refresh_operation=operations.begin({'action':'Pass'},'controller_refresh',operation_source(),
                                                       memory.views[current],len(env.audit))
                    save()
                try:
                    refresh = perform({"action": "Pass"}, "discovery_refresh")
                except ReceivedExecutionError as exc:
                    feedback = deepcopy(exc.execution)
                    row.update(status='refresh_postprocessing_failed',refresh_execution=feedback)
                    if operations is not None:
                        retain_after_failure(exc,feedback,'refresh_operation_feedback',
                            lambda: operations.finish(refresh_operation,feedback,len(env.audit)))
                    retain_after_failure(exc,feedback,'refresh_checkpoint',save)
                    raise
                except BaseException as exc:
                    feedback = deepcopy(exc.execution) if isinstance(exc, NativeDispatchError) else {'message':str(exc)}
                    row.update(status='refresh_dispatch_failed',refresh_execution=feedback)
                    if operations is not None:
                        unknown = feedback.get('native_dispatch_status') != 'not_sent'
                        retain_after_failure(exc,feedback,'refresh_operation_feedback',
                            lambda: operations.finish(refresh_operation,feedback,len(env.audit),unknown=unknown))
                    retain_after_failure(exc,feedback,'refresh_checkpoint',save)
                    raise
                if operations is not None:
                    operations.finish(refresh_operation,refresh,len(env.audit))
                    save()
                if not refresh["success"]:
                    return "discovery_refresh_failed"
            if len(env.audit) >= args.max_steps:
                if planning_v2:
                    row['status'] = 'native_budget_before_actor'
                    save()
                    continue
                return "discovery_budget_exhausted"
            scene = observation_packet(env.get_visible_objects_summary())
            if capability_feasibility:
                if observe:
                    scene['operation_access']='capability-feasibility-v1'
                else:
                    event=getattr(env,'last_event',None)
                    scene=actor_operation_scene(scene,
                        getattr(event,'metadata',{}).get('objects',[]),
                        getattr(event,'instance_masks',None),getattr(env,'_alias',None))
            actor_aliases = deepcopy(getattr(env, 'alias_to_id', {}))
            if planning_v2:
                scene['allow_planning_feedback'] = True
                if target_reports is not None:
                    scene['allow_target_reports'] = True
                if target_correspondence_enabled:scene['allow_target_correspondence']=True
            # The actor may see native affordances only after a hypothesis exists.
            scene["allowed_action_names"] = (sorted(OBSERVATION_ACTIONS | ({'RequestReplan'} if planning_v2 else set()))
                                             if observe else None)
            task = proposal["goal"]
            before_checks = deepcopy(env.snapshot())
            check_bindings = bind_model_checks(proposal['checks'], env, before_checks)
            prompt = policy_request(task, completion_managed=False, allow_destructive=args.allow_destructive_tools)
            prompt['model_checks'] = deepcopy(proposal['checks'])
            prompt['model_check_scope'] = CHECK_SCOPE
            if previous is not None:
                prompt["previous_execution"] = previous
            prompt["hypothesis"] = {k: proposal[k] for k in ("hypothesis_id", "hypothesis", "reason", "evidence", "missing_information")}
            prompt["hypothesis"]["evidence"] = bound_evidence
            if context_evidence_enabled:
                from observation_memory import project_context_evidence
                prompt['hypothesis']['context_evidence'] = project_context_evidence(bound_context,
                    delivered_in_call_id=response['model_call_id'])
                prompt['context_evidence_scope'] = ('Exact non-RGB scalar values delivered to the originating planner input. '
                    'Delivery call and declared data author are separate;original roles survive history quotation. '
                    'Historical feedback and model claims remain historical/unverified; '
                    'not current native state, target-match confirmation or goal completion. No RGB is attached by these citations.')
                if 'execution_feedback_chain' in request:
                    prompt['execution_feedback_chain'] = deepcopy(request['execution_feedback_chain'])
            if planning_v2:
                prompt['task_goal'] = proposal['task_goal']
                prompt['plan_steps'] = deepcopy(proposal['plan_steps'])
                prompt['planning_source'] = {'model_call_id': response['model_call_id'],
                    'scope': 'Model-authored task/steps and next objective; not controller-supplied instructions or verified progress'}
                prompt['planning_feedback_instruction'] = ('RequestReplan(reason) is available to report your own '
                    'observed blocker, uncertainty or disagreement without a native command. '
                    'Past execution is historical, not automatic fulfillment of this newly issued objective. '
                    'If you judge the objective already satisfied, report the supporting evidence instead '
                    'of advancing to a later plan step. Do not substitute an unsupported object or objective '
                    'just to issue an action. The original planner decides whether to inspect or revise; '
                    'your report is unverified and does not certify impossibility or completion.')
            prior_execution = self.report["hypotheses"].get(proposal["hypothesis_id"], {}).get("last_execution")
            if prior_execution is not None:
                prompt["hypothesis"]["previous_execution"] = deepcopy(prior_execution)
            prompt["decision"] = proposal["decision"]
            if capability_provenance:
                prompt['task_provenance'] = planner_authored_task_provenance(
                    proposal['hypothesis_id'], response['model_call_id'], turn)
                prompt['robot_operation_contract'] = deepcopy(request['robot_operation_contract'])
            if shared_inputs:prompt['input_reference_contract']=deepcopy(request['input_reference_contract'])
            if task_continuity:
                prompt['task_continuity'] = task_continuity_context(
                    self.report['hypotheses'],proposal['hypothesis_id'])['active']
                prompt['planning_memory_provenance'] = {
                    'proposal_call_id':response['model_call_id'],
                    'received_history':[{'hypothesis_id':page['hypothesis_id'],
                                         'delivery_receipt':deepcopy(page['delivery_receipt'])}
                                        for page in request.get('recalled_plan_history',[])],
                    'scope':'Exact content supplied to the originating planner call;not proof of use,understanding or current scene state'}
            prompt["simulator_assistance"] = "Current native affordances are execution preconditions, not independent proof of the problem"
            rgb_observe = observe and self.report['rgb_observation_actor_enabled']
            actor_current = current if rgb_observe else memory.current(env)
            if operations is not None:
                if refresh_operation is not None:
                    operations.attach_after(refresh_operation,memory.views[actor_current])
                    save()
                prompt['operation_history']=operations.context()
            actor_ids = list(dict.fromkeys([e["observation_id"] for e in bound_evidence] + [actor_current]))
            if visual_inspection:
                actor_ids=memory.actor_context_ids([e['observation_id'] for e in bound_evidence],actor_current)
                prompt['visual_inspection_contract']=deepcopy(request['visual_inspection_contract'])
            prompt["image_views"] = memory.packet(actor_ids, actor_current,**view_options)
            prompt["image_scope"] = "Historical evidence may be stale. Only current=true corresponds to current native aliases; compare pose/time and RGB before acting. Evidence cues are model hypotheses, not verified facts."
            if rgb_observe:
                # No environment command intervenes for observe. Reuse the exact
                # recorded input, including its ID/index order; not a new capture.
                actor_current = current
                actor_ids = list(identities)
                prompt, scene = rgb_observation_input(prompt,scene,request)
                row['actor_input_mode'] = 'same_recorded_rgb_own_body_stateless'
            if target_correspondence_enabled:
                from observation_memory import actor_selection_contract
                prompt['actor_selection_contract']=actor_selection_contract(actor_current)
            if target_reports is not None:
                from observation_memory import historical_feedback
                for field in ('previous_execution','operation_history','hypothesis'):
                    if field in prompt:prompt[field]=historical_feedback(prompt[field])
            actor_tools = available_tools(scene, False, args.allow_destructive_tools)
            actor_tool_scope = actor_offer_contract(actor_tools, proposal['decision'],
                'recorded_rgb_own_body' if rgb_observe else 'native_assisted_actor',
                provenance=({'proposal_call_id': response['model_call_id'],
                    'actor_call_id': f'call-{evidence.sequence+1:06d}', 'observation_id': actor_current}
                    if capability_provenance else None),scene=scene if capability_feasibility else None)
            prompt['actor_tool_scope'] = deepcopy(actor_tool_scope)
            row["actor_image_views"] = prompt["image_views"]
            action_frames = process([memory.frames[i] for i in actor_ids], args.frame_width, args.frame_height)
            if rendering is not None:prompt['rgb_rendering_contract']=deepcopy(rendering)
            if shared_images:
                from image_slots import share
                # RGB observer copies the planner's already-shared views. Build
                # this call's original ordered bindings before its own mapping.
                prompt['image_views']=memory.packet(actor_ids,actor_current,**view_options)
                prompt,action_frames=share(prompt,action_frames,'image_views')
                row['actor_image_views']=deepcopy(prompt['image_views'])
            if shared_inputs:
                prompt=working_context(prompt,actor=True)
                actor_tool_scope=deepcopy(prompt['actor_tool_scope'])
            parsed = None
            for attempt in range(args.format_retries + 1):
                if capability_provenance:
                    # Formatting correction is another model call against the
                    # same prepared offer, not a new affordance observation.
                    actor_tool_scope['provenance']['actor_call_id'] = f'call-{evidence.sequence+1:06d}'
                    prompt['actor_tool_scope'] = deepcopy(actor_tool_scope)
                response['actor_invocation_started']=True
                parsed, raw = evidence.invoke("action", turn, attempt, prompt, scene, action_frames,
                    lambda: client.chat(prompt, json.dumps(scene), action_frames), lambda: getattr(client, "last_call_stats", {}),
                    on_error=failed_model_call('actor_decoding',actor_current))
                actor_call_id = f'call-{evidence.sequence:06d}'
                if parsed["action"]["action"] != "FormatError":
                    break
                prompt["formatting_correction"] = parsed.get("format_error", "Invalid action")
            if parsed["action"]["action"] == "FormatError":
                row.update(status="action_format_rejected", action_error=parsed.get("format_error", "Invalid action"))
                previous = {"success": False, "message": row["action_error"], **native_dispatch_feedback('not_sent'),
                            "instruction": "Re-evaluate your observation objective or hypothesis. No physical action occurred."}
                previous['actor_tool_scope'] = (actor_offer_history_reference(actor_tool_scope,
                    self.report['hypotheses'][proposal['hypothesis_id']]) if target_reports_enabled
                    else deepcopy(actor_tool_scope))
                previous['execution_provenance']=execution_provenance('actor_decoding',
                    operation_source(actor_call_id),previous,
                    before_observation_id=actor_current,after_observation_id=actor_current)
                if operations is not None:
                    identity=operations.begin({'action':'FormatError'},'model_control',
                        operation_source(actor_call_id),memory.views[actor_current],len(env.audit))
                    operations.finish(identity,{'success':False,'category':'action_format_rejected',
                        'error':row['action_error'],**native_dispatch_feedback('not_sent'),
                        'execution_provenance':deepcopy(previous['execution_provenance'])},len(env.audit),memory.views[actor_current])
                    previous['operation_history_reference']=operations.reference(operation_start)
                    previous['instruction']='Re-evaluate your observation objective or hypothesis. No actor native command occurred;any recorded controller refresh remains part of actual history.'
                if capability_provenance:
                    previous['task_provenance'] = deepcopy(prompt['task_provenance'])
                if task_continuity:
                    previous['task_continuity_reference'] = task_continuity_reference(prompt['task_continuity'])
                    if 'planning_memory_provenance' in prompt:
                        previous['planning_memory_provenance'] = deepcopy(prompt['planning_memory_provenance'])
                previous['model_check_progress'] = model_check_feedback(proposal, check_bindings, before_checks,
                    env.snapshot(), turn, response['model_call_id'], actor_call_id)
                row['model_check_progress'] = deepcopy(previous['model_check_progress'])
                remember_execution(self.report["hypotheses"], proposal, turn, previous,
                                   actor_call_id, archive=memory.archive,
                                   operation_records=operations.since(operation_start) if operations is not None else None,
                        actor_input_offer=prompt['actor_tool_scope'] if target_reports_enabled else None)
                save()
                continue
            raw_action=parsed['action']
            action={k:deepcopy(v) for k,v in raw_action.items() if k!='selection_report'}
            selection_record=None
            audit_before_action = len(env.audit)
            conditions = compile_checks(proposal["checks"], env)
            row["goal_verification_conditions"] = conditions
            operation_identity=None
            if operations is not None:
                operation_identity=operations.begin(action,
                    'model_control' if action['action'] in {'RequestReplan','Done'} else 'model_actor',
                    operation_source(actor_call_id),memory.views[actor_current],len(env.audit))
                save()
            received_error = None
            dispatch_entered = False
            decision_review = None
            try:
                validate_available_action(raw_action, actor_tools)
                if target_correspondence_enabled:
                    from observation_memory import bind_actor_selection
                    selection_record=bind_actor_selection(raw_action,prompt,memory,actor_ids,actor_current,
                                                         operation_source(actor_call_id))
                if observe and action["action"] not in (OBSERVATION_ACTIONS | ({'RequestReplan'} if planning_v2 else set())):
                    raise ProtocolError("Observation objective only permits body/camera motion, Pass, or bounded held-object translation")
                if label_review_enabled and 'objectId' in action:
                    from decision_review import review_selection
                    review_packet={'actor_context':deepcopy(prompt),'actor_observation':deepcopy(scene),
                                   'selected_action':deepcopy(raw_action), 'actor_call_id':actor_call_id}
                    def invoke_review(role,packet):
                        import time
                        review_started=time.monotonic()
                        assessment,raw_review=evidence.invoke(role,turn,0,packet,packet,action_frames,
                            lambda: client.review_dispute(role,packet,action_frames),
                            lambda: getattr(client,'last_call_stats',{}))
                        stats=getattr(client,'last_call_stats',{})
                        cost={key:stats[key] for key in ('input_tokens','output_tokens','reasoning_tokens','seconds')
                              if isinstance(stats.get(key),(int,float)) and not isinstance(stats.get(key),bool)}
                        cost['host_elapsed_seconds']=time.monotonic()-review_started
                        return {'version':'review-model-assessment-v1','role':role,
                                'call_id':f'call-{evidence.sequence:06d}','assessment':assessment,'cost':cost}
                    decision_review=review_selection(invoke_review,review_packet,actor_ids)
                    row['decision_review']=deepcopy(decision_review)
                if decision_review is not None and not decision_review['release_current_action']:
                    execution={'success':False,'category':'model_selection_review',
                        'error':'Independent selection review withheld this command. Reconsider the provisional labels and evidence; select new observations if unresolved.',
                        **native_dispatch_feedback('not_sent')}
                elif action['action'] == 'RequestReplan':
                    execution = {'success': False, 'category': 'model_replan_request',
                        'error': action['reason'], **native_dispatch_feedback('not_sent')}
                elif action["action"] == "Done":
                    execution = {"success": False, "category": "evaluation_request", "error": "Re-evaluate the hypothesis; Done does not certify discovery", **native_dispatch_feedback('not_sent')}
                else:
                    dispatch_entered = True
                    execution = perform(action, "discovery_observation" if observe else "discovery_policy")
            except ReceivedExecutionError as exc:
                received_error = exc
                execution = exc.execution
            except ProtocolError as exc:
                if target_correspondence_enabled:
                    from observation_memory import rejected_actor_selection
                    selection_record=rejected_actor_selection(raw_action,operation_source(actor_call_id),actor_current,exc)
                execution = {"success": False, "category": "precondition_not_met", "error": str(exc), **native_dispatch_feedback('not_sent')}
            except BaseException as exc:
                unknown_feedback={'action':deepcopy(action),'message':str(exc),'success':None,
                                  'category':None,**native_dispatch_feedback('outcome_unknown' if dispatch_entered else 'not_sent')}
                if isinstance(exc, NativeDispatchError):
                    unknown_feedback.update(deepcopy(exc.execution))
                effect_unknown = unknown_feedback['native_dispatch_status'] == 'outcome_unknown'
                unknown_feedback['execution_provenance']=execution_provenance('actor_handling',
                    operation_source(actor_call_id),unknown_feedback,
                    before_observation_id=actor_current,selection=actor_selection_reference(action,scene,actor_current,actor_call_id))
                if selection_record is not None:unknown_feedback['actor_selection_report']=deepcopy(selection_record)
                if decision_review is not None:unknown_feedback['decision_review']=deepcopy(decision_review)
                row['status']=('native_effect_unknown' if effect_unknown else
                    'actor_dispatch_not_sent' if dispatch_entered else 'actor_preflight_failed')
                row['failed_execution_feedback'] = unknown_feedback
                if operations is not None:
                    def retain_operation():
                        operations.finish(operation_identity,unknown_feedback,len(env.audit),unknown=effect_unknown)
                        unknown_feedback['operation_history_reference']=operations.reference(operation_start)
                    retain_after_failure(exc,unknown_feedback,'operation_feedback',retain_operation)
                retain_after_failure(exc,unknown_feedback,'task_memory',lambda: remember_execution(self.report['hypotheses'],proposal,turn,unknown_feedback,
                    actor_call_id,archive=memory.archive,
                    operation_records=operations.since(operation_start) if operations is not None else None,
                        actor_input_offer=prompt['actor_tool_scope'] if target_reports_enabled else None))
                retain_after_failure(exc,unknown_feedback,'failure_checkpoint',save)
                raise
            previous = {"action": action, "success": execution["success"], "message": execution.get("error", ""),
                        **project_dispatch_feedback(execution),
                        "category": execution.get("category"),
                        "observation_change": None}
            previous['actor_tool_scope'] = (actor_offer_history_reference(actor_tool_scope,
                self.report['hypotheses'][proposal['hypothesis_id']]) if target_reports_enabled
                else deepcopy(actor_tool_scope))
            if decision_review is not None:previous['decision_review']=deepcopy(decision_review)
            if selection_record is not None:previous['actor_selection_report']=deepcopy(selection_record)
            previous['execution_provenance']=execution_provenance('actor_handling',
                operation_source(actor_call_id),previous,
                before_observation_id=actor_current,
                selection=actor_selection_reference(action,scene,actor_current,actor_call_id))
            if capability_provenance:
                previous['task_provenance'] = deepcopy(prompt['task_provenance'])
            if task_continuity:
                previous['task_continuity_reference'] = task_continuity_reference(prompt['task_continuity'])
                if 'planning_memory_provenance' in prompt:
                    previous['planning_memory_provenance'] = deepcopy(prompt['planning_memory_provenance'])
            if 'physical_time' in execution:
                previous['physical_time'] = deepcopy(execution['physical_time'])
            after=None
            processing_stage='observation_capture'
            try:
                if received_error is not None:
                    processing_stage = received_error.stage
                    raise received_error.cause
                after = memory.current(env)
                processing_stage='observation_projection'
                progress = memory.feedback(actor_current, after)
                previous['observation_change']=progress
                previous['execution_provenance']=execution_provenance('actor_handling',
                    operation_source(actor_call_id),previous,
                    before_observation_id=actor_current,after_observation_id=after,
                    selection=actor_selection_reference(action,scene,actor_current,actor_call_id))
                processing_stage='target_report'
                if target_reports is not None and execution.get('category') == 'model_replan_request':
                    target_report = target_reports.remember(action, scene,
                        actor_call_id=actor_call_id,
                        proposal_call_id=response['model_call_id'], observation_id=actor_current)
                    if target_report is not None:
                        previous['actor_target_report'] = target_report
                        previous['actor_target_report_id'] = target_report['report_id']
                        row['actor_target_report'] = deepcopy(target_report)
                        memory.archive.append('actor_target_report', target_report)
                processing_stage='operation_feedback'
                if operations is not None:
                    operations.finish(operation_identity,previous,len(env.audit),memory.views[after])
                    previous['operation_history_reference']=operations.reference(operation_start)
                processing_stage='model_check'
                previous['model_check_progress'] = model_check_feedback(proposal, check_bindings, before_checks,
                    env.snapshot(), turn, response['model_call_id'], actor_call_id)
                processing_stage='actor_control_report'
                if action['action'] == 'RequestReplan':
                    previous['actor_planning_report'] = {'reason': action['reason'],
                        'model_call_id': actor_call_id,
                        'input_mode': 'recorded_rgb_own_body' if rgb_observe else 'native_assisted_actor',
                        'scope': 'Only reason is exact model-authored text; model_call_id, input_mode and this scope are controller metadata. The reason is not an independently verified fact or a native engine failure. No actor native command ran; a preceding act refresh may already have run.'}
                processing_stage='interaction_memory'
                if planning_v2 and not observe:
                    reference = interactions.remember(action, scene, actor_aliases,
                        actor_current, actor_call_id, execution.get('environment_command_executed') is True,
                        share_actor_identity=True, execution_feedback=previous)
                    if reference is not None:
                        previous['interaction_reference'] = reference
                        row['interaction_reference'] = deepcopy(reference)
                        memory.archive.append('interaction_reference', reference)
                row['model_check_progress'] = deepcopy(previous['model_check_progress'])
                processing_stage='task_memory'
                remember_execution(self.report["hypotheses"], proposal, turn, previous,
                                   actor_call_id, archive=memory.archive,
                                   operation_records=operations.since(operation_start) if operations is not None else None,
                        actor_input_offer=prompt['actor_tool_scope'] if target_reports_enabled else None)
                row["observation_change"] = progress
                processing_stage='task_verification'
                row.update(status=('planner_reconsideration_requested' if action['action'] == 'RequestReplan'
                                   else 'observed' if observe else 'action_attempted'), action=action, execution=execution,
                           goal_verification=verify_task(env.snapshot(), conditions, baseline, args.allow_destructive_tools))
                processing_stage='action_record'
                record({"kind": "open_discovery_policy", "round": turn, "action": action, "raw_response": raw,
                        "model_call_id": actor_call_id, **execution})
                processing_stage='preservation_verification'
                if execution.get("category") in {"preservation_violation", "preservation_unknown"}:
                    return "constraint_violation" if execution['category']=='preservation_violation' else 'preservation_unknown'
                if verify_task(env.snapshot(), None, baseline, args.allow_destructive_tools)["preservation"]["status"] == "violation":
                    return "constraint_violation"
                processing_stage='manifest_save'
                save()
            except BaseException as exc:
                # This action already returned a receipt. Later processing failure
                # must never turn that known result into unknown/retryable native IO.
                failure=postprocessing_failure(processing_stage,exc)
                previous['post_execution_processing_error']=failure
                observation_failed=processing_stage in {'observation_capture','observation_projection','native_observation_capture'}
                if observation_failed:previous['post_execution_observation_error']=str(exc)
                previous['execution_provenance']=execution_provenance('actor_handling',
                    operation_source(actor_call_id),previous,
                    before_observation_id=actor_current,after_observation_id=after,
                    selection=actor_selection_reference(action,scene,actor_current,actor_call_id))
                row.update(status='post_execution_observation_failed' if observation_failed else 'post_execution_processing_failed',
                    action=deepcopy(action),execution=deepcopy(execution),post_execution_processing_error=failure)
                if observation_failed:row['post_execution_observation_error']=str(exc)
                row['failed_execution_feedback'] = previous
                if operations is not None:
                    def retain_operation_failure():
                        operations.retain_processing_failure(operation_identity,previous,len(env.audit),
                            memory.views[after] if after is not None else None)
                        previous['operation_history_reference']=operations.reference(operation_start)
                    retain_after_failure(exc,previous,'operation_processing_failure',retain_operation_failure)
                retain_after_failure(exc,previous,'task_memory',lambda: remember_execution(self.report['hypotheses'],proposal,turn,previous,
                    actor_call_id,archive=memory.archive,
                    operation_records=operations.since(operation_start) if operations is not None else None,
                        actor_input_offer=prompt['actor_tool_scope'] if target_reports_enabled else None))
                retain_after_failure(exc,previous,'failure_checkpoint',save)
                raise
        return "discovery_round_limit"
