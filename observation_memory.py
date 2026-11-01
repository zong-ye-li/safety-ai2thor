"""RGB/camera provenance and descriptive progress, without problem or action rules."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import time
import numpy as np

CAMERA_KEYS = ("frame", "observation_id", "captured_utc", "monotonic_seconds",
               "position", "rotation", "yaw", "horizon", "physical_time")

CONTEXT_EVIDENCE_VERSION = 'delivered-input-field-evidence-v2'
SCALAR_ORIGIN_VERSION='delivered-scalar-origin-v1'
SCALAR_SOURCE_ROLES={'unclassified_delivered_value','actor_model_claim',
    'actor_rgb_cue_binding_not_pixel_delivery','controller_reference_validation',
    'native_assisted_actor_input_snapshot','controller_recorded_provenance',
    'native_io_result','native_outcome_unknown','controller_result_without_native_outcome','planner_model_claim','reviewer_model_claim'}
SCALAR_ORIGIN_REQUIRED={'version','role','author_model_call_id'}
SCALAR_ORIGIN_OPTIONAL={'planner_call_id','actor_call_id','record_observation_id',
    'intent_sha256','snapshot_role','command_id','content_kind'}
CONTEXT_SOURCES = {
    'own_inventory': 'Own inventory at this planner input; embedded identity is historical actor assistance.',
    'known_interactions': 'Historical selected interaction input/result, not current state or benefit.',
    'actor_target_reports': 'Historical actor-selected identity/affordance and model report, not an interaction or verified claim.',
    'previous_execution': 'Recorded controller feedback and explicitly unverified model drafts/reports; retain nested source roles.',
    'operation_history': 'Historical command/feedback records; model intent is unverified and not a current offer.',
    'execution_feedback_chain': 'Controller causal index of prior proposals/actual operations; not effects or task verification.',
    'prior_hypotheses': 'Unverified model-authored task/assessment and recorded feedback, not observed truth.',
    'recalled_plan_history': 'Delivered exact task-history text with original roles; not new facts.'}


def context_evidence_contract():
    return {'version': CONTEXT_EVIDENCE_VERSION, 'source_roots': deepcopy(CONTEXT_SOURCES),
        'scope': 'input_path selects one scalar in this delivered logical request using JSON pointer '
            'escaping (~0 for tilde, ~1 for slash). Array positions are zero-based. Only present '
            'fields may be cited; no native lookup, history read or RGB access occurs. Binding '
            'preserves the exact value. delivered_in_call_id identifies delivery, not authorship; '
            'data_origin records only declared writer/call metadata. Missing author stays unknown. '
            'Controller reference validation,actor claims and native receipts remain different;none verifies a cue or benefit.',
        'task_field_mapping': {'proposal_output': 'hypothesis', 'prior_hypotheses_summary': 'claim',
            'history_event': '/proposal/hypothesis',
            'scope': 'Schema mapping only. Cite a present path yourself; no alias resolution or missing field repair. '
                'Summary field_lineage separates retained content, accepted assessment and rejected draft authors.'},
        'content_kind_scope':'data_origin.role identifies provenance; content_kind identifies schema-defined logical use. '
            'An intent is a requested objective, a hypothesis is a candidate explanation, a step status/outcome is a model report. '
            'None becomes an observation, native result or current command merely by being recalled or cited.'}


def _empty_scalar_origin():
    return {'version':SCALAR_ORIGIN_VERSION,'role':'unclassified_delivered_value',
        'author_model_call_id':None}


def _valid_scalar_origin(origin):
    from discovery_protocol import MODEL_CONTENT_KINDS
    # Earlier full nullable records remain readable and are never rewritten.
    return (isinstance(origin,dict) and SCALAR_ORIGIN_REQUIRED<=set(origin)
        and set(origin)<=SCALAR_ORIGIN_REQUIRED|SCALAR_ORIGIN_OPTIONAL
        and origin.get('version')==SCALAR_ORIGIN_VERSION
        and all(v is None or isinstance(v,str) for v in origin.values())
        and origin.get('role') in SCALAR_SOURCE_ROLES
        and (origin.get('content_kind') is None or origin['content_kind'] in MODEL_CONTENT_KINDS))


def _scalar_origin(node,part,previous):
    """Ownership from controller-owned record structure,never cue/value content."""
    if not isinstance(node,dict):return previous
    version=node.get('version');source=node.get('source')
    source=source if isinstance(source,dict) else {}
    from discovery_protocol import planner_content_kind
    role=None;author=None;content_kind=None
    if version=='review-model-assessment-v1':
        result=_empty_scalar_origin()
        result.update(role='reviewer_model_claim' if part=='assessment' else 'controller_recorded_provenance',
                      author_model_call_id=node.get('call_id') if part=='assessment' else None)
        return result
    if version in {'selection-review-gate-v1','provisional-label-review-v1'}:
        result=_empty_scalar_origin()
        result.update(role='controller_recorded_provenance')
        if version=='provisional-label-review-v1' and part=='provisional_label' and node.get(part) is not None:
            result.update(role='reviewer_model_claim',author_model_call_id=node.get('label_author_call_id'))
        if part=='original_dispute':result.update(role='unclassified_delivered_value')
        return result
    if (previous.get('snapshot_role')=='actor_planning_report'
            and 'reason' in node and 'model_call_id' in node):
        # The report container is controller-owned; only its reason is copied
        # from the actor. Keep this boundary in selected-container origins too.
        result=_empty_scalar_origin()
        result.update(role='controller_recorded_provenance')
        actor=previous.get('actor_call_id')
        if isinstance(actor,str):result['actor_call_id']=actor
        if part=='reason':
            result.update(role='actor_model_claim',author_model_call_id=actor)
        return result
    if previous.get('content_kind')=='model_plan' and part in {'objective','status','note','id'}:
        result=deepcopy(previous);result['content_kind']=planner_content_kind(['plan_steps','0',part]);return result
    if previous.get('role')=='planner_model_claim' and previous.get('content_kind') is None:
        content_kind=planner_content_kind([part])
        if content_kind is not None:
            result=deepcopy(previous);result['content_kind']=content_kind;return result
    lineage=node.get('field_lineage')
    if isinstance(lineage,dict) and lineage.get('version') in {'model-task-field-lineage-v1','model-task-field-lineage-v2'} and part in lineage.get('fields',{}):
        record=lineage['fields'][part]
        raw=json.dumps(node.get(part),ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
        if hashlib.sha256(raw.encode('utf-8')).hexdigest()!=record.get('value_sha256'):
            raise ValueError('Task field differs from its declared authored projection')
        role='planner_model_claim';author=record.get('model_call_id')
        content_kind=record.get('content_kind')
        source={'planner_call_id':author}
    elif 'field_lineage' in node and part in {'field_lineage','latest_event','assessment_source_call_id',
            'source_round','content_source_round','status','task_state','contract_valid','pending_contract_draft'}:
        role='controller_recorded_provenance'
    elif node.get('kind') in {'proposal','revision','withdrawal','contract_rejection'} and 'model_call_id' in node:
        if part=='proposal':
            role='planner_model_claim';author=node.get('model_call_id');source={'planner_call_id':author}
        elif part not in {'execution','bound_evidence','bound_context_evidence'}:
            role='controller_recorded_provenance'
    elif version in ('delivered-input-field-evidence-v1',CONTEXT_EVIDENCE_VERSION):
        if part=='value':
            existing=node.get('data_origin')
            if version==CONTEXT_EVIDENCE_VERSION and _valid_scalar_origin(existing):
                return deepcopy(existing)
            return _empty_scalar_origin()
        if part in {'cue','input_path'}:
            role='planner_model_claim';author=context_delivery_call_id(node)
            content_kind='model_interpretation' if part=='cue' else 'model_reference'
            source={'planner_call_id':author}
        else:role='controller_recorded_provenance'
    elif version in (SCALAR_ORIGIN_VERSION,'historical-actor-offer-reference-v1'):
        role='controller_recorded_provenance'
    elif version in ('actor-intent-selection-v1','actor-selected-target-report-v1'):
        if part in {'model_report','model_assessment'}:
            role='actor_model_claim';author=source.get('actor_call_id',node.get('actor_call_id'))
        elif part in {'bound_evidence'}:role='actor_rgb_cue_binding_not_pixel_delivery'
        elif part in {'binding_status','binding_error'}:role='controller_reference_validation'
        elif part in {'actor_observed_target'}:role='native_assisted_actor_input_snapshot'
        else:role='controller_recorded_provenance'
    elif version=='decision-execution-provenance-v1':
        role='native_assisted_actor_input_snapshot' if part=='selected_argument' else 'controller_recorded_provenance'
    elif node.get('native_dispatch_version')=='native-dispatch-result-v1':
        provenance=node.get('execution_provenance')
        provenance=provenance if isinstance(provenance,dict) else {}
        source=provenance.get('source',{})
        source=source if isinstance(source,dict) else {}
        if part=='rejected_proposal' and provenance.get('stage')=='planner_validation':
            role='planner_model_claim'
            author=source.get('planner_call_id')
            if author!=node.get('proposal_call_id'):author=None
        elif part=='actor_planning_report':
            report=node.get(part)
            result=_empty_scalar_origin()
            result.update(role='controller_recorded_provenance',snapshot_role='actor_planning_report')
            actor=report.get('model_call_id') if isinstance(report,dict) else None
            if isinstance(actor,str):result['actor_call_id']=actor
            return result
        elif part in {'success','category','message','error','native_dispatch_status','native_outcome_received',
                       'native_action_command_id','environment_command_executed','native_dispatch_version'}:
            if part in {'message','error'} and node.get('category')=='model_replan_request':
                role='actor_model_claim';author=source.get('actor_call_id')
            else:
                role={'outcome_received':'native_io_result','outcome_unknown':'native_outcome_unknown',
                      'not_sent':'controller_result_without_native_outcome'}.get(node.get('native_dispatch_status'))
    elif node.get('assessment_source')=='model_only_unverified' and part in {
            'original_intent','changed_current_intent','original_plan_steps','current_plan_steps'}:
        role='planner_model_claim'
        author=node.get('original_model_call_id') if part.startswith('original_') else node.get('last_accepted_plan_call_id')
    elif 'operation_id' in node and 'actor_call_id' in node and part in {'model_target_assessment','selection_reference_status'}:
        role='actor_model_claim' if part=='model_target_assessment' else 'controller_reference_validation'
        if part=='model_target_assessment':author=node.get('actor_call_id')
    if role is None:return previous
    result=_empty_scalar_origin();result.update(role=role,author_model_call_id=author if isinstance(author,str) else None)
    if content_kind is not None:result['content_kind']=content_kind
    for key in ('planner_call_id','actor_call_id','intent_sha256'):
        value=source.get(key,node.get(key))
        if isinstance(value,str):result[key]=value
    for key,value in [('record_observation_id',node.get('actor_observation_id',node.get('before_observation_id',node.get('observation_id')))),
                      ('snapshot_role',node.get('snapshot_role')),('command_id',node.get('native_action_command_id'))]:
        if isinstance(value,str):result[key]=value
    return result


def context_delivery_call_id(record):
    if record.get('version')=='delivered-input-field-evidence-v1':return record.get('source_model_call_id')
    if record.get('version')==CONTEXT_EVIDENCE_VERSION:return record.get('delivered_in_call_id')
    raise ValueError('Unsupported bound context-evidence version')


def history_field_origin(event, path):
    """Retain ancestor-derived roles when an exact history subtree is selected."""
    value=event
    origin=_empty_scalar_origin()
    for part in _context_pointer_parts(path):
        origin=_scalar_origin(value,part,origin)
        if isinstance(value,dict) and part in value:value=value[part]
        elif isinstance(value,list) and part.isascii() and part.isdigit() and str(int(part))==part and int(part)<len(value):
            value=value[int(part)]
        else:raise ValueError('History origin path unavailable: '+path)
    raw=json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
    return {'version':'selected-history-origin-v1','value_sha256':hashlib.sha256(raw.encode()).hexdigest(),
            'data_origin':origin}


def selected_history_origin(record,value):
    raw=json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
    if (not isinstance(record,dict) or set(record)!={'version','value_sha256','data_origin'}
            or record.get('version')!='selected-history-origin-v1'
            or record.get('value_sha256')!=hashlib.sha256(raw.encode()).hexdigest()
            or not _valid_scalar_origin(record.get('data_origin'))):
        raise ValueError('Selected history field differs from its origin binding')
    return deepcopy(record['data_origin'])


def _context_pointer_parts(path):
    if not isinstance(path,str) or not path.startswith('/'):
        raise ValueError('expected an absolute JSON pointer into this request')
    parts=path[1:].split('/')
    for part in parts:
        for offset,char in enumerate(part):
            if char=='~' and (offset+1==len(part) or part[offset+1] not in '01'):
                raise ValueError('invalid JSON pointer escape')
    return [part.replace('~1','/').replace('~0','~') for part in parts]


def project_context_evidence(rows,*,delivered_in_call_id=None):
    """Receiver/history check exact typed values;never invent legacy authorship."""
    if not isinstance(rows,list):raise ValueError('Bound context evidence must be a list')
    for record in rows:
        if not isinstance(record,dict):raise ValueError('Bound context evidence must be an object')
        root=_context_pointer_parts(record.get('input_path'))[0]
        if root not in CONTEXT_SOURCES or root!=record.get('source_root'):
            raise ValueError('Context citation source root/path changed after delivery')
        call_id=context_delivery_call_id(record)
        if (not isinstance(call_id,str) or not call_id
                or delivered_in_call_id is not None and call_id!=delivered_in_call_id):
            raise ValueError('Context evidence belongs to a different delivery call')
        if 'value' not in record or isinstance(record['value'],(dict,list)):
            raise ValueError('Context evidence must retain a present scalar,including explicit null')
        if record.get('semantic_support')!='unverified_model_cue_not_a_controller_conclusion':
            raise ValueError('Context citation cannot become a verified semantic conclusion')
        raw=json.dumps(record.get('value'),ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
        if hashlib.sha256(raw.encode()).hexdigest()!=record.get('value_sha256'):
            raise ValueError('Context scalar type/value changed after delivery')
        if record['version']==CONTEXT_EVIDENCE_VERSION:
            if 'source_model_call_id' in record:
                raise ValueError('New context evidence must not mix legacy delivery naming with authorship')
            origin=record.get('data_origin')
            if not _valid_scalar_origin(origin):
                raise ValueError('Context evidence origin is incomplete or untyped')
    return deepcopy(rows)


def bind_context_evidence(rows, request, call_id):
    """Resolve only delivered inputs, never runtime state or undeclared history."""
    from input_references import decode_input_packet
    packet = decode_input_packet(request)
    result = []
    seen = set()
    for index, row in enumerate(rows):
        path = row['input_path']
        prefix = 'context_evidence[' + str(index) + '].input_path=' + repr(path) + ': '
        try:parts=_context_pointer_parts(path)
        except ValueError as exc:raise ValueError(prefix+str(exc)) from exc
        root = parts[0]
        if root not in CONTEXT_SOURCES:
            raise ValueError(prefix + 'source root is not a delivered feedback source; see context_evidence_contract')
        if path in seen:
            raise ValueError(prefix + 'duplicate citation path')
        seen.add(path)
        value = packet
        provenance = []
        origin=_empty_scalar_origin()
        selected_history=None
        provenance_keys = ('reference_id', 'report_id', 'observation_id', 'actor_call_id',
            'proposal_call_id', 'planner_call_id', 'model_call_id', 'source_round',
            'target_alias', 'source', 'scope', 'snapshot_role', 'version','delivered_in_call_id','source_model_call_id')
        for part in parts:
            # History envelopes are produced by history_pages and validated by
            # its delivery receipt. Selected fields keep literal pointer keys;
            # only a proposal subtree is authored by that event's planner.
            if isinstance(value,dict) and isinstance(value.get('selected_fields'),dict) and isinstance(value.get('event_id'),str):
                selected_history=(value['selected_fields'],value['event_id'],value.get('selected_field_origins'))
            if selected_history is not None and value is selected_history[0]:
                origin=_empty_scalar_origin()
                if part=='/proposal' or part.startswith('/proposal/'):
                    origin.update(role='planner_model_claim',author_model_call_id=selected_history[1],
                                  planner_call_id=selected_history[1])
                    from discovery_protocol import planner_content_kind
                    kind=planner_content_kind(_context_pointer_parts(part)[1:])
                    if kind is not None:origin['content_kind']=kind
                if selected_history[2] is not None:
                    origins=selected_history[2]
                    if not isinstance(origins,dict) or set(origins)!=set(value):
                        raise ValueError('Selected history origin paths differ from delivered fields')
                    origin=selected_history_origin(origins[part],value[part])
            origin=_scalar_origin(value,part,origin)
            if isinstance(value, dict):
                metadata = {key: deepcopy(value[key]) for key in provenance_keys
                            if key in value and not isinstance(value[key], (dict, list))}
                if metadata: provenance.append(metadata)
            if isinstance(value, dict) and part in value:
                value = value[part]
            elif isinstance(value, list) and part.isascii() and part.isdigit() and str(int(part)) == part and int(part) < len(value):
                value = value[int(part)]
            else:
                raise ValueError(prefix + 'field was not delivered at this path; no data was retrieved')
        if isinstance(value, (dict, list)):
            raise ValueError(prefix + 'cite a specific scalar field, not a container')
        canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
        result.append({'input_path': path, 'cue': row['cue'], 'value': deepcopy(value),
            'value_sha256': hashlib.sha256(canonical.encode('utf-8')).hexdigest(),
            'delivered_in_call_id': call_id, 'data_origin':origin,'source_root': root,
            'source_provenance': provenance,
            'source_scope': CONTEXT_SOURCES[root], 'version': CONTEXT_EVIDENCE_VERSION,
            'semantic_support': 'unverified_model_cue_not_a_controller_conclusion'})
        if root == 'own_inventory' and isinstance(packet.get(root, {}).get('snapshot_provenance'), dict):
            result[-1]['inventory_snapshot_provenance'] = deepcopy(packet[root]['snapshot_provenance'])
    return project_context_evidence(result,delivered_in_call_id=call_id)


class InteractionMemory:
    """Remember actual actor-selected interactions, never infer checks or goals."""
    def __init__(self):
        self.references = {}
        self.bindings = {}

    def remember(self, action, scene, aliases, observation_id, call_id, executed,
                 share_actor_identity=False, execution_feedback=None):
        if not executed or not isinstance(action.get('objectId'), str):
            return None
        alias = action['objectId']
        offered = next((row for row in scene.get('objects', []) if row.get('objectId') == alias), None)
        if offered is None or alias not in aliases:
            return None
        identity = f'interaction-{len(self.references)+1:06d}'
        reference = {'reference_id': identity, 'target_alias': alias,
            'boolean_fields': sorted(key for key, value in offered.items() if type(value) is bool),
            'action': deepcopy(action), 'observation_id': observation_id, 'actor_call_id': call_id,
            'scope': 'Privileged actor interaction contract: field names only, no values, desired predicate, issue or remedy. This attempted native interaction is not proof of benefit or success.'}
        if share_actor_identity and isinstance(offered.get('objectType'), str):
            reference['scope'] = 'Privileged actor interaction contract: actor-observed selected target identity and Boolean field names only, no state values, desired predicate, issue or remedy. This attempted native interaction is not proof of benefit or success.'
            reference['actor_observed_target'] = {'target_alias': alias,
                'object_type': offered['objectType'],
                'source': 'Native-assisted actor input at the recorded call, not RGB recognition or the planner requested target.'}
        if share_actor_identity and execution_feedback is not None:
            result = {}
            for key, kind in (('success', bool), ('category', str), ('environment_command_executed', bool)):
                if key in execution_feedback:
                    value = execution_feedback[key]
                    if value is not None and type(value) is not kind:
                        raise ValueError('Invalid recorded execution feedback field: ' + key)
                    result[key] = deepcopy(value)
            reference['execution_feedback'] = result
            from runtime_support import project_dispatch_feedback,project_execution_provenance
            reference['execution_feedback'].update(project_dispatch_feedback(execution_feedback))
            provenance=project_execution_provenance(execution_feedback)
            if provenance is not None:reference['execution_feedback']['execution_provenance']=provenance
            selection=execution_feedback.get('actor_selection_report')
            if selection is not None:
                selection=project_actor_selection(selection,source=provenance['source'] if provenance else None,
                    before_observation_id=observation_id)
                reference['actor_selection_report_reference']={'version':selection['version'],
                    'actor_call_id':call_id,'binding_status':selection['binding_status'],
                    'model_assessment':(selection.get('model_report') or {}).get('assessment'),
                    'history_field':'/execution/feedback/actor_selection_report',
                    'scope':'Model-authored correspondence claim at that call;accepted references do not verify identity'}
            if 'physical_time' in execution_feedback:
                reference['execution_feedback']['physical_time'] = deepcopy(execution_feedback['physical_time'])
            reference['result_scope'] = 'Known controller feedback for this recorded attempt; missing/null stays unknown. Not current object state, intended-target match, functional benefit or task verification.'
        self.references[identity] = reference
        self.bindings[identity] = aliases[alias]
        return deepcopy(reference)

    def context(self):
        return deepcopy(list(self.references.values()))

    def inventory_identity(self, held_alias, aliases):
        """Only repeat an actor-seen identity while its native alias binding agrees."""
        if not held_alias or held_alias not in aliases:
            return None
        for identity, reference in reversed(list(self.references.items())):
            if (reference['target_alias'] == held_alias
                    and self.bindings[identity] == aliases[held_alias]
                    and 'actor_observed_target' in reference):
                return {'reference_id': identity,
                    **deepcopy(reference['actor_observed_target']),
                    'scope': 'Historical actor-observed identity for the currently held alias; not a new native scan, RGB recognition, intended target match or task success.'}
        return None

    def check_error(self, checks, aliases):
        for check in checks:
            identity = check.get('reference_id')
            reference = self.references.get(identity)
            if reference is None:
                return 'Check reference was not acquired through an actual previous actor interaction'
            if check['target_alias'] != reference['target_alias'] or check['field'] not in reference['boolean_fields']:
                return 'Check alias/field does not match its recorded interaction contract'
            if aliases.get(check['target_alias']) != self.bindings[identity]:
                return 'Check alias no longer binds the recorded interaction identity'
        return None


TARGET_REPORT_VERSION = 'actor-selected-target-report-v1'
AFFORDANCES = ('pickupable', 'receptacle', 'openable', 'toggleable', 'sliceable',
              'cookable', 'dirtyable', 'breakable', 'canFillWithLiquid')


class ActorTargetReports:
    """Keep reports separate from actual interaction/check references."""
    def __init__(self):
        self.records = []

    def remember(self, action, scene, *, actor_call_id, proposal_call_id, observation_id):
        if action.get('action') != 'RequestReplan' or 'target_alias' not in action:
            return None
        alias = action['target_alias']
        target = next((obj for obj in scene.get('objects', []) if obj.get('objectId') == alias), None)
        if target is None or not isinstance(target.get('objectType'), str):
            raise ValueError('Target report requires a typed candidate in this actor input')
        record = {'report_id': f'actor-report-{len(self.records)+1:06d}',
            'version': TARGET_REPORT_VERSION, 'actor_call_id': actor_call_id,
            'proposal_call_id': proposal_call_id, 'observation_id': observation_id,
            'model_report': {'reason': action['reason'], 'target_alias': alias},
            'actor_observed_target': {'target_alias': alias, 'object_type': target['objectType'],
                'affordances_at_actor_input': {key: target[key] for key in AFFORDANCES
                                              if type(target.get(key)) is bool}},
            'environment_command_executed': False,
            'scope': 'Selected identity/affordances copied only from this recorded actor input. '
                'The actor chose the alias and authored the reason. Not an interaction/check '
                'reference, current scene scan, intended-target match, impossibility or benefit.'}
        self.records.append(record)
        return deepcopy(record)

    def context(self):
        return deepcopy(self.records)


def historical_feedback(value):
    """Label projected old offers; raw journals and exact history reads stay original."""
    result = deepcopy(value)
    def visit(item):
        if isinstance(item, list):
            for child in item:
                visit(child)
        elif isinstance(item, dict):
            selection_report=item.get('actor_selection_report')
            if isinstance(selection_report,dict):
                selection_report['snapshot_role']='historical_actor_input_not_current_planner_state'
            selection=item.get('selected_argument')
            if isinstance(selection,dict):
                selection['snapshot_role']='historical_actor_input_not_current_planner_state'
            offer = item.get('actor_tool_scope')
            if isinstance(offer, dict):
                if 'current_feasibility' in offer:
                    offer['feasibility_at_actor_input'] = offer.pop('current_feasibility')
                offer['snapshot_role'] = 'historical_actor_input_not_current_planner_state'
                offer['identity_boundary'] = ('Candidate eligibility does not identify the object. '
                    'Only selected identity reports/interactions name an actor-observed type. '
                    'Use the original observation/call provenance; do not infer an alias from eligibility.')
            for child in item.values():
                visit(child)
    visit(result)
    return result


ACTOR_SELECTION_VERSION='actor-intent-selection-v1'
ACTOR_INTENT_ROOTS=('task','task_goal','plan_steps','task_continuity')
ACTOR_INTENT_TEXT_VERSION='actor-intent-text-reference-v3'


def actor_intent_text_role(path, *, legacy=False):
    """Classify explicit authored text locations, never metadata or step status."""
    parts=_context_pointer_parts(path)
    if parts in (['task'],['task_goal']):return 'current_planner_text'
    if (len(parts)==3 and parts[0]=='plan_steps' and parts[1].isascii()
            and parts[1].isdigit() and str(int(parts[1]))==parts[1]
            and parts[2] in ({'objective','note'} if legacy else {'objective'})):return 'current_planner_text'
    if parts[0]=='task_continuity':
        if (len(parts)==3 and parts[1] in ({'original_intent','changed_current_intent'} if legacy else {'changed_current_intent'})
                and parts[2] in {'goal','task_goal'}):
            return 'original_planner_text' if parts[1]=='original_intent' else 'current_planner_text'
        if (len(parts)==4 and parts[1] in ({'original_plan_steps','current_plan_steps'} if legacy else {'current_plan_steps'})
                and parts[2].isascii() and parts[2].isdigit() and str(int(parts[2]))==parts[2]
                and parts[3] in ({'objective','note'} if legacy else {'objective'})):
            return 'original_planner_text' if parts[1]=='original_plan_steps' else 'current_planner_text'
    raise ValueError('intent_path must select a current planner goal or step objective; historical intent, notes, assessments and metadata are context, not current commands')


def actor_intent_text_reference(request,path,source):
    """Separate delivery actor from original/current planner authorship."""
    from episode_memory import _json_pointer
    role=actor_intent_text_role(path)
    parts=_context_pointer_parts(path)
    if parts[0]=='plan_steps' or parts[:2]==['task_continuity','current_plan_steps']:
        step=_json_pointer(request,path.rsplit('/',1)[0])
        if step.get('status') not in {'pending','in_progress'}:
            raise ValueError('intent_path selects a non-operative plan step; reported progress or cancelled/blocked text is not a current command')
    value=_json_pointer(request,path)
    if not isinstance(value,str) or not value.strip():
        raise ValueError('intent_path must identify nonempty delivered planner text')
    continuity=request.get('task_continuity') or {}
    if role=='original_planner_text':
        author=continuity.get('original_model_call_id')
    else:
        author=source.get('planner_call_id')
        declared=[request.get('planning_source',{}).get('model_call_id'),
                  request.get('task_provenance',{}).get('proposal_call_id'),
                  continuity.get('last_accepted_plan_call_id')]
        if any(call is not None and call!=author for call in declared):
            raise ValueError('Delivered current planner text belongs to a different accepted revision')
    if author is not None and (not isinstance(author,str) or not author):
        raise ValueError('Planner text author must be a recorded call ID or unknown')
    return {'version':ACTOR_INTENT_TEXT_VERSION,'input_path':path,
            'value_sha256':hashlib.sha256(json.dumps(value,ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),
            'source_actor_call_id':source['actor_call_id'],'source_planner_call_id':author,
            'text_role':role,'content_kind':'model_intent','command_authority':'current_planner_request',
            'assessment_source':'model_authored_planner_text_unverified'}


def actor_selection_contract(current_observation_id):
    return {'version':ACTOR_SELECTION_VERSION,'intent_roots':list(ACTOR_INTENT_ROOTS),
        'intent_text_reference_version':ACTOR_INTENT_TEXT_VERSION,
        'step_intent_statuses':['pending','in_progress'],
        'intent_text_paths':['/task','/task_goal','/plan_steps/{index}/objective',
            '/task_continuity/changed_current_intent/{goal|task_goal}',
            '/task_continuity/current_plan_steps/{index}/objective'],
        'current_observation_id':current_observation_id,
        'scope':'selection_report is actor-authored metadata,not native parameters. intent_path selects '
            'delivered planner goal/step text using JSON pointer; IDs, status and controller metadata '
            'are not intent text. Notes, assessments and original intent remain context, not current commands. '
            'Reissue an earlier objective through a current planner revision if it is intended again. ObjectId actions require matches_intent and a '
            'current RGB citation;uncertain/different can be reported with RequestReplan. Reference '
            'validation never certifies intended identity,correct visual interpretation or benefit.'}


def rejected_actor_selection(action,source,current_observation_id,error):
    return {'version':ACTOR_SELECTION_VERSION,'binding_status':'rejected','source':deepcopy(source),
        'snapshot_role':'actor_input_at_recorded_call',
        'actor_observation_id':current_observation_id,'model_report':deepcopy(action.get('selection_report')),
        'intent_reference':None,'bound_evidence':[],'binding_error':str(error),
        'scope':'Actor-authored selection claim retained with controller reference validation only; '
            'not recognition,intent-match verification,native effect or task completion. '
            'Actor cues/observation IDs do not deliver RGB pixels to the planner.'}


def bind_actor_selection(action,request,memory,actor_ids,current_observation_id,source):
    """Bind delivered planner text and actor RGB;never determine the correct target."""
    from action_protocol import OBJECT_ACTIONS,ProtocolError,validate_selection_report
    from input_references import decode_input_packet
    from image_slots import enabled as shared_images
    report=action.get('selection_report')
    if report is None:
        if action['action'] in OBJECT_ACTIONS:raise ProtocolError('Object-directed action requires an actor selection_report')
        return None
    validate_selection_report(report)
    try:
        intent_reference=actor_intent_text_reference(decode_input_packet(request),report['intent_path'],source)
        bound=memory.bind(report['evidence'],actor_ids,
            image_views=request.get('image_views') if shared_images(request) else None,exposure_role='actor')
        if action['action'] in OBJECT_ACTIONS:
            if report['assessment']!='matches_intent':
                raise ValueError('Uncertain/different selection does not authorize an object-directed native attempt')
            if current_observation_id not in {r['observation_id'] for r in bound}:
                raise ValueError('Object-directed selection requires evidence from this actor current observation')
    except (ValueError,KeyError,IndexError) as exc:
        raise ProtocolError('Actor selection reference rejected: '+str(exc)) from exc
    result=rejected_actor_selection(action,source,current_observation_id,'')
    result.update(binding_status='accepted',binding_error=None,bound_evidence=bound,
        intent_reference=intent_reference)
    return result


def project_actor_selection(record,*,source=None,before_observation_id=None):
    """Copy stored declarations without turning accepted references into recognition."""
    if record is None:return None
    from action_protocol import validate_selection_report
    keys={'version','binding_status','source','snapshot_role','actor_observation_id','model_report',
          'intent_reference','bound_evidence','binding_error','scope'}
    if (not isinstance(record,dict) or set(record)!=keys or record.get('version')!=ACTOR_SELECTION_VERSION
            or record.get('binding_status') not in {'accepted','rejected'}):
        raise ValueError('Unsupported actor selection report')
    if (not isinstance(record.get('source'),dict)
            or record.get('snapshot_role') not in {'actor_input_at_recorded_call','historical_actor_input_not_current_planner_state'}
            or not isinstance(record.get('bound_evidence'),list)):
        raise ValueError('Actor selection report lacks its recorded input role')
    if record['model_report'] is not None:validate_selection_report(record['model_report'])
    if source is not None and (record.get('source')!=source
            or record.get('actor_observation_id')!=before_observation_id):
        raise ValueError('Actor selection report belongs to another input/call/intent')
    if record['binding_status']=='accepted':
        validate_selection_report(record.get('model_report'))
        reference=record.get('intent_reference') or {}
        if (reference.get('input_path')!=record['model_report']['intent_path']
                or reference.get('source_actor_call_id')!=(record.get('source') or {}).get('actor_call_id')
                or not isinstance(reference.get('value_sha256'),str) or len(reference['value_sha256'])!=64):
            raise ValueError('Actor selection intent reference is incomplete')
        if 'version' in reference:
            legacy=reference.get('version')=='actor-intent-text-reference-v2'
            if (reference.get('version') not in {ACTOR_INTENT_TEXT_VERSION,'actor-intent-text-reference-v2'}
                    or reference.get('text_role')!=actor_intent_text_role(reference['input_path'],legacy=legacy)
                    or reference.get('assessment_source')!='model_authored_planner_text_unverified'
                    or 'source_planner_call_id' not in reference):
                raise ValueError('Actor selection planner text role is inconsistent')
            if not legacy and (reference.get('content_kind')!='model_intent'
                    or reference.get('command_authority')!='current_planner_request'):
                raise ValueError('Actor selection requires a current model intent, not contextual text')
            author=reference['source_planner_call_id']
            if (author is not None and (not isinstance(author,str) or not author)
                    or reference['text_role']=='current_planner_text' and author!=record['source'].get('planner_call_id')):
                raise ValueError('Actor selection planner text author is inconsistent')
        if record['binding_error'] is not None:raise ValueError('Accepted references cannot retain a rejection')
    elif (record['intent_reference'] is not None or record['bound_evidence']
            or not isinstance(record['binding_error'],str) or not record['binding_error']):
        raise ValueError('Rejected selection must preserve its error without accepted bindings')
    return deepcopy(record)


class ObservationMemory:
    def __init__(self, root=None):
        self.read_only = False
        if root is not None:
            from episode_memory import EpisodeMemory
            self.archive = EpisodeMemory(root)
            self.frames = self.archive.frames
        else:
            self.archive = None
            self.frames = {}
        self.views = {}
        self.discovery_exposure = set()
        self.streak = 0

    @classmethod
    def from_archive(cls, root):
        """Read historical experience offline; no native state or action replay."""
        from pathlib import Path
        from episode_memory import EpisodeMemory, FrameArchive
        memory = cls()
        memory.frames = FrameArchive.open_read_only(Path(root)/'rgb')
        for event in EpisodeMemory.read_events(root):
            payload = event['payload']
            if event['kind'] == 'capture':
                identity=payload['observation_id']
                previous=memory.views.get(identity)
                if previous is not None:
                    # Exposure is journal metadata, not a new capture pose/time.
                    prior={k:v for k,v in previous.items() if k!='presented_to_discovery'}
                    incoming={k:v for k,v in payload.items() if k!='presented_to_discovery'}
                    if prior!=incoming:
                        raise ValueError('Observation ID has conflicting archived capture provenance')
                memory.views[identity] = {**deepcopy(payload),
                    **({'presented_to_discovery':True} if identity in memory.discovery_exposure else {})}
            elif event['kind'] == 'discovery_exposure':
                for identity in payload['observation_ids']:
                    if identity not in memory.views:
                        raise ValueError('Exposure journal precedes its capture')
                    memory.discovery_exposure.add(identity)
                    memory.views[identity]['presented_to_discovery'] = True
        if set(memory.frames) != set(memory.views):
            raise ValueError('Capture index and memory journal disagree; no automatic recovery')
        if any(memory.frames.references[i]['rgb_sha256']!=v.get('rgb_sha256')
               for i,v in memory.views.items()):
            raise ValueError('Capture journal RGB identity differs from archived raster index')
        memory.read_only = True
        return memory

    @classmethod
    def branch_from(cls, source, root, *, expected_files):
        """Import exposed RGB into an independent writer, keeping the reader immutable."""
        from episode_memory import EpisodeMemory
        archive = EpisodeMemory.branch_from(source, root, expected_files=expected_files)
        historical = cls.from_archive(root)
        memory = cls()
        memory.archive = archive
        memory.frames = archive.frames
        memory.views = deepcopy(historical.views)
        memory.discovery_exposure = set(historical.discovery_exposure)
        return memory

    def add(self, frame, view, *, derivation=None):
        if self.read_only:
            raise ValueError('Offline memory cannot capture or resume an episode')
        view = {k: deepcopy(view[k]) for k in CAMERA_KEYS if k in view}
        if derivation is not None:
            view['visual_inspection']=deepcopy(derivation)
        identity = view.get("observation_id")
        if identity in self.frames:
            if not np.array_equal(self.frames[identity], frame):
                raise ValueError("Observation ID reused with different RGB")
            existing=self.views[identity]
            if any(key not in existing or existing[key]!=value for key,value in view.items()):
                raise ValueError('Observation ID reused with conflicting capture provenance')
            if existing.get('visual_inspection')!=derivation:
                raise ValueError('Observation ID reused with conflicting region derivation')
            return identity
        if identity is None:
            # Adapters without capture provenance are explicitly marked unknown.
            identity = f"adapter-{len(self.frames)+1:06d}"
            sequence=len(self.frames)+1
            while identity in self.frames:
                sequence+=1;identity=f"adapter-{sequence:06d}"
        view.update(observation_id=identity, rgb_sha256=hashlib.sha256(frame.tobytes()).hexdigest())
        view.setdefault("captured_utc", datetime.now(timezone.utc).isoformat())
        view.setdefault("monotonic_seconds", time.monotonic())
        view.setdefault("position", None)
        view.setdefault("rotation", None)
        stored_frame=frame.copy();stored_frame.flags.writeable=False
        self.frames[identity] = stored_frame
        self.views[identity] = view
        if self.archive is not None:
            self.archive.append('capture', view)
        return identity

    def current(self, env):
        if self.read_only:
            raise ValueError('Offline memory cannot access a live environment')
        frames = env.get_recent_frames(1)
        if not frames:
            raise ValueError("Current RGB unavailable")
        info = getattr(env, "frame_info", [])
        return self.add(frames[-1], info[-1] if info else {})

    def packet(self, identities, current, *, displayed_size=None):
        result=[{**deepcopy(self.views[identity]), "image_index": i,
                 "current": identity == current and 'visual_inspection' not in self.views[identity]}
                for i, identity in enumerate(identities)]
        if displayed_size is not None:
            for view in result:view['displayed_size_wh']=list(displayed_size)
        return result

    def prepare_regions(self, regions, canvas_factory, model_call_id):
        """Validate every requested source/crop before any archive mutation."""
        from discovery_protocol import validate_region_requests
        validate_region_requests(regions)
        prepared=[];canvases={}
        for region in regions:
            source_id=region['observation_id'];self.recall([source_id])
            if 'visual_inspection' in self.views[source_id]:
                raise ValueError('Select an original exposed capture,not a region of a region')
            if source_id not in canvases:
                canvases[source_id]=canvas_factory(self.frames[source_id])
            canvas,render=canvases[source_id]
            x0,y0,x1,y1=region['bbox_xyxy']
            if (canvas.dtype!=np.uint8 or canvas.ndim!=3 or canvas.shape[2]!=3
                    or x1>canvas.shape[1] or y1>canvas.shape[0]):
                raise ValueError('Region is outside the recorded displayed RGB canvas')
            crop=canvas[y0:y1,x0:x1].copy()
            derivation={'version':'model-selected-visual-inspection-v1',
                'source_observation_id':source_id,'source_native_rgb_sha256':self.views[source_id]['rgb_sha256'],
                'source_render':deepcopy(render),'bbox_xyxy':list(region['bbox_xyxy']),
                'crop_size_wh':[x1-x0,y1-y0],
                'crop_rgb_sha256':hashlib.sha256(crop.tobytes()).hexdigest(),
                'requested_by_model_call_id':model_call_id,
                'scope':'Model-selected crop of previously displayed RGB;pose/time belong to the parent capture. Fitting may magnify pixels,not add detail. Not a current native observation,target identity,reachability or task benefit.'}
            key=repr((source_id,region['bbox_xyxy'],model_call_id,derivation['crop_rgb_sha256']))
            identity='region-'+hashlib.sha256(key.encode()).hexdigest()[:24]
            view={**self.views[source_id],'observation_id':identity}
            prepared.append((crop,view,derivation))
        return prepared

    def commit_regions(self, prepared, *, context_ids=()):
        if self.read_only:raise ValueError('Offline memory cannot create new inspection evidence')
        identities=set(context_ids)
        for _,view,derivation in prepared:
            identities.update((view['observation_id'],derivation['source_observation_id']))
        if len(identities)>8:
            raise ValueError('Requested parent/context/region images exceed the eight-image limit;no inspection committed')
        return [self.add(frame,view,derivation=derivation) for frame,view,derivation in prepared]

    def actor_context_ids(self, identities, current):
        parents=[self.views[i]['visual_inspection']['source_observation_id'] for i in identities
                 if 'visual_inspection' in self.views[i]]
        result=list(dict.fromkeys(parents+list(identities)+[current]))
        if len(result)>8:
            raise ValueError('Selected regions and parent context exceed actor image capacity;reduce cited evidence')
        return result

    def mark_discovery_exposure(self, identities, model_call_id=None):
        if self.read_only:
            raise ValueError('Offline memory cannot add model exposure')
        for identity in identities:
            if identity not in self.frames:
                raise ValueError('Cannot expose an unavailable observation')
            self.discovery_exposure.add(identity)
            self.views[identity]['presented_to_discovery'] = True
        if self.archive is not None:
            self.archive.append('discovery_exposure', {'observation_ids': list(identities),
                'model_call_id': model_call_id, 'scope': 'Actual submitted planning RGB; not proof of perception or understanding'})

    def index(self, *, displayed_size=None):
        """Chronological exposed capture metadata; no semantic ranking or hidden RGB."""
        result=[deepcopy(view) for identity, view in self.views.items()
                if identity in self.discovery_exposure]
        if displayed_size is not None:
            for view in result:view['displayed_size_wh']=list(displayed_size)
        return result

    def recall(self, identities):
        """Resolve only prior exposure, without motion or reconstructed pixels."""
        if not identities or len(identities) > 4 or len(set(identities)) != len(identities):
            raise ValueError('Recall requires one to four distinct observation IDs')
        for identity in identities:
            if identity not in self.discovery_exposure:
                raise ValueError('Recall ID was not previously presented to discovery: ' + identity)
            if hashlib.sha256(self.frames[identity].tobytes()).hexdigest() != self.views[identity]['rgb_sha256']:
                raise ValueError('Recall RGB hash mismatch: ' + identity)
        return list(identities)

    def bind(self, evidence, identities, *, image_views=None,exposure_role='discovery'):
        if exposure_role not in {'discovery','actor'}:raise ValueError('Unknown evidence delivery role')
        result = []
        for position, row in enumerate(evidence):
            identity = row.get("observation_id")
            index = row.get("image_index")
            if image_views is not None and index is not None:
                candidates=[v['observation_id'] for v in image_views if v['image_index']==index]
                if not candidates:raise ValueError('Evidence refers to an unavailable image')
                if identity is None:
                    if len(candidates)!=1:raise ValueError('Shared image requires an explicit observation_id')
                    identity=candidates[0]
                if identity not in candidates:raise ValueError('Evidence ID and shared image index disagree')
                index=None
            if index is not None:
                if index >= len(identities):
                    raise ValueError("Evidence refers to an unavailable image")
                if identity is not None and identity != identities[index]:
                    raise ValueError("Evidence ID and image index disagree")
                identity = identities[index]
            if identity not in self.frames:
                raise ValueError('evidence[' + str(position) + '].observation_id=' + repr(identity)
                    + ': unavailable RGB observation ID. Inventory/interaction/report IDs are not RGB; '
                    'use only a previously presented observation ID for pixels.')
            if exposure_role=='actor' and identity not in identities:
                raise ValueError('Selection evidence was not delivered to this actor call')
            if exposure_role=='discovery' and identity not in self.discovery_exposure:
                raise ValueError('evidence[' + str(position) + '].observation_id=' + repr(identity)
                    + ': RGB was not previously presented to discovery; capture existence does not authorize citation or recall')
            result.append({"observation_id": identity, "cue": row["cue"]})
        return result

    def difference(self, before, after):
        old, new = self.frames[before], self.frames[after]
        delta = float(np.abs(old.astype(float)-new.astype(float)).mean()/255) if old.shape == new.shape else None
        a, b = self.views[before], self.views[after]
        position = None
        if a.get("position") is not None and b.get("position") is not None:
            position = sum((a["position"][k]-b["position"][k])**2 for k in ("x", "y", "z"))**0.5
        yaw = None if a.get("yaw") is None or b.get("yaw") is None else abs((b["yaw"]-a["yaw"]+180)%360-180)
        horizon = None if a.get("horizon") is None or b.get("horizon") is None else abs(b["horizon"]-a["horizon"])
        return {"rgb_mean_absolute_delta_0_1": delta, "translation_m": position,
                "yaw_delta_degrees": yaw, "horizon_delta_degrees": horizon}

    @staticmethod
    def similar(metrics):
        values = tuple(metrics.values())
        return None if None in values else all(v < threshold for v, threshold in zip(values, (0.01, 0.01, 1, 1)))

    def revisited_views(self, before, after):
        matches = []
        compared = unknown = 0
        # Captured but never presented images are not the model's prior experience.
        for identity in self.views:
            if identity in {before, after} or identity not in self.discovery_exposure:
                continue
            metrics = self.difference(identity, after)
            similar = self.similar(metrics)
            if similar is None:
                unknown += 1
            else:
                compared += 1
                if similar:
                    matches.append({"observation_id": identity, **metrics})
        return {"matching_exposed_count": len(matches), "recent_matches": matches[-3:],
                "compared_exposed_count": compared, "uncomparable_exposed_count": unknown,
                "scope": "Earlier discovery-exposed RGB/own-camera similarity excluding the immediate before/after capture. Similarity is not proof of no new semantic evidence; no match does not prove novelty. No action or withdrawal is selected."}

    def feedback(self, before, after):
        metrics = self.difference(before, after)
        delta, position, yaw, horizon = metrics.values()
        low = delta is not None and delta < 0.01 and position is not None and position < 0.01 and yaw is not None and yaw < 1 and horizon is not None and horizon < 1
        self.streak = self.streak+1 if low else 0
        return {"before_observation_id": before, "after_observation_id": after,
                "rgb_mean_absolute_delta_0_1": delta, "translation_m": position,
                "yaw_delta_degrees": yaw, "horizon_delta_degrees": horizon,
                "low_change": low if None not in (delta, position, yaw, horizon) else None,
                "consecutive_low_change": self.streak,
                "revisited_views": self.revisited_views(before, after),
                "scope": "Pixel/pose change only, not semantic evidence or goal progress. Low change: RGB<0.01, translation<0.01m, yaw/horizon<1 degree. Re-evaluate missing information; action and hypothesis remain your choice."}
