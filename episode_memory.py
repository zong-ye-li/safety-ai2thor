"""Append-only episode memory and bounded RAM RGB access; never resumes native IO."""
from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
from evidence_io import atomic_json_write
from runtime_support import native_dispatch_feedback, project_dispatch_feedback, project_execution_provenance, project_postprocessing_failure

HISTORY_DELIVERY_VERSION = 'exact-history-delivery-receipt-v2'
EPISODE_DISPOSITION_VERSION='recorded-episode-disposition-v1'
TASK_FIELD_LINEAGE_VERSION='model-task-field-lineage-v2'


def task_field_lineage(item, summary):
    """Attribute exact retained fields to events, never to the latest delivery.

    Rejected-first content remains a rejected candidate. Withdrawal authors an
    assessment, not a replacement plan. Missing/mismatching legacy text stays
    unknown; no saved event or summary value is repaired here.
    """
    from discovery_protocol import planner_content_kind
    events=item.get('events', [])
    content=next((e for e in reversed(events) if e.get('kind') in {'proposal','revision'}),None)
    if content is None:
        candidate=item.get('original', {})
        if candidate.get('kind')=='contract_rejection':content=candidate
    assessment=next((e for e in reversed(events)
        if e.get('kind') in {'proposal','revision','withdrawal'}),None)
    fields={}
    for field in ('claim','goal','reason','missing_information','observation','checks',
                  'nonblocking_unknowns','task_goal','plan_steps','outcome','outcome_reason','decision'):
        source_field='hypothesis' if field=='claim' else field
        event=assessment if field in {'outcome','outcome_reason','decision'} else content
        proposal=(event or {}).get('proposal',{})
        if field not in summary or source_field not in proposal or summary[field]!=proposal[source_field]:continue
        call=(event or {}).get('model_call_id')
        if not isinstance(call,str) or not call:continue
        raw=json.dumps(summary[field],ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
        fields[field]={'model_call_id':call,'event_kind':event['kind'],
            'proposal_field':source_field,'content_kind':planner_content_kind([source_field]),
            'value_sha256':hashlib.sha256(raw.encode('utf-8')).hexdigest()}
    original=item.get('original',{})
    retained=summary.get('original_plan')
    if (retained and isinstance(original.get('model_call_id'),str)
            and all(k in original.get('proposal',{}) and v==original['proposal'][k] for k,v in retained.items())):
        raw=json.dumps(retained,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
        fields['original_plan']={'model_call_id':original['model_call_id'],'event_kind':original['kind'],
            'proposal_field':'proposal','value_sha256':hashlib.sha256(raw.encode('utf-8')).hexdigest()}
    return {'version':TASK_FIELD_LINEAGE_VERSION,'fields':fields,
        'scope':'Exact model-authored fields only; rejected candidate is not accepted content. '
                'claim stores proposal hypothesis. No authorship is inferred from latest_event or delivery.'}


def latest_task_assessment(item):
    """Latest accepted model assessment, separate from retained plan content.

    Rejections and actual execution feedback cannot author an outcome. Legacy
    events without outcome text stay unknown; this never repairs saved history.
    """
    for event in reversed(item.get('events', [])):
        if event.get('kind') not in {'proposal', 'revision', 'withdrawal'}:
            continue
        proposal = event.get('proposal') or {}
        if 'outcome' not in proposal:
            return None
        return {'model_call_id': event['model_call_id'],
                'source_round': event.get('source_round'),
                'decision': proposal.get('decision'), 'outcome': proposal['outcome'],
                'outcome_reason': proposal.get('outcome_reason'),
                'assessment_source': 'model_only_unverified'}
    return None


def task_assessment_state(item):
    """Project accepted assessment state; a rejected first draft has none.

    Legacy accepted plans without assessment fields retain their recorded state.
    Rejected text remains available as a candidate and in exact event history.
    """
    assessment = latest_task_assessment(item)
    if assessment is not None:
        return ('withdrawn' if assessment['decision'] == 'withdraw' else
                {'ongoing': 'open', 'resolved': 'reported_resolved',
                 'difficult': 'reported_difficult', 'no_issue': 'reported_no_issue'}[assessment['outcome']])
    if any(e.get('kind') in {'proposal', 'revision', 'withdrawal'} for e in item.get('events', [])):
        return item.get('task_state', 'unverified')
    return 'unverified'


def episode_disposition(report,stop_reason=None,error=None):
    """Output-only index of saved evidence;never query an environment or certify a loop."""
    recorded=None
    for row in reversed(report.get('rounds',[])):
        for calls_key,input_key in (('revision_proposals','revision_input'),('proposals','input')):
            calls=row.get(calls_key) or []
            call=next((c for c in reversed(calls) if c.get('model_call_id')),None)
            packet=row.get(input_key)
            if call is not None and isinstance(packet,dict):
                recorded=(call,packet);break
        if recorded is not None:break
    call,packet=recorded if recorded is not None else ({},{})
    inventory=packet.get('own_inventory')
    inventory=inventory if isinstance(inventory,dict) else {}
    call_id=call.get('model_call_id')
    operations=report.get('operation_history_records') or []
    last=operations[-1] if operations else None
    tasks=[]
    for identity,item in report.get('hypotheses',{}).items():
        events=item.get('events') or []
        accepted=next((e for e in reversed(events) if e.get('kind') in {'proposal','revision'}),{})
        assessment=latest_task_assessment(item)
        tasks.append(dict(hypothesis_id=identity,model_task_state=task_assessment_state(item),
            model_outcome=assessment['outcome'] if assessment is not None else None,
            latest_model_assessment=assessment,contract_valid=item.get('contract_valid'),
            content_source_call_id=accepted.get('model_call_id'),
            last_recorded_event_id=events[-1].get('model_call_id') if events else None,
            model_step_states=[{k:deepcopy(s.get(k)) for k in ('id','status')} for s in item.get('plan_steps',[])],
            assessment_source='model_only_unverified'))
    result=dict(version=EPISODE_DISPOSITION_VERSION,stop_reason=stop_reason,
        stop_origin='exception_interrupted' if error is not None else
            'model_requested_end' if stop_reason=='discovery_model_requested_end' else
            'controller_read_only_assessment' if stop_reason=='discovery_assessed_at_limit' else
            'controller_stop_without_model_end',
        model_report_path='/model_episode_end' if isinstance(report.get('model_episode_end'),dict) else
            '/final_model_assessment' if isinstance(report.get('final_model_assessment'),dict) else None,
        task_dispositions=tasks,
        recorded_inventory=dict(recorded_in_call_id=call_id,field_present='heldObject' in inventory,
            held_alias=deepcopy(inventory.get('heldObject')),
            snapshot_role='last_recorded_planner_input_not_terminal_native_inventory',
            later_operation_ids=[r['operation_id'] for r in operations
                if call_id is not None and r['source']['planner_call_id']==call_id]),
        latest_operation=None,pending_operation_ids=[r['operation_id'] for r in operations if r['phase']=='pending'],
        completion_certification='not_established',benefit_certification='not_established',
        extra_effects_certification='requires_separate_audit',room_safety='unknown',
        scope='Derived only from existing controller records and model-authored claims. Reported resolved/done is unverified. '
            'Inventory belongs to a recorded call payload;it proves neither model receipt/perception nor final native state. '
            'A later operation may have changed inventory. Pending is not a fabricated native acknowledgment. '
            'This output is not model input,an additional observation,an autonomous success score or permission to replay effects.')
    if last is not None:
        feedback=last.get('feedback') or {}
        result['latest_operation']=dict(operation_id=last['operation_id'],role=last['role'],phase=last['phase'],
            source=deepcopy(last['source']),
            **{k:deepcopy(feedback.get(k)) for k in ('native_dispatch_status','native_action_command_id',
                'native_outcome_received','success','post_execution_processing_error')})
    if error is not None:result['interruption']=deepcopy(error)
    return result


def validate_history_delivery(page):
    """Check actual payload/receipt agreement,not the model's use of delivered text."""
    if not isinstance(page, dict) or not isinstance(page.get('events'), list) or not isinstance(page.get('fields'), list):
        raise ValueError('History page requires actual event/field lists')
    receipt = page.get('delivery_receipt')
    if not isinstance(receipt, dict) or receipt.get('version') != HISTORY_DELIVERY_VERSION:
        raise ValueError('Current history delivery requires a versioned receipt')
    delivered = [row['event_id'] for row in page['events']]
    requested = receipt.get('requested_event_ids')
    if (not isinstance(requested, list) or any(not isinstance(identity, str) for identity in requested)
            or len(set(requested)) != len(requested) or delivered != requested[:len(delivered)]):
        raise ValueError('History receipt requested/delivered event identities disagree')
    whole = not bool(page['fields'])
    expected = {'delivered_event_ids': delivered, 'requested_fields': page['fields'],
        'delivery_mode': 'whole_events' if whole else 'selected_fields',
        'requested_event_count': len(requested), 'delivered_event_count': len(delivered),
        'content_delivered': bool(delivered), 'whole_events_delivered': bool(delivered) and whole,
        'all_requested_events_delivered': len(delivered) == len(requested),
        'not_delivered_requested_event_ids': requested[len(delivered):]}
    if any(type(receipt.get(key)) is not type(value) or receipt[key] != value for key, value in expected.items()):
        raise ValueError('History receipt mode/count/coverage disagrees with delivered content')
    for row in page['events']:
        keys={'event_id', 'event' if whole else 'selected_fields'}
        if not whole and 'selected_field_origins' in row:keys.add('selected_field_origins')
        if set(row) != keys:
            raise ValueError('History payload does not match its declared delivery mode')
        if whole and (not isinstance(row['event'], dict) or row['event'].get('model_call_id') != row['event_id']):
            raise ValueError('History event source identity disagrees with its envelope')
        if not whole and (not isinstance(row['selected_fields'], dict) or set(row['selected_fields']) != set(page['fields'])):
            raise ValueError('History selected field paths disagree with its request')
        if 'selected_field_origins' in row:
            from observation_memory import selected_history_origin
            origins=row['selected_field_origins']
            if not isinstance(origins,dict) or set(origins)!=set(row['selected_fields']):
                raise ValueError('History origin paths disagree with delivered fields')
            for path,value in row['selected_fields'].items():selected_history_origin(origins[path],value)
    expected_status = (('partial' if delivered else 'capacity_exceeded') if len(delivered) < len(requested)
                       else ('page' if page['has_more'] else 'complete'))
    if page['status'] != expected_status:
        raise ValueError('History status disagrees with its delivered coverage')
    return receipt


FEEDBACK_CHAIN_VERSION='execution-feedback-causal-index-v2'


def execution_feedback_chain(rounds, operation_records, limit=None):
    """Small causal index over existing records, not another event journal."""
    if limit is not None and (type(limit) is not int or limit<1):
        raise ValueError('Explicit feedback-chain limit must be a positive integer')
    entries = []
    for row in (rounds if limit is None else rounds[-limit:]):
        calls = [item['model_call_id'] for key in ('proposals', 'revision_proposals')
                 for item in row.get(key, []) if 'model_call_id' in item]
        operations = [record for record in operation_records
                      if record['source']['planner_call_id'] in calls]
        status = row.get('status')
        proposal = row.get('proposal') or {}
        rejected = status in {'proposal_guard_rejected', 'revision_guard_rejected'}
        entries.append({'round': row['round'], 'planner_call_ids': calls,
            'proposal_dispositions':[{'model_call_id':item['model_call_id'],'stage':stage,
                'format_valid':item.get('format_valid'),'contract_valid':item.get('contract_valid'),
                'actor_invocation_started':item.get('actor_invocation_started'),
                'operation_ids':[r['operation_id'] for r in operations if r['source']['planner_call_id']==item['model_call_id']]}
                for key,stage in (('proposals','proposal'),('revision_proposals','revision'))
                for item in row.get(key,[]) if 'model_call_id' in item],
            'hypothesis_id': proposal.get('hypothesis_id'), 'decision': proposal.get('decision'),
            'controller_status': status, 'proposal_rejected_before_execution': rejected,
            'operation_links': [{'operation_id': record['operation_id'],
                'role': record['role'], 'actor_call_id': record['source']['actor_call_id'],
                'phase': record['phase'],
                **({'post_execution_processing_error':project_postprocessing_failure(record['feedback']['post_execution_processing_error'])}
                   if (record.get('feedback') or {}).get('post_execution_processing_error') is not None else {}),
                'feedback_stage':((record.get('feedback') or {}).get('execution_provenance') or {}).get('stage'),
                'selected_argument_alias':((((record.get('feedback') or {}).get('execution_provenance') or {}).get('selected_argument') or {}).get('target_alias')),
                'argument_role':((((record.get('feedback') or {}).get('execution_provenance') or {}).get('selected_argument') or {}).get('argument_role')),
                'selection_reference_status':((record.get('feedback') or {}).get('actor_selection_report') or {}).get('binding_status'),
                'model_target_assessment':(((record.get('feedback') or {}).get('actor_selection_report') or {}).get('model_report') or {}).get('assessment'),
                'decision_review':deepcopy((record.get('feedback') or {}).get('decision_review')),
                'environment_command_executed': (record.get('feedback') or {}).get('environment_command_executed'),
                'native_outcome_received': (record.get('feedback') or {}).get('native_outcome_received'),
                'native_dispatch_status': (record.get('feedback') or {}).get('native_dispatch_status'),
                'success': (record.get('feedback') or {}).get('success'),
                'before_observation_id': (record.get('before_observation') or {}).get('observation_id'),
                'after_observation_id': (record.get('after_observation') or {}).get('observation_id')}
                for record in operations],
            'actor_report_id': (row.get('actor_target_report') or {}).get('report_id'),
            'interaction_reference_id': (row.get('interaction_reference') or {}).get('reference_id')})
    return {'version': FEEDBACK_CHAIN_VERSION, 'entries': entries,
        'omitted_round_count': max(0, len(rounds)-len(entries)),
        'scope': 'Chronological index of prior planner rounds and their actual operation records; '
            'links are not extra RGB delivery or native state. Rejected proposals did not reach actor/native '
            'execution. Per-call format/contract acceptance and actor invocation are separate; null means '
            'legacy data did not record that stage. Selected argument alias/role belongs to the linked '
            'actor input, not the planner intended target or current state. model_target_assessment is an '
            'actor-authored claim, not recognition or benefit;selection_reference_status validates sources only. Controller refresh is distinct '
            'from actor action. False/unknown execution is not '
            'success; success is not intended-target match, benefit or task completion. Read full task '
            'events through the existing history page interface; no outcome or plan is inferred here.'}


class OperationHistory:
    """Literal controller-handled attempts;never selects a task or next action."""
    VERSION='model-operation-history-v1'
    ROLES={'controller_refresh','model_actor','model_control'}

    def __init__(self,archive=None,initial_native_audit_entries=0):
        if type(initial_native_audit_entries) is not int or initial_native_audit_entries<0:
            raise ValueError('Initial native audit counter must be nonnegative')
        self.records=[]
        self.archive=archive
        self.initial_native_audit_entries=initial_native_audit_entries
        self.read_only=False
        self.branches=[]
        if archive is not None:
            archive.append('operation_history_start',{'version':self.VERSION,
                'initial_native_audit_entries':initial_native_audit_entries})

    def _writable(self):
        if self.read_only:raise ValueError('Offline operation history cannot continue or replay effects')

    @classmethod
    def branch_from(cls, events, archive, initial_native_audit_entries=0):
        historical=cls.from_events(events)
        if any(r['phase'] in {'pending','effect_unknown'} for r in historical.records):
            raise ValueError('Cannot branch unresolved operation effects')
        history=cls(initial_native_audit_entries=initial_native_audit_entries)
        history.records=deepcopy(historical.records)
        history.archive=archive
        history.branches=deepcopy(historical.branches)
        boundary={'version':'operation-history-branch-v1',
                  'inherited_records':len(history.records),
                  'current_initial_native_audit_entries':initial_native_audit_entries,
                  'scope':'Earlier record audit counters belong to their source episode. '
                          'New records use the new episode counter; no effects are replayed.'}
        history.branches.append(boundary)
        archive.append('operation_history_branch',boundary)
        return history

    @staticmethod
    def validate_provenance(record):
        feedback=record.get('feedback') or {}
        project_postprocessing_failure(feedback.get('post_execution_processing_error'))
        if feedback.get('actor_selection_report') is not None:
            from observation_memory import project_actor_selection
            selection_report=project_actor_selection(feedback['actor_selection_report'],source=record['source'],
                before_observation_id=(record.get('before_observation') or {}).get('observation_id'))
            if (selection_report['binding_status']=='accepted' and 'objectId' in record['action']
                    and (selection_report['model_report']['assessment']!='matches_intent'
                         or selection_report['actor_observation_id'] not in
                         {r['observation_id'] for r in selection_report['bound_evidence']})):
                raise ValueError('Stored object selection lacks its declared current-view correspondence')
        provenance=project_execution_provenance(feedback)
        if provenance is None:return
        before=(record.get('before_observation') or {}).get('observation_id')
        after=(record.get('after_observation') or {}).get('observation_id')
        selected=provenance.get('selected_argument')
        if (provenance['source']!=record['source'] or provenance['before_observation_id']!=before
                or provenance['after_observation_id'] is not None and provenance['after_observation_id']!=after
                or selected is not None and selected['target_alias']!=record['action'].get('objectId',record['action'].get('target_alias'))):
            raise ValueError('Stored execution provenance disagrees with operation source/argument/observations')

    @classmethod
    def from_events(cls,events):
        history=cls();started=False
        for event in events:
            kind=event['kind'];value=event['payload']
            if kind=='operation_history_start':
                if started or history.records or value.get('version')!=cls.VERSION:
                    raise ValueError('Operation history origin changed')
                history.initial_native_audit_entries=value['initial_native_audit_entries'];started=True
            elif kind=='operation_history_branch':
                if (not started or value.get('version')!='operation-history-branch-v1'
                        or value.get('inherited_records')!=len(history.records)
                        or type(value.get('current_initial_native_audit_entries')) is not int
                        or value['current_initial_native_audit_entries']<0
                        or any(r['phase'] in {'pending','effect_unknown'} for r in history.records)):
                    raise ValueError('Invalid operation-history branch boundary')
                history.branches.append(deepcopy(value))
                history.initial_native_audit_entries=value['current_initial_native_audit_entries']
            elif kind=='operation_intent':
                if not started or value['sequence']!=len(history.records)+1 or value['phase']!='pending':
                    raise ValueError('Operation intent sequence is incomplete')
                if history.branches and value.get('audit_epoch')!=len(history.branches):
                    raise ValueError('Operation uses a different branch audit counter')
                history.records.append(deepcopy(value))
            elif kind=='operation_postprocessing_failure':
                original=history._record(value['operation_id'])
                if (original['phase']!='feedback_received' or value['source']!=original['source']
                        or value['prior_feedback_sha256']!=cls.feedback_digest(original['feedback'])
                        or 'post_execution_processing_error' in original['feedback']):
                    raise ValueError('Postprocessing annotation changed or repeated a native receipt')
                original['feedback']['post_execution_processing_error']=project_postprocessing_failure(value['failure'])
                cls.validate_provenance(original)
            elif kind in {'operation_feedback','operation_observation'}:
                original=history._record(value['operation_id'])
                keys=('operation_id','sequence','role','source','action','before_observation','native_audit_entries_before')
                if any(value[k]!=original[k] for k in keys):raise ValueError('Operation source identity changed')
                if value.get('audit_epoch')!=original.get('audit_epoch'):
                    raise ValueError('Operation audit epoch changed')
                if kind=='operation_feedback':
                    if original['phase']!='pending' or value['phase'] not in {'feedback_received','effect_unknown'}:
                        raise ValueError('Operation feedback cannot be replayed or reordered')
                elif (original['phase']=='pending' or any(value[k]!=v for k,v in original.items() if k!='after_observation')):
                    raise ValueError('Operation observation changed recorded feedback')
                cls.validate_provenance(value)
                history.records[value['sequence']-1]=deepcopy(value)
        if not started:raise ValueError('No operation-history contract was recorded')
        history.read_only=True
        return history

    @staticmethod
    def view_reference(view):
        result={key:deepcopy(view.get(key)) for key in
                ('observation_id','rgb_sha256','captured_utc','yaw','horizon')}
        for key in ('position','rotation'):
            value=view.get(key)
            result[key]=({axis:deepcopy(value.get(axis)) for axis in ('x','y','z')}
                         if isinstance(value,dict) else None)
        return result

    def _publish(self,kind,record):
        if self.archive is not None:self.archive.append(kind,deepcopy(record))

    def begin(self,action,role,source,before_view,native_audit_entries):
        self._writable()
        if role not in self.ROLES:raise ValueError('Unknown operation provenance role')
        if type(native_audit_entries) is not int or native_audit_entries<0:
            raise ValueError('Native audit counter must be a nonnegative integer')
        origin={key:deepcopy(source.get(key)) for key in
                ('hypothesis_id','planner_call_id','actor_call_id','intent_sha256')}
        if (not origin['planner_call_id'] or (role=='controller_refresh' and origin['actor_call_id'] is not None)
                or (role!='controller_refresh' and not origin['actor_call_id'])):
            raise ValueError('Operation source must distinguish planner and actual actor calls')
        record={'operation_id':f'operation-{len(self.records)+1:06d}',
                'sequence':len(self.records)+1,'role':role,'phase':'pending',
                'source':origin,'action':deepcopy(action),
                'before_observation':self.view_reference(before_view),'after_observation':None,
                'native_audit_entries_before':native_audit_entries,
                'native_audit_entries_after':None,'native_audit_entries_added':None,
                'feedback':None,'native_clock_command_ids':None}
        if self.branches:record['audit_epoch']=len(self.branches)
        self.records.append(record);self._publish('operation_intent',record)
        return record['operation_id']

    def _record(self,identity):
        for record in self.records:
            if record['operation_id']==identity:return record
        raise ValueError('Unknown recorded operation')

    def finish(self,identity,feedback,native_audit_entries,after_view=None,*,unknown=False):
        self._writable()
        record=self._record(identity)
        if record['phase']!='pending':raise ValueError('Operation already has a recorded result;never replay it')
        if type(native_audit_entries) is not int or native_audit_entries<record['native_audit_entries_before']:
            raise ValueError('Native audit counter moved backwards')
        public={}
        for key,kind in (('success',bool),('category',str),('environment_command_executed',bool)):
            value=feedback.get(key)
            if value is not None and type(value) is not kind:
                raise ValueError('Invalid reported execution field: '+key)
            public[key]=deepcopy(value)
        message=feedback.get('message',feedback.get('error'))
        if message is not None and not isinstance(message,str):raise ValueError('Execution message must be text or unknown')
        public['message']=message
        failure=project_postprocessing_failure(feedback.get('post_execution_processing_error'))
        if failure is not None:public['post_execution_processing_error']=failure
        if 'post_execution_observation_error' in feedback:
            if not isinstance(feedback['post_execution_observation_error'],str):
                raise ValueError('Observation error must remain text')
            public['post_execution_observation_error']=feedback['post_execution_observation_error']
        public.update(project_dispatch_feedback(feedback))
        if 'native_dispatch_error' in feedback:
            if not isinstance(feedback['native_dispatch_error'], dict):
                raise ValueError('Native dispatch error must remain a structured IO-owner report')
            public['native_dispatch_error'] = deepcopy(feedback['native_dispatch_error'])
        if unknown:
            if public.get('native_dispatch_status') not in (None, 'outcome_unknown'):
                raise ValueError('Unknown operation cannot overwrite an explicit native receipt')
            public.update(success=None,category=None,environment_command_executed=None)
            public.update(native_dispatch_feedback('outcome_unknown', public.get('native_action_command_id')))
        provenance=project_execution_provenance({**public,'execution_provenance':feedback.get('execution_provenance')})
        if provenance is not None:
            if (provenance['source']!=record['source']
                    or provenance['before_observation_id']!=(record.get('before_observation') or {}).get('observation_id')):
                raise ValueError('Operation feedback belongs to a different actor/intent/observation')
            public['execution_provenance']=provenance
        if feedback.get('actor_selection_report') is not None:
            from observation_memory import project_actor_selection
            public['actor_selection_report']=project_actor_selection(feedback['actor_selection_report'],
                source=record['source'],before_observation_id=(record.get('before_observation') or {}).get('observation_id'))
        if feedback.get('decision_review') is not None:
            from decision_review import project_selection_review
            public['decision_review']=project_selection_review(feedback['decision_review'])
        report_id = feedback.get('actor_target_report_id')
        if report_id is not None and not unknown:
            if (record['role'] != 'model_control' or record['action'].get('action') != 'RequestReplan'
                    or not isinstance(report_id, str) or not report_id.startswith('actor-report-')
                    or public['environment_command_executed'] is not False):
                raise ValueError('Actor report reference requires a known no-command planning report')
            public['actor_target_report_id'] = report_id
        timing=feedback.get('physical_time')
        command_ids=({key:deepcopy(timing.get(key)) for key in
                      ('native_action_command_id','frozen_endpoint_command_id')} if isinstance(timing,dict) else None)
        updated={**record,'phase':'effect_unknown' if unknown else 'feedback_received','feedback':public,
                 'native_audit_entries_after':native_audit_entries,
                 'native_audit_entries_added':native_audit_entries-record['native_audit_entries_before'],
                 'native_clock_command_ids':command_ids}
        if after_view is not None:updated['after_observation']=self.view_reference(after_view)
        self.validate_provenance(updated)
        record.clear();record.update(updated)
        self._publish('operation_feedback',record)

    @staticmethod
    def feedback_digest(feedback):
        return hashlib.sha256(json.dumps(feedback,sort_keys=True,ensure_ascii=False,
            separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()

    def retain_processing_failure(self,identity,feedback,native_audit_entries,after_view=None):
        """Keep the received effect immutable;annotate only its later processing failure."""
        self._writable();record=self._record(identity)
        failure=project_postprocessing_failure(feedback.get('post_execution_processing_error'))
        if failure is None:raise ValueError('Processing failure annotation is required')
        if record['phase']=='pending':
            self.finish(identity,feedback,native_audit_entries,after_view);return
        if record['phase']!='feedback_received':
            raise ValueError('Postprocessing cannot change an unknown native effect')
        prior=record['feedback']
        immutable=('success','category','message','execution_provenance','actor_selection_report','decision_review')
        if ('post_execution_processing_error' in prior
                or project_dispatch_feedback(prior)!=project_dispatch_feedback(feedback)
                or self.feedback_digest({k:prior.get(k) for k in immutable})
                   !=self.feedback_digest({k:feedback.get(k) for k in immutable})
                or type(native_audit_entries) is not int
                or native_audit_entries!=record['native_audit_entries_after']
                or after_view is not None and self.view_reference(after_view)!=record.get('after_observation')):
            raise ValueError('Postprocessing cannot change or repeat received feedback')
        delta=dict(operation_id=identity,source=deepcopy(record['source']),
            prior_feedback_sha256=self.feedback_digest(prior),failure=failure)
        self._publish('operation_postprocessing_failure',delta)
        prior['post_execution_processing_error']=failure

    def attach_after(self,identity,view):
        self._writable()
        record=self._record(identity)
        record['after_observation']=self.view_reference(view)
        self._publish('operation_observation',record)

    def since(self,count):
        return deepcopy(self.records[count:])

    def reference(self,count):
        records=self.records[count:]
        return {'version':self.VERSION,'operation_ids':[r['operation_id'] for r in records],
                'history_field':'/execution/operation_history_records',
                'scope':'Exact recorded round operations;not a human task,certified effect or useful outcome'}

    def context(self,limit=None,byte_budget=None):
        # Default runtime delivery retains the complete bounded-episode record.
        # Explicit limited views remain available to historical/offline readers.
        if ((limit is not None and (type(limit) is not int or limit<1))
                or (byte_budget is not None and (type(byte_budget) is not int or byte_budget<1))):
            raise ValueError('Operation history capacity must be positive integers')
        selected=[];blocked=None
        for record in reversed(self.records if limit is None else self.records[-limit:]):
            candidate=[record]+selected
            size=len(json.dumps(candidate,ensure_ascii=False,separators=(',',':')).encode('utf-8'))
            if byte_budget is not None and size>byte_budget:
                blocked={'operation_id':record['operation_id'],
                         'record_bytes':len(json.dumps(record,ensure_ascii=False,separators=(',',':')).encode('utf-8'))}
                break
            selected=candidate
        return {**({'branch_boundaries':deepcopy(self.branches)} if self.branches else {}),
                'version':self.VERSION,'initial_native_audit_entries':self.initial_native_audit_entries,
                'total_records':len(self.records),'record_limit':limit,'record_byte_budget':byte_budget,
                'records':deepcopy(selected),'delivered_record_ids':[r['operation_id'] for r in selected],
                'omitted_record_count':len(self.records)-len(selected),'has_older':len(selected)<len(self.records),
                'capacity_blocked_next_record':blocked,
                'scope':'Chronological controller-handled operations and received feedback only. Refresh is controller-owned;control reports do not execute native commands. Audit entries are the existing action-budget unit,not low-level clock steps. Missing/null stays unknown. No loop,route,target,benefit or task completion is inferred;full round records remain in exact task history.'}


def _json_pointer(value, pointer):
    """Select an explicitly requested exact subtree; never infer a field or summary."""
    if not isinstance(pointer, str) or not pointer.startswith('/'):
        raise ValueError('History field selection requires a JSON pointer starting with /')
    for token in pointer[1:].split('/'):
        token = token.replace('~1', '/').replace('~0', '~')
        if isinstance(value, dict) and token in value:
            value = value[token]
        elif isinstance(value, list) and token.isdigit() and int(token) < len(value):
            value = value[int(token)]
        else:
            raise ValueError('Requested history field is unavailable: ' + pointer)
    return deepcopy(value)


def actor_offer_history_reference(offer, task):
    """Keep mode/authorization in feedback; exact offer belongs to the task event."""
    events = task['events']
    event = events[-1]
    provenance = offer['provenance']
    if event['model_call_id'] != provenance['proposal_call_id']:
        raise ValueError('Historical offer must belong to the current task event')
    raw = json.dumps(offer, ensure_ascii=False, sort_keys=True,
                     separators=(',', ':'), allow_nan=False).encode('utf-8')
    pointer = '/execution/actor_input_offer'
    fields = [pointer+'/'+key.replace('~','~0').replace('/','~1') for key in offer]
    for key, value in offer.get('current_feasibility', {}).items():
        fields.append(pointer+'/current_feasibility/'+key.replace('~','~0').replace('/','~1'))
    return {key: deepcopy(offer[key]) for key in (
        'planner_decision', 'input_mode', 'restriction', 'restriction_source',
        'available_action_names', 'provenance') if key in offer} | {
        'version': 'historical-actor-offer-reference-v1',
        'snapshot_role': 'historical_actor_input_not_current_planner_state',
        'full_offer_delivered': False,
        'original_offer_sha256': hashlib.sha256(raw).hexdigest(),
        'original_offer_bytes': len(raw),
        'history_page_request': {'hypothesis_id': task['id'],
            'after_event_id': events[-2]['model_call_id'] if len(events)>1 else '',
            'limit': 1, 'fields': [pointer]},
        'available_offer_fields': fields,
        'scope': 'Past mode and authorized action names only. Full numeric/interface/feasibility '
            'contents are stored, not delivered here. Planner can read exact fields via '
            'history_page_requests; actor can RequestReplan for needed history. '
            'This is not current feasibility, object identity or task success.'}


def history_field_index(event):
    """Exact field locations/sizes for a model's own event,not selected contents."""
    fields={}
    def add(path,value):
        fields[path]=len(json.dumps(value,ensure_ascii=False,separators=(',',':')).encode('utf-8'))
    for key,value in event.items():
        path='/'+key.replace('~','~0').replace('/','~1')
        add(path,value)
        if isinstance(value,dict):
            for child,item in value.items():
                add(path+'/'+child.replace('~','~0').replace('/','~1'),item)
    return {'event_id':event['model_call_id'],'field_bytes':fields,
            'scope':'Locations and exact serialized value sizes only;contents require your explicit fields request'}


def task_continuity_context(ledger,active_id=None):
    """Literal authored intent/source links;never merge tasks or assess completion."""
    catalog=[]
    active=None
    for identity,item in ledger.items():
        origin=item['original'];proposal=origin['proposal']
        original_intent={key:deepcopy(proposal[key]) for key in ('task_goal','goal') if key in proposal}
        current_intent={key:deepcopy(item.get(key)) for key in original_intent
                        if item.get(key)!=original_intent[key]}
        entry={'hypothesis_id':identity,'original_model_call_id':origin['model_call_id'],
               'original_intent':original_intent,'changed_current_intent':current_intent,
               'current_task_state':task_assessment_state(item),
               'current_projection_source_round':item.get('content_source_round'),
               'event_count':len(item['events']),
               'latest_event_id':item['events'][-1]['model_call_id'],
               'latest_event_kind':item['events'][-1]['kind']}
        entry['plan_status_authority']='model_authored_unverified_claims'
        assessment=latest_task_assessment(item)
        entry['assessment_source_call_id']=assessment['model_call_id'] if assessment is not None else None
        entry['model_outcome']=assessment['outcome'] if assessment is not None else None
        entry['last_accepted_plan_call_id']=next((e['model_call_id'] for e in reversed(item['events'])
                                                if e['kind'] in {'proposal','revision'}),None)
        entry['last_execution_model_call_id']=(item.get('last_execution') or {}).get('model_call_id')
        entry['current_plan_contract_valid']=item.get('contract_valid')
        entry['pending_rejected_draft']=deepcopy(item.get('pending_contract_draft'))
        catalog.append(entry)
        if identity==active_id:
            active={'version':'model-task-continuity-v1','assessment_source':'model_only_unverified',**deepcopy(entry),
                    'original_plan_steps':deepcopy(proposal.get('plan_steps',[])),
                    'current_plan_steps':deepcopy(item.get('plan_steps',[]))}
    return {'version':'model-task-continuity-v1','assessment_source':'model_only_unverified',
            'catalog':catalog,'active':active,
            'scope':'Exact original intent and current model revisions. Same task/step ID does not prove unchanged purpose or completion. You may revise,withdraw or switch;no task is selected or merged for you. Historical text is not current scene evidence.'}


def task_continuity_reference(active):
    """Link exact actor input without duplicating entire plans into every event."""
    if active is None:return None
    def fingerprint(value):
        return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,
                                        separators=(',',':')).encode('utf-8')).hexdigest()
    reference = {key:deepcopy(active[key]) for key in (
        'hypothesis_id','original_model_call_id','latest_event_id','current_projection_source_round')} | {
        'last_accepted_plan_call_id':active.get('last_accepted_plan_call_id'),
        'last_execution_model_call_id':active.get('last_execution_model_call_id'),
        'scope':'Exact actor-input lineage;recover full authored text from recorded source calls/history. Not evidence of intent match,execution or benefit'}
    for key,target in (('original_intent','original_intent_sha256'),
                       ('changed_current_intent','changed_current_intent_sha256'),
                       ('original_plan_steps','original_plan_sha256'),('current_plan_steps','current_plan_sha256')):
        if key in active:reference[target]=fingerprint(active[key])
    reference['delivered_continuity_fields']=list(active)
    return reference


def history_pages(ledger, first_ids, requests, byte_budget=16384, *, include_field_index=False):
    """Paginate exact authored events with visible cursors and byte-capacity feedback."""
    if type(byte_budget) is not int or byte_budget < 1:
        raise ValueError('History event-byte budget must be positive')
    queries = [{'hypothesis_id': identity, 'after_event_id': '', 'limit': 2, 'fields': []}
               for identity in first_ids] + deepcopy(requests)
    if len(queries) > 3 or len({q['hypothesis_id'] for q in queries}) != len(queries):
        raise ValueError('Request up to three distinct task history pages')
    # Resolve identities, cursors and field paths before exposing any partial response.
    resolved = []
    for query in queries:
        identity = query['hypothesis_id']
        if identity not in ledger:
            raise ValueError('Plan recall refers to an unknown hypothesis')
        events = ledger[identity]['events']
        ids = [event['model_call_id'] for event in events]
        cursor = query['after_event_id']
        if cursor and cursor not in ids:
            raise ValueError('History cursor does not belong to the requested hypothesis')
        start = ids.index(cursor)+1 if cursor else 0
        selected = []
        for event in events[start:start+query['limit']]:
            payload = ({'selected_fields': {field: _json_pointer(event, field) for field in query['fields']}}
                       if query['fields'] else {'event': deepcopy(event)})
            if query['fields']:
                from observation_memory import history_field_origin
                payload['selected_field_origins']={field:history_field_origin(event,field) for field in query['fields']}
            selected.append({'event_id': event['model_call_id'], **payload})
        resolved.append((query, events, start, selected))
    remaining = byte_budget
    pages = []
    for query, events, start, selected in resolved:
        page = {'hypothesis_id': query['hypothesis_id'], 'after_event_id': query['after_event_id'],
            'events': [], 'event_count_total': len(events), 'requested_limit': query['limit'],
            'fields': query['fields'], 'next_after_event_id': query['after_event_id'],
            'event_bytes': 0, 'status': 'complete',
            'scope': 'Exact model-authored event data or explicitly requested JSON-pointer subtrees. Reported outcomes are unverified. The byte budget is not a model-token coverage guarantee.'}
        for event in selected:
            size = len(json.dumps(event, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
            if size > remaining:
                page.update(status='capacity_exceeded' if not page['events'] else 'partial',
                    blocked_event_id=event['event_id'], blocked_event_bytes=size)
                break
            page['events'].append(event)
            page['next_after_event_id'] = event['event_id']
            page['event_bytes'] += size
            remaining -= size
        page['has_more'] = start + len(page['events']) < len(events)
        page['remaining_event_byte_budget'] = remaining
        if page['has_more'] and page['status'] == 'complete':
            page['status'] = 'page'
        if include_field_index:
            page['requested_event_field_index']=[history_field_index(event)
                for event in events[start:start+query['limit']]]
            page['delivery_receipt']={
                'version': HISTORY_DELIVERY_VERSION,
                'requested_event_ids':[event['event_id'] for event in selected],
                'delivered_event_ids':[event['event_id'] for event in page['events']],
                'requested_fields':deepcopy(query['fields']),
                'delivery_mode':'selected_fields' if query['fields'] else 'whole_events',
                'requested_event_count':len(selected), 'delivered_event_count':len(page['events']),
                'content_delivered':bool(page['events']),
                'whole_events_delivered':bool(page['events']) and not bool(query['fields']),
                'all_requested_events_delivered':len(page['events'])==len(selected),
                'not_delivered_requested_event_ids':[event['event_id'] for event in selected[len(page['events']):]],
                'scope':'Requested mode is distinct from actual content/count. all_requested_events_delivered covers this requested page,not the entire task history. Zero delivered events means no event text was received;metadata,has_more and omitted events do not establish reading or understanding.'}
        pages.append(page)
    return pages


class FrameArchive(Mapping):
    def __init__(self, root, cache_size=8):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.cache_size = cache_size
        self.references = {}
        self.cache = OrderedDict()
        self.read_only = False
        self._save_index()

    @classmethod
    def open_read_only(cls, root, cache_size=8):
        store = cls.__new__(cls)
        store.root = Path(root)
        index = json.loads((store.root/'index.json').read_text(encoding='utf-8'))
        if index.get('schema') != 1 or not isinstance(index.get('frames'), dict):
            raise ValueError('Invalid archived capture index')
        store.references = index['frames']
        for ref in store.references.values():
            if not (store.root/ref['path']).resolve().is_relative_to(store.root.resolve()):
                raise ValueError('Archived RGB path escapes its capture store')
        store.cache_size = cache_size
        store.cache = OrderedDict()
        store.read_only = True
        return store

    def _save_index(self):
        atomic_json_write(self.root/'index.json', {'schema': 1, 'frames': self.references})

    def __len__(self):
        return len(self.references)

    def __iter__(self):
        return iter(self.references)

    def __contains__(self, identity):
        return identity in self.references

    def _cache(self, identity, frame):
        frame.flags.writeable = False
        self.cache[identity] = frame
        self.cache.move_to_end(identity)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

    def __setitem__(self, identity, frame):
        if self.read_only:
            raise ValueError('Archived memory is read-only; native continuation is not supported')
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError('Memory capture must be uint8 RGB')
        pixel_sha = hashlib.sha256(frame.tobytes()).hexdigest()
        if identity in self.references:
            if self.references[identity]['rgb_sha256'] != pixel_sha:
                raise ValueError('Observation identity cannot be rebound to different RGB')
            return
        buffer = io.BytesIO()
        Image.fromarray(frame).save(buffer, format='PNG')
        data = buffer.getvalue()
        name = pixel_sha + f'-{frame.shape[1]}x{frame.shape[0]}.png'
        path = self.root/name
        if not path.exists():
            with path.open('xb') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        elif path.read_bytes() != data:
            raise ValueError('Existing memory raster conflicts with its content address')
        self.references[identity] = {'path': name, 'rgb_sha256': pixel_sha,
            'file_sha256': hashlib.sha256(data).hexdigest(), 'shape': list(frame.shape)}
        self._save_index()
        self._cache(identity, frame.copy())

    def __getitem__(self, identity):
        ref = self.references[identity]
        if identity in self.cache:
            self.cache.move_to_end(identity)
            return self.cache[identity]
        data = (self.root/ref['path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != ref['file_sha256']:
            raise ValueError('Archived RGB file SHA mismatch: ' + identity)
        with Image.open(io.BytesIO(data)) as image:
            frame = np.array(image.convert('RGB'))
        if list(frame.shape) != ref['shape'] or hashlib.sha256(frame.tobytes()).hexdigest() != ref['rgb_sha256']:
            raise ValueError('Archived RGB pixels mismatch: ' + identity)
        self._cache(identity, frame)
        return frame


class EpisodeMemory:
    def __init__(self, root, cache_size=8):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.frames = FrameArchive(self.root/'rgb', cache_size)
        self.events = self.root/'events.jsonl'
        self.events.touch(exist_ok=False)
        self.sequence = 0
        self._journal_write_failed = False
        self._plan_slots = {}

    @classmethod
    def branch_from(cls, source, root, *, expected_files):
        """Create a new journal from verified history; never make a reader writable.

        The caller must separately establish the reconstructed physical endpoint
        and allocate fresh live IDs. This method grants no native dispatch.
        """
        source, root = Path(source).resolve(), Path(root).resolve()
        if root == source or root.is_relative_to(source) or source.is_relative_to(root):
            raise ValueError('Memory branch must be separate from its source')
        if root.exists():
            raise FileExistsError('Memory branch already exists; never resume an uncertain import')
        if not isinstance(expected_files, dict) or not expected_files:
            raise ValueError('Pinned source file inventory required for a memory branch')
        files = {}
        for path in source.rglob('*'):
            if path.is_symlink():
                raise ValueError('Linked memory source is not supported')
            if path.is_file():
                files[path.relative_to(source).as_posix()] = path.read_bytes()
        actual = {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}
        if actual != expected_files:
            raise ValueError('Memory source differs from its pinned original inventory')
        # Validate all committed plans and every RGB before creating a destination.
        from observation_memory import ObservationMemory
        history = ObservationMemory.from_archive(source)
        events = cls.read_events(source)
        plan_ids = {e['payload']['hypothesis_id'] for e in events if e['kind'] == 'plan_state'}
        for identity in plan_ids:
            cls.read_plan(source, identity)
        for identity in history.frames:
            history.frames[identity]
        operations = OperationHistory.from_events(events)
        if any(r['phase'] in {'pending', 'effect_unknown'} for r in operations.records):
            raise ValueError('Unresolved operation cannot authorize a new memory branch')
        # Detect a source changing while the readers inspected it.
        if any((source/name).read_bytes() != raw for name, raw in files.items()):
            raise ValueError('Memory source changed during validation')
        branch = cls(root)
        try:
            for name, raw in files.items():
                target = root/name
                target.parent.mkdir(parents=True, exist_ok=True)
                if name in {'events.jsonl', 'rgb/index.json'}:
                    target.write_bytes(raw)  # Only the fresh empty destination files.
                else:
                    with target.open('xb') as stream:
                        stream.write(raw)
            branch.frames.references = deepcopy(history.frames.references)
            branch.sequence = len(events)
            for identity in plan_ids:
                last = next(e['payload'] for e in reversed(events)
                            if e['kind'] == 'plan_state' and e['payload']['hypothesis_id'] == identity)
                name = last['path']
                branch._plan_slots[identity] = int(name[-6]) if '-slot-' in name else 0
            branch.append('memory_branch_origin', {
                'version': 'verified-memory-branch-v1',
                'source_events_sha256': actual['events.jsonl'],
                'source_files': actual, 'inherited_event_count': len(events),
                'scope': 'Imported historical model experience only. New writable journal; '
                         'no native action replay, physical checkpoint or model-understanding claim.'})
        except BaseException:
            branch._journal_write_failed = True
            raise
        return branch

    def append(self, kind, payload):
        if self._journal_write_failed:
            raise RuntimeError('Memory journal durability is uncertain; further writes refused, no replay')
        event = {'sequence': self.sequence + 1, 'kind': kind, 'payload': deepcopy(payload)}
        encoded = json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n'
        try:
            with self.events.open('a', encoding='utf-8', newline='\n') as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            # A complete row may already exist. Do not reuse its sequence or
            # attempt another memory commit after an uncertain write.
            self._journal_write_failed = True
            raise
        self.sequence += 1

    def store_plan(self, item):
        """Immutable events are journaled; latest full history is available by model ID."""
        if self._journal_write_failed:
            raise RuntimeError('Memory journal durability is uncertain; plan write refused, no replay')
        digest = hashlib.sha256(item['id'].encode('utf-8')).hexdigest()
        folder = self.root/'plans'
        folder.mkdir(exist_ok=True)
        # Alternate relative to the committed snapshot, never attempt parity.
        # The prior committed plan survives a snapshot or journal failure.
        slot = 1 - self._plan_slots.get(item['id'], 0)
        relative = 'plans/'+digest+'-slot-'+str(slot)+'.json'
        path = self.root/relative
        atomic_json_write(path, item)
        try:
            self.append('plan_state', {'hypothesis_id': item['id'], 'path': relative,
                'latest_model_call_id': item['events'][-1]['model_call_id'], 'event_count': len(item['events']),
                'snapshot_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'event': item['events'][-1], 'task_state': item.get('task_state', 'unverified')})
        except BaseException:
            self._journal_write_failed = True
            raise
        self._plan_slots[item['id']] = slot

    @staticmethod
    def read_plan(root, identity):
        events = EpisodeMemory.read_events(root)
        states = [event['payload'] for event in events if event['kind'] == 'plan_state'
                  and event['payload']['hypothesis_id'] == identity]
        if not states:
            raise ValueError('No recorded plan for this hypothesis ID')
        digest = hashlib.sha256(identity.encode('utf-8')).hexdigest()
        relative = states[-1].get('path')
        allowed = {'plans/'+digest+'.json'} | {'plans/'+digest+'-slot-'+str(i)+'.json' for i in (0,1)}
        if relative not in allowed:
            raise ValueError('Plan journal path differs from its hypothesis identity')
        path = Path(root)/relative
        if path.is_symlink() or path.parent.is_symlink():
            raise ValueError('Linked plan snapshot refused')
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != states[-1]['snapshot_sha256']:
            raise ValueError('Plan snapshot differs from its committed memory event')
        item = json.loads(data.decode('utf-8'))
        if item.get('id') != identity:
            raise ValueError('Plan identity mismatch')
        return item

    @staticmethod
    def read_events(root):
        """Offline inspection only. A partial tail fails; this never replays actions."""
        path = Path(root)/'events.jsonl'
        data = path.read_bytes()
        if data and not data.endswith(b'\n'):
            raise ValueError('Incomplete memory journal tail; no automatic recovery or replay')
        events = [json.loads(line) for line in data.decode('utf-8').splitlines()]
        if [event['sequence'] for event in events] != list(range(1, len(events)+1)):
            raise ValueError('Memory journal sequence is incomplete')
        return events
