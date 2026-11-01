"""Bounded model-authored label review. No native authority or hidden truth labels."""
from copy import deepcopy
import json

AUDITOR_EVIDENCE_CONTRACT='balanced-auditor-evidence-v1'

SCHEMA = {'type':'object','properties':{
    'status':{'type':'string','enum':['resolved','unresolved']},
    'label':{'type':'string','minLength':1,'maxLength':128},
    'reason':{'type':'string','minLength':1,'maxLength':1024},
    'observation_ids':{'type':'array','maxItems':8,'items':{'type':'string','minLength':1,'maxLength':128}}},
    'required':['status','label','reason','observation_ids'],'additionalProperties':False}

SELECTION_SCHEMA = {'type':'object','properties':{
    'verdict':{'type':'string','enum':['consistent','conflict','uncertain']},
    'planner_label':{'type':'string','minLength':1,'maxLength':128},
    'selected_label':{'type':'string','minLength':1,'maxLength':128},
    'reason':SCHEMA['properties']['reason'],
    'observation_ids':SCHEMA['properties']['observation_ids']},
    'required':['verdict','planner_label','selected_label','reason','observation_ids'],
    'additionalProperties':False}


def parse_selection_review(text):
    value=json.loads(text)
    if not isinstance(value,dict) or set(value)!=set(SELECTION_SCHEMA['required']):
        raise ValueError('Invalid selection review fields')
    if value['verdict'] not in ('consistent','conflict','uncertain'):
        raise ValueError('Invalid selection review verdict')
    for label in ('planner_label','selected_label'):
        parse_review(json.dumps({'status':'resolved' if value['verdict']!='uncertain' else 'unresolved',
            'label':value[label],'reason':value['reason'],'observation_ids':value['observation_ids']}))
    return value


def parse_review(text):
    value=json.loads(text)
    if not isinstance(value,dict) or set(value)!=set(SCHEMA['required']):
        raise ValueError('Invalid review fields')
    if value['status'] not in ('resolved','unresolved'):
        raise ValueError('Invalid review status')
    for name,limit in [('label',128),('reason',1024)]:
        if not isinstance(value[name],str) or not value[name].strip() or len(value[name])>limit:
            raise ValueError('Invalid review '+name)
    ids=value['observation_ids']
    if not isinstance(ids,list) or len(ids)>8 or any(not isinstance(x,str) or not x.strip() or len(x)>128 for x in ids):
        raise ValueError('Invalid review observation IDs')
    if value['status']=='resolved' and not ids:
        raise ValueError('Resolved labels require observation citations')
    return value


def review_messages(role, packet, frames, build_payload):
    if role not in ('planner_review','auditor','selection_inspector'):
        raise ValueError('Unknown reviewer role')
    system=('Review the disputed object label using only supplied observations and images. '
        'Native object categories, previous labels and model claims have separate provenance; '
        'none is visual ground truth. Preserve the target alias; a label does not rename an alias. '
        'Return status, label, reason and observation_ids as JSON. Use unresolved when evidence '
        'cannot decide; do not invent certainty or authorize an action. ')
    system+=('Reconsider the planner claim once; unresolved disagreement may be arbitrated. '
             'When defending the original planner_label, copy that label exactly rather than paraphrasing it.'
             if role=='planner_review' else
             'You are a temporary adjudicator in a fresh call. Assess evidence independently '
             'of which participant defended a claim. Your label is provisional, not a tool command.')
    if role=='selection_inspector':
        system=('Independently compare the selected actor target with the planner intent in the supplied '
            'context and RGB. Do not rely on actor matches_intent. Return verdict consistent/conflict/uncertain, '
            'planner_label, selected_label, reason and observation_ids. Labels are model interpretations, '
            'native metadata is not visual truth. Preserve aliases. consistent requires supported correspondence; '
            'use uncertain if either identity or intended use lacks evidence. During the enabled thinking '
            'phase inspect the RGB target before comparing it with the planner intent. Give short identity '
            'labels, not copied actor explanations. Distinguish what the RGB shows from metadata categories '
            'and prior model claims. Do not authorize commands.')
    system+=(' Decide the narrow identity/evidence question and finish once supported. '
        'Do not repeat the same alternatives to fill thinking. Assess the supplied pixels and '
        'actual results independently; no participant or majority is authoritative. '
        'Report unresolved when the supplied evidence cannot discriminate the claims.')
    # Original actor view bindings describe physical RGB slots, including shared slots.
    dispute=packet.get('dispute')
    context=packet.get('actor_context') or (dispute.get('actor_context') if isinstance(dispute,dict) else None)
    labels=None
    if isinstance(context,dict) and context.get('image_views'):
        from policy_io import image_identity_labels
        labels=image_identity_labels(context,len(frames),'image_views')
    content=(build_payload(json.dumps(packet,ensure_ascii=False),frames,labels)
             if labels is not None else build_payload(json.dumps(packet,ensure_ascii=False),frames))
    return [{'role':'system','content':system},{'role':'user','content':content}]


def resolve_dispute(invoke, dispute, observation_ids):
    """One planner review, at most one stateless arbitration; never retries a call.

    invoke(role, packet) returns {call_id, assessment}. Caller owns RGB delivery,
    original model evidence, task memory and applying a returned provisional label.
    """
    original=deepcopy(dispute)
    allowed=set(observation_ids)
    records=[]
    packet={'dispute':deepcopy(original)}
    for role in ('planner_review','auditor'):
        delivered=auditor_packet(original) if role=='auditor' else deepcopy(packet)
        reply=invoke(role,delivered)
        if not isinstance(reply.get('call_id'),str) or not reply['call_id']:
            raise ValueError('Review requires its own model call ID')
        assessment=parse_review(json.dumps(reply['assessment']))
        if not set(assessment['observation_ids']) <= allowed:
            raise ValueError('Review cites an observation not delivered')
        records.append({'version':'review-model-assessment-v1','role':role,'call_id':reply['call_id'],
                        'assessment':assessment,'cost':deepcopy(reply.get('cost',{}))})
        defending_original = (role=='planner_review' and
                              assessment['label'].strip().casefold()==str(original.get('planner_label','')).strip().casefold())
        if assessment['status']=='resolved' and not defending_original:
            break
        packet['planner_review']=deepcopy(records[0])
    final=records[-1]
    return {'version':'provisional-label-review-v1','original_dispute':original,
            'reviews':records,'status':final['assessment']['status'],
            'provisional_label':final['assessment']['label'] if final['assessment']['status']=='resolved' else None,
            'label_author_call_id':final['call_id'],
            'requires_new_observation':final['assessment']['status']=='unresolved',
            'scope':'Model-authored assessment; no native action authorized; original alias/claims retained'}


def auditor_packet(dispute):
    """Symmetric hypotheses and shared evidence, independent of claimant role.

    Caller passes this actual packet to ModelEvidence before invoking the model.
    Native extent/category have explicit provenance, never certify visual identity.
    """
    context=dispute.get('actor_context',{})
    action=dispute.get('selected_action',{})
    alias=action.get('objectId')
    reference={'target_alias':alias,'scope':'Selected alias only; not an asserted visual identity'}
    shared_evidence={}
    for obj in dispute.get('actor_observation',{}).get('objects',[]):
        if obj.get('objectId')==alias:
            if isinstance(obj.get('rendered_evidence'),dict):
                reference['rendered_evidence']=deepcopy(obj['rendered_evidence'])
                extent=obj['rendered_evidence']
                box=extent.get('pixel_bounds_xyxy');size=extent.get('image_size_wh')
                if (isinstance(box,list) and len(box)==4 and isinstance(size,list) and len(size)==2
                        and all(type(n) in (int,float) for n in box+size)
                        and 0 <= box[0] <= box[2] <= size[0] and 0 <= box[1] <= box[3] <= size[1]):
                    reference['localization']={'left_x_px':box[0],'top_y_px':box[1],
                        'right_x_px':box[2],'bottom_y_px':box[3],
                        'image_width_px':size[0],'image_height_px':size[1],
                        'coordinate_system':'Origin at top-left; x increases right, y increases down',
                        'source':'Controller naming of recorded native-assisted extent; not semantic recognition'}
            if isinstance(obj.get('objectType'),str):
                shared_evidence['native_category']={'value':obj['objectType'],
                    'source':'recorded actor native-assisted observation.objectType',
                    'scope':'Interaction metadata category; not visual recognition, intended use or a required answer'}
            break
    bindings={k:deepcopy(context[k]) for k in ('image_views','image_slot_contract','shared_image_contract') if k in context}
    return {'version':AUDITOR_EVIDENCE_CONTRACT,'actor_context':bindings,'target_reference':reference,'shared_evidence':shared_evidence,
            'candidate_labels':sorted({dispute[k] for k in ('planner_label','conflicting_label') if isinstance(dispute.get(k),str)}),
            'evidence_boundary':'Task goals, claimant roles, model defences and actor selection_report withheld. Each candidate is an equally formatted unverified hypothesis, not an exhaustive answer. RGB, localization and native category are separate shared evidence sources. Disagreement is not solved by task preference or automatically by metadata. Explain conflicts and return unresolved if evidence cannot distinguish identity; no action authorization.'}


def review_selection(invoke, packet, observation_ids):
    """Independent check on every selected target, including actor matches_intent.

    A conflict always returns to planning; even a resolved label never releases
    the already proposed command. A later actor selection is checked afresh.
    """
    independent_packet=deepcopy(packet)
    # The current actor's verdict is the assertion under review, not an input
    # for the first independent judgment. Full original action stays retained.
    if isinstance(independent_packet.get('selected_action'),dict):
        independent_packet['selected_action'].pop('selection_report',None)
    independent_packet['inspection_boundary']='Current actor selection_report withheld; selected alias/action, planner intent and delivered observations retained'
    checked=invoke('selection_inspector',independent_packet)
    inspection=parse_selection_review(json.dumps(checked['assessment']))
    if not checked.get('call_id') or not set(inspection['observation_ids']) <= set(observation_ids):
        raise ValueError('Selection inspection requires its call ID and delivered citations')
    result={'version':'selection-review-gate-v1','inspection':deepcopy(checked),
            'release_current_action':inspection['verdict']=='consistent',
            'scope':'Model assessment of correspondence, not physical safety certification'}
    if inspection['verdict']!='consistent':
        dispute={**deepcopy(packet),'planner_label':inspection['planner_label'],
                 'conflicting_label':inspection['selected_label'],'independent_inspection':deepcopy(checked)}
        result['label_review']=resolve_dispute(invoke,dispute,observation_ids)
        # Full reviewer inputs already belong to ModelEvidence, not a recursively
        # nested previous_execution payload carried into every future round.
        result['label_review']['original_dispute']={
            'planner_label':inspection['planner_label'],
            'conflicting_label':inspection['selected_label'],
            'selected_action':deepcopy(packet.get('selected_action')),
            'actor_call_id':packet.get('actor_call_id'),
            'inspection_call_id':checked['call_id'],
            'input_retention':'Full context/RGB in original selection_inspector model call input'}
    return result


def project_selection_review(value):
    """Validate the bounded controller envelope before history publication."""
    if value.get('version')!='selection-review-gate-v1':
        raise ValueError('Unknown decision review version')
    checked=value['inspection']
    if checked.get('version')!='review-model-assessment-v1' or checked.get('role')!='selection_inspector':
        raise ValueError('Incorrect inspection author role')
    inspection=parse_selection_review(json.dumps(checked['assessment']))
    if not isinstance(checked.get('call_id'),str) or not checked['call_id']:
        raise ValueError('Missing inspection call')
    if type(value['release_current_action']) is not bool or value['release_current_action'] != (inspection['verdict']=='consistent'):
        raise ValueError('Review release contradicts inspection')
    if not value['release_current_action']:
        arbitration=value['label_review']
        records=arbitration['reviews']
        if not 1 <= len(records) <= 2 or [r['role'] for r in records] != ['planner_review','auditor'][:len(records)]:
            raise ValueError('Invalid review sequence')
        for record in records:
            parse_review(json.dumps(record['assessment']))
            if record.get('version')!='review-model-assessment-v1':
                raise ValueError('Missing review authorship envelope')
            if not isinstance(record.get('call_id'),str) or not record['call_id']:
                raise ValueError('Missing reviewer call')
        if arbitration['label_author_call_id'] != records[-1]['call_id']:
            raise ValueError('Incorrect label author')
        assessment=records[-1]['assessment']
        unresolved=assessment['status']=='unresolved'
        if (arbitration['status']!=assessment['status']
                or arbitration['requires_new_observation'] is not unresolved
                or arbitration['provisional_label'] != (None if unresolved else assessment['label'])):
            raise ValueError('Derived arbitration fields disagree with original assessment')
    return deepcopy(value)
