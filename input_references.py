"""Bounded,typed,lossless sharing of repeated model-input values only."""
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import time

VERSION='lossless-shared-input-v1'
STRUCTURED_RECORD_VERSIONS=('decision-execution-provenance-v1','actor-intent-selection-v1','delivered-scalar-origin-v1')
REF='$input_ref'
POOL='shared_input_values'
VALUE_ID_PREFIX='input-value-'
MAX_SHARED_VALUES=64
RECORD_TABLE_KEY='$input_records'
RECORD_TABLE_VERSION='exact-record-columns-v1'
RECORD_TABLE_PATHS=(('observation_index',),('operation_history','records'),('execution_feedback_chain','entries'))
RECORD_TABLE_CONTRACT={
    'version':RECORD_TABLE_VERSION,'key':RECORD_TABLE_KEY,
    'scope':'An object containing only $input_records encodes an ordered list of records: columns gives exact field names; each rows array gives values in that order. Reconstruct every row as an object before reading original JSON paths or citing evidence. No row,field,value,type,author or order is omitted. Current image bindings,tools and intent stay literal.'}
PROTECTED={'views','image_views','observation_index','observation','robot_operation_contract',
           'actor_tool_scope','response_roles','input_reference_contract','task','goal','task_goal',
           'plan_steps','task_contract','own_inventory','remaining_budget','information_boundary','image_slot_contract','rgb_rendering_contract',
           'context_evidence_contract','execution_feedback_chain','actor_selection_contract'}
CONTRACT={'version':VERSION,'reference_key':REF,'value_table':POOL,
          'max_shared_values':MAX_SHARED_VALUES,
          'structured_record_versions':list(STRUCTURED_RECORD_VERSIONS),
          'scope':'Table keys input-value-* name exact shared JSON values,never observation IDs,image slots,model calls or action targets. Resolve their values when reading;original typed facts and source/version roles remain. References are input-only:emit concrete values under the normal output schema,never reference objects. Image mappings,current tools and scene remain literal. This shares data,not tasks or semantic conclusions.'}


class InputReferenceError(ValueError):pass
class InputReferenceDepthError(InputReferenceError):pass


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)


def enabled(packet):
    contract=packet.get('input_reference_contract') if isinstance(packet,dict) else None
    return isinstance(contract,dict) and contract.get('version')==VERSION


def _kind(value):
    if isinstance(value,dict):return 'object'
    if isinstance(value,list):return 'array'
    if isinstance(value,str):return 'string'
    if type(value) is bool:return 'boolean'
    if value is None:return 'null'
    if type(value) in (int,float):return 'number'
    raise InputReferenceError('Shared value is not a JSON type')


def decode_input_packet(packet,max_depth=2):
    if not isinstance(packet,dict):raise InputReferenceError('Input packet must be an object')
    if not enabled(packet):return deepcopy(packet)
    pool=packet.get(POOL,{})
    if not isinstance(pool,dict):raise InputReferenceError('Shared input table must be an object')
    tables=packet.get('input_reference_contract',{}).get('record_tables')==RECORD_TABLE_CONTRACT
    def decode(value,stack=(),table_depth=0):
        if tables and isinstance(value,dict) and RECORD_TABLE_KEY in value:
            if set(value)!={RECORD_TABLE_KEY} or table_depth:
                raise InputReferenceError('Malformed or nested record table')
            table=value[RECORD_TABLE_KEY]
            if not isinstance(table,dict) or set(table)!={'columns','rows'}:
                raise InputReferenceError('Malformed record table fields')
            columns,rows=table['columns'],table['rows']
            if (not isinstance(columns,list) or not columns or
                    not all(isinstance(k,str) for k in columns) or len(set(columns))!=len(columns) or
                    any(k in {REF,RECORD_TABLE_KEY} for k in columns) or
                    not isinstance(rows,list) or not rows or
                    any(not isinstance(row,list) or len(row)!=len(columns) for row in rows)):
                raise InputReferenceError('Record table column/row mismatch')
            return [{key:decode(item,stack,table_depth+1) for key,item in zip(columns,row)} for row in rows]
        if isinstance(value,dict) and REF in value:
            if set(value)!={REF} or not isinstance(value[REF],str):raise InputReferenceError('Malformed input reference')
            identity=value[REF]
            if identity not in pool or identity in stack:raise InputReferenceError('Dangling or cyclic input reference')
            if len(stack)>=max_depth:raise InputReferenceDepthError('Input reference nesting exceeds contract')
            entry=pool[identity]
            if not isinstance(entry,dict) or set(entry)!={'kind','value'}:raise InputReferenceError('Malformed shared value')
            resolved=decode(entry['value'],stack+(identity,),table_depth)
            if _kind(resolved)!=entry['kind']:raise InputReferenceError('Shared value type changed')
            return resolved
        if isinstance(value,dict):return {k:decode(v,stack,table_depth) for k,v in value.items()}
        if isinstance(value,list):return [decode(v,stack,table_depth) for v in value]
        return deepcopy(value)
    return {k:decode(v) for k,v in packet.items() if k!=POOL}


def _record_tables(packet,min_gain):
    """Share column spellings on three historical lists, never select their facts."""
    selections=[]
    if packet.get('input_reference_contract',{}).get('record_tables')!=RECORD_TABLE_CONTRACT:
        return packet,selections
    result=deepcopy(packet)
    def reserved(value):
        if isinstance(value,dict):return RECORD_TABLE_KEY in value or any(reserved(v) for v in value.values())
        return isinstance(value,list) and any(reserved(v) for v in value)
    if reserved(result):raise InputReferenceError('Logical input uses reserved record-table key')
    for path in RECORD_TABLE_PATHS:
        node=result
        for key in path[:-1]:
            if not isinstance(node,dict) or key not in node:break
            node=node[key]
        else:
            rows=node.get(path[-1]) if isinstance(node,dict) else None
            if not isinstance(rows,list) or len(rows)<2 or not all(isinstance(row,dict) and row for row in rows):continue
            if any(REF in row for row in rows):continue
            columns=sorted(rows[0])
            if not all(sorted(row)==columns for row in rows):continue
            table={RECORD_TABLE_KEY:{'columns':columns,'rows':[[row[k] for k in columns] for row in rows]}}
            gain=len(canonical(rows).encode('utf-8'))-len(canonical(table).encode('utf-8'))
            if gain<min_gain:continue
            node[path[-1]]=table
            selections.append({'path':list(path),'rows':len(rows),'columns':len(columns),
                               'net_serialized_bytes_saved':gain})
    return result,selections


def _paths(packet):
    def walk(value,path):
        if isinstance(value,dict) and REF in value:return
        if isinstance(value,(dict,list,str)):
            if not isinstance(value,str) or len(value.encode('utf-8'))>=96:
                raw=canonical(value).encode('utf-8')
                if len(raw)>=96:yield path,value,raw
        if isinstance(value,dict):
            for key,item in value.items():yield from walk(item,path+(key,))
        elif isinstance(value,list):
            for index,item in enumerate(value):yield from walk(item,path+(index,))
    for key,value in packet.items():
        if key in PROTECTED:continue
        if key==POOL:
            for identity,entry in value.items():yield from walk(entry['value'],(POOL,identity,'value'))
        else:yield from walk(value,(key,))


def _replace(packet,path,value):
    node=packet
    for key in path[:-1]:node=node[key]
    node[path[-1]]=deepcopy(value)


def _has_reserved(value):
    if isinstance(value,dict):return REF in value or any(_has_reserved(v) for v in value.values())
    if isinstance(value,list):return any(_has_reserved(v) for v in value)
    return False


def _logical_strings(value):
    """Reserve existing opaque identity spellings without interpreting their content."""
    if isinstance(value,str):yield value
    elif isinstance(value,dict):
        for key,item in value.items():
            yield key
            yield from _logical_strings(item)
    elif isinstance(value,list):
        for item in value:yield from _logical_strings(item)


def _value_identity(index,occupied):
    identity=base=f'{VALUE_ID_PREFIX}{index+1:02d}'
    suffix=0
    while identity in occupied:
        suffix+=1;identity=base+'-'+str(suffix)
    return identity


def _sharing_gain(packet,trial,original):
    """Both sharing paths use the same exactness and actual wire-cost boundary."""
    expanded=decode_input_packet(trial)
    if canonical(expanded)!=canonical(original):
        raise InputReferenceError('Input sharing changed original typed fields')
    return len(canonical(packet).encode('utf-8'))-len(canonical(trial).encode('utf-8'))


@lru_cache(maxsize=2)
def _encode_serialized(text,min_gain,max_values):
    started=time.perf_counter();original=json.loads(text)
    if POOL in original or _has_reserved(original):raise InputReferenceError('Logical input uses reserved reference fields')
    occupied=set(_logical_strings(original))
    packet=deepcopy(original);selections=[];skipped=set()
    # Share complete feedback before sharing its children. Child-first pooling
    # can consume the reference-depth/slot budget and leave whole copies behind.
    # Equality is exact typed content, never just a call ID or claimed source.
    for _ in range(max_values):
        groups=defaultdict(list)
        for path,value,raw in _paths(packet):
            if isinstance(value,dict) and ('model_check_progress' in value
                    or value.get('native_dispatch_version')=='native-dispatch-result-v1'):
                groups[raw].append((path,value))
        candidates=[(raw,items) for raw,items in groups.items() if len(items)>1 and raw not in skipped]
        accepted=False
        for raw,items in sorted(candidates,key=lambda pair:(-len(pair[0]),pair[0])):
            trial=deepcopy(packet);identity=_value_identity(len(selections),occupied)
            for path,_ in items:_replace(trial,path,{REF:identity})
            trial.setdefault(POOL,{})[identity]={'kind':'object','value':deepcopy(items[0][1])}
            try:gain=_sharing_gain(packet,trial,original)
            except InputReferenceDepthError:skipped.add(raw);continue
            if gain<=0:skipped.add(raw);continue
            selections.append({'id':identity,'kind':'object','occurrences':len(items),
                'replaced_paths':[list(path) for path,_ in items],
                'encoded_value_sha256':hashlib.sha256(raw).hexdigest(),
                'sharing_rule':'exact_complete_feedback','net_serialized_bytes_saved':gain})
            packet=trial;accepted=True;break
        if not accepted:break
    # Current provenance/report records have one authoritative typed value even
    # when last feedback,operation tail and task lineage repeat it. Reserve at
    # most four existing reference slots; no new table,summary or record journal.
    if original.get('input_reference_contract',{}).get('structured_record_versions')==list(STRUCTURED_RECORD_VERSIONS):
        for _ in range(min(4,max_values-len(selections))):
            groups=defaultdict(list)
            for path,value,raw in _paths(packet):
                if isinstance(value,dict) and value.get('version') in STRUCTURED_RECORD_VERSIONS:
                    groups[raw].append((path,value))
            candidates=[(raw,items) for raw,items in groups.items() if len(items)>1 and raw not in skipped]
            accepted=False
            for raw,items in sorted(candidates,key=lambda pair:(-len(pair[1]),pair[0])):
                trial=deepcopy(packet);identity=_value_identity(len(selections),occupied)
                for path,_ in items:_replace(trial,path,{REF:identity})
                trial.setdefault(POOL,{})[identity]={'kind':'object','value':deepcopy(items[0][1])}
                try:gain=_sharing_gain(packet,trial,original)
                except InputReferenceDepthError:skipped.add(raw);continue
                if gain<=0:skipped.add(raw);continue
                selections.append({'id':identity,'kind':'object','occurrences':len(items),
                    'replaced_paths':[list(path) for path,_ in items],
                    'encoded_value_sha256':hashlib.sha256(raw).hexdigest(),
                    'sharing_rule':'exact_structured_record',
                    'net_serialized_bytes_saved':gain})
                packet=trial;accepted=True;break
            if not accepted:break
    for _ in range(max_values):
        if len(selections)>=max_values:break
        groups=defaultdict(list)
        for path,value,raw in _paths(packet):groups[raw].append((path,value))
        candidates=[]
        for raw,items in groups.items():
            if len(items)<2 or raw in skipped:continue
            gain=(len(items)-1)*len(raw)-len(items)*24-80
            if gain>=min_gain:candidates.append((gain,raw,items))
        if not candidates:break
        accepted=False
        for _,raw,items in sorted(candidates,key=lambda x:(-x[0],x[1])):
            trial=deepcopy(packet);identity=_value_identity(len(selections),occupied);value=deepcopy(items[0][1])
            for path,_ in items:_replace(trial,path,{REF:identity})
            trial.setdefault(POOL,{})[identity]={'kind':_kind(value),'value':value}
            try:gain=_sharing_gain(packet,trial,original)
            except InputReferenceDepthError:skipped.add(raw);continue
            if gain<min_gain:skipped.add(raw);continue
            packet=trial;accepted=True
            selections.append({'id':identity,'kind':_kind(value),'occurrences':len(items),
                               'replaced_paths':[list(path) for path,_ in items],
                               'encoded_value_sha256':hashlib.sha256(raw).hexdigest(),
                               'net_serialized_bytes_saved':gain})
            break
        if not accepted:break
    packet,record_tables=_record_tables(packet,min_gain)
    if canonical(decode_input_packet(packet))!=canonical(original):raise InputReferenceError('Input reference roundtrip failed')
    wire=json.dumps(packet,ensure_ascii=False,separators=(',',':'),allow_nan=False)
    if len(wire.encode('utf-8'))>len(text.encode('utf-8')):
        raise InputReferenceError('Input sharing expanded the serialized wire input')
    profile={'version':VERSION,'roundtrip_verified':True,'reference_depth_limit':2,
             'logical_packet_sha256':hashlib.sha256(canonical(original).encode('utf-8')).hexdigest(),
             'wire_text_sha256':hashlib.sha256(wire.encode('utf-8')).hexdigest(),
             'logical_compact_bytes':len(text.encode('utf-8')),'wire_bytes':len(wire.encode('utf-8')),
             'shared_value_count':len(selections),'selections':selections,
             'encoder_miss_seconds':time.perf_counter()-started,
             'scope':'Exact typed input projection only;not proof the model used references correctly;no image or source fact omitted'}
    if record_tables:profile['record_tables']=record_tables
    return wire,json.dumps(profile,ensure_ascii=False,separators=(',',':'))


def encode_input_packet(packet,min_gain=128,max_values=MAX_SHARED_VALUES):
    if not enabled(packet):return deepcopy(packet),None
    if type(min_gain) is not int or min_gain<1 or type(max_values) is not int or not 1<=max_values<=MAX_SHARED_VALUES:
        raise InputReferenceError('Input-sharing capacity is outside the versioned contract')
    text=json.dumps(packet,ensure_ascii=False,separators=(',',':'),allow_nan=False)
    wire,profile=_encode_serialized(text,min_gain,max_values)
    return json.loads(wire),json.loads(profile)


def shared_input_text(packet):
    wire,profile=encode_input_packet(packet)
    return json.dumps(wire,ensure_ascii=False,separators=(',',':'),allow_nan=False),profile
