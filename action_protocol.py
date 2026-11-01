"""Strict, dependency-free validation of model output before any simulator action."""
import json
import math
import re


OBJECT_ACTIONS = {
    "PickupObject", "PutObject", "OpenObject", "CloseObject", "ToggleObjectOn",
    "ToggleObjectOff", "SliceObject", "CookObject", "CleanObject", "DirtyObject",
    "BreakObject", "FillObjectWithLiquid",
}
MOVE_ACTIONS = {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight"}
ROTATE_ACTIONS = {"RotateLeft", "RotateRight"}
LOOK_ACTIONS = {"LookUp", "LookDown"}
HAND_AXES = ("right", "up", "ahead")
HAND_MOVE_MAX_METRES = 0.25
HAND_MOVE_MAX_ATTEMPTS = 3
ALLOWED_ACTIONS = OBJECT_ACTIONS | MOVE_ACTIONS | ROTATE_ACTIONS | LOOK_ACTIONS | {
    "DropHandObject", "ThrowObject", "Done", "Pass", "MoveHeldObject", "RequestReplan",
}


class ProtocolError(ValueError):
    pass


SELECTION_REPORT_SCHEMA={'type':'object','properties':{
    'intent_path':{'type':'string','minLength':1,'maxLength':256},
    'assessment':{'type':'string','enum':['matches_intent','uncertain','different']},
    'reason':{'type':'string','minLength':1,'maxLength':512},
    'evidence':{'type':'array','maxItems':2,'items':{'type':'object','properties':{
        'observation_id':{'type':'string','minLength':1,'maxLength':64},
        'cue':{'type':'string','minLength':1,'maxLength':512}},
        'required':['observation_id','cue'],'additionalProperties':False}}},
    'required':['intent_path','assessment','reason','evidence'],'additionalProperties':False}


def validate_selection_report(value):
    if not isinstance(value,dict) or set(value)!=set(SELECTION_REPORT_SCHEMA['required']):
        raise ProtocolError('selection_report requires intent_path,assessment,reason,evidence')
    if value['assessment'] not in SELECTION_REPORT_SCHEMA['properties']['assessment']['enum']:
        raise ProtocolError('Invalid model-authored selection assessment')
    for key,maximum in [('intent_path',256),('reason',512)]:
        if not isinstance(value[key],str) or not value[key].strip() or len(value[key])>maximum:
            raise ProtocolError('Invalid selection_report '+key)
    if not value['intent_path'].startswith('/'):
        raise ProtocolError('selection_report.intent_path must be a delivered-input JSON pointer')
    rows=value['evidence']
    if not isinstance(rows,list) or len(rows)>2:raise ProtocolError('selection_report permits at most two RGB citations')
    for row in rows:
        if not isinstance(row,dict) or set(row)!={'observation_id','cue'}:
            raise ProtocolError('Selection RGB citation requires observation_id and cue')
        if any(not isinstance(row[k],str) or not row[k].strip() or len(row[k])>n
               for k,n in [('observation_id',64),('cue',512)]):
            raise ProtocolError('Invalid selection RGB citation')
    return value


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def strip_leading_thinking(text):
    """Remove only explicit leading reasoning blocks; quoted payload data is exact."""
    text = text.strip()
    while text.startswith('<think>'):
        end = text.find('</think>')
        if end < 0:
            raise ProtocolError('Thinking output was truncated before a final response')
        text = text[end+len('</think>'):].lstrip()
    return text


def extract_json(text):
    """Decode one complete object, including legacy fenced replies, without guessing actions."""
    if not isinstance(text, str):
        raise ProtocolError("Response must be text")
    text = strip_leading_thinking(text)
    decoder = json.JSONDecoder(object_pairs_hook=_unique_keys,
                               parse_constant=lambda value: (_ for _ in ()).throw(
                                   ProtocolError(f"Non-finite JSON constant: {value}")))
    objects = []
    position = 0
    while position < len(text):
        match = re.search(r"[\[{]", text[position:])
        if match is None:
            if '<think>' in text[position:]:
                raise ProtocolError('Unexpected reasoning outside the final JSON payload')
            break
        start = position + match.start()
        if '<think>' in text[position:start]:
            raise ProtocolError('Unexpected reasoning before the final JSON payload')
        try:
            value, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError as exc:
            # Do not salvage an inner command from a truncated outer object.
            raise ProtocolError(f"Invalid JSON: {exc.msg}") from exc
        objects.append(value)
        position = end
    if len(objects) != 1 or not isinstance(objects[0], dict):
        raise ProtocolError("Return exactly one JSON object")
    return objects[0]


def validate_action(action):
    if not isinstance(action, dict):
        raise ProtocolError("action must be an object, not a string or list")
    name = action.get("action")
    if not isinstance(name, str) or name not in ALLOWED_ACTIONS:
        raise ProtocolError("Unknown or missing action name")
    allowed = {"action"}
    if 'selection_report' in action:
        allowed.add('selection_report');validate_selection_report(action['selection_report'])
    if name == 'RequestReplan':
        allowed.add('reason')
        if not isinstance(action.get('reason'), str) or not action['reason'].strip() or len(action['reason']) > 512:
            raise ProtocolError('RequestReplan requires a model-authored reason of at most 512 characters')
        if 'target_alias' in action:
            allowed.add('target_alias')
            if (not isinstance(action['target_alias'], str) or not action['target_alias'].strip()
                    or len(action['target_alias']) > 32):
                raise ProtocolError('RequestReplan target_alias must name one current actor candidate')
    if name == "MoveHeldObject":
        allowed.update(HAND_AXES)
        if any(type(action.get(key)) not in (int, float) or not math.isfinite(action[key]) for key in HAND_AXES):
            raise ProtocolError("MoveHeldObject requires finite signed right/up/ahead metres")
        distance = math.hypot(*(action[key] for key in HAND_AXES))
        if not 0 < distance <= HAND_MOVE_MAX_METRES:
            raise ProtocolError("MoveHeldObject vector length must be greater than zero and at most 0.25m")
    if name in OBJECT_ACTIONS:
        allowed.add("objectId")
        target = action.get("objectId")
        if not isinstance(target, str) or not target.strip():
            raise ProtocolError(f"{name} requires a nonempty objectId copied from visible candidates")
    if name in MOVE_ACTIONS | {"ThrowObject"}:
        allowed.add("moveMagnitude")
    if name in ROTATE_ACTIONS | LOOK_ACTIONS:
        allowed.add("degrees")
    if name == "FillObjectWithLiquid":
        allowed.add("fillLiquid")
        if not isinstance(action.get("fillLiquid"), str) or action["fillLiquid"] not in {"water", "coffee", "wine"}:
            raise ProtocolError("FillObjectWithLiquid requires fillLiquid: water, coffee or wine")
    extra = set(action) - allowed
    if extra:
        raise ProtocolError("Unexpected action parameters: " + ", ".join(sorted(extra)))
    for key, maximum in (("moveMagnitude", 5), ("degrees", 90 if name in LOOK_ACTIONS else 360)):
        if key in action:
            value = action[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value) or not 0 < value <= maximum:
                raise ProtocolError(f"{key} must be a finite number greater than 0 and at most {maximum}")
    return dict(action)


def parse_action_response(text):
    value = extract_json(text)
    if set(value) - {"thought", "action"}:
        raise ProtocolError("Only thought and action are allowed in the response")
    thought = value.get("thought", "")
    if not isinstance(thought, str) or len(thought) > 1200:
        raise ProtocolError("thought must be a short string (at most 1200 characters)")
    return {"thought": thought, "action": validate_action(value.get("action"))}


def parse_goal_response(text):
    value = extract_json(text)
    if not all(isinstance(value.get(key), str) for key in ("observation", "goal")):
        raise ProtocolError("Goal response requires observation and goal strings")
    if not value["goal"].strip():
        raise ProtocolError("goal must not be empty")
    return {"observation": value["observation"][:2000], "goal": value["goal"][:2000]}
