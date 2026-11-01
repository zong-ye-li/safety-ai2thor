"""Qwen's native function-call wire format; independent execution validation remains mandatory."""
import itertools
import json
import re
import math
from functools import lru_cache

from action_protocol import ProtocolError, validate_action, HAND_AXES, HAND_MOVE_MAX_METRES, strip_leading_thinking
from action_catalog import CATALOG, contextual_names
from policy_io import observation_packet


def select_protocol(template, requested="auto"):
    native = all(marker in (template or "") for marker in ("<function=", "<parameter="))
    if requested == "auto":
        return "qwen_xml" if native else "action_json"
    if requested == "qwen_xml" and not native:
        raise ValueError("Processor template does not advertise the Qwen function/parameter protocol")
    if requested not in {"qwen_xml", "action_json"}:
        raise ValueError("Unknown policy protocol")
    return requested


def parse_native_call(text):
    if not isinstance(text, str):
        raise ProtocolError("Tool response must be text")
    text = strip_leading_thinking(text)
    match = re.fullmatch(r"<tool_call>\s*<function=([A-Za-z]+)>\s*(.*?)\s*</function>\s*</tool_call>", text, re.S)
    if not match:
        raise ProtocolError("Return exactly one complete native tool_call/function block")
    action = {"action": match[1]}
    body = match[2]
    parameter = re.compile(r"\s*<parameter=([A-Za-z]+)>\s*", re.S)
    decoder = json.JSONDecoder()
    position = 0
    while position < len(body):
        if not body[position:].strip():
            break
        part = parameter.match(body, position)
        if not part:
            raise ProtocolError("Malformed native parameter block")
        key = part[1]
        if key in action:
            raise ProtocolError("Duplicate native parameter: " + key)
        start = part.end()
        if body[start:start+1] == '"':
            try:
                value, end = decoder.raw_decode(body, start)
            except ValueError as exc:
                raise ProtocolError("Malformed quoted tool parameter") from exc
            closing = re.match(r'\s*</parameter>', body[end:])
            if closing is None:
                raise ProtocolError('Quoted tool parameter must end before its closing tag')
            position = end + closing.end()
            action[key] = value
            continue
        end = body.find('</parameter>', start)
        if end < 0:
            raise ProtocolError('Missing native parameter closing tag')
        value = body[start:end].strip()
        if key in {"degrees", "moveMagnitude", *HAND_AXES}:
            try:
                value = json.loads(value)
            except ValueError as exc:
                raise ProtocolError("Numeric tool parameter must be a JSON number") from exc
        elif any(character in value for character in "<>"):
            raise ProtocolError("Unexpected markup in parameter")
        action[key] = value
        position = end + len('</parameter>')
    return {"thought": "", "action": validate_action(action)}


def native_transport_tools(tools):
    """Current XML transport exposes numeric/enum parameters; free text stays JSON-only."""
    return {name:spec for name,spec in tools.items()
            if not any(p.get('type') == 'string' and 'enum' not in p
                       for key,p in spec['properties'].items() if key != 'action')}


def function_tools(tools, task, scene, guide_mode="contextual"):
    """Tool definitions include eligible aliases, never fabricated example targets."""
    tools = native_transport_tools(tools)
    detailed = set(tools) if guide_mode == "full" else set(contextual_names(task, scene))
    result = []
    for name, schema in tools.items():
        item = CATALOG[name]
        purpose = item["purpose"]
        if name in {"ToggleObjectOn", "ToggleObjectOff"}:
            purpose = "Switch a faucet, switch or compatible device " + ("on." if name.endswith("On") else "off.")
        description = purpose
        if guide_mode != "none" and name in detailed:
            description += " Preconditions: " + item["preconditions"] + " Effect: " + item["expected_effect"] + " Failure: " + item["on_failure"]
        properties = {key: {**value, "description": item["parameters"].get(key, "")}
                      for key, value in schema["properties"].items() if key != "action"}
        result.append({"type": "function", "function": {
            "name": name, "description": description,
            "parameters": {"type": "object", "properties": properties,
                           "required": [key for key in schema["required"] if key != "action"],
                           "additionalProperties": False}}})
    return result


def native_call_regex(tools):
    """Grammar preserves every available tool and continuous numeric parameters.

    Numeric range checks remain the responsibility of validate_action; lexical
    enforcement is not a replacement for the independent execution guard.
    """
    tools = native_transport_tools(tools)
    if not tools:
        raise ValueError('No XML-compatible tools; free-text planning reports require action_json')
    ws = r"[ \t\r\n]{0,8}"
    number = r"-?(0|[1-9][0-9]{0,8})(\.[0-9]{1,9})?([eE][+-]?[0-9]{1,3})?"
    alternatives = []
    for name, schema in tools.items():
        parameters = []
        for key, spec in schema["properties"].items():
            if key == "action":
                continue
            if spec["type"] == "number":
                value = number
            elif spec['type'] == 'string' and 'enum' not in spec:
                # Bounded model-authored text, either plain without markup or
                # JSON-quoted with escaped markup. It cannot create another call.
                maximum = spec.get('maxLength', 512)
                quoted = r'"(?:[^"\\\x00-\x1f<>]|\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4})){1,' + str(maximum) + r'}"'
                plain = r'[^"\x00-\x1f<>][^\x00-\x1f<>]{0,' + str(maximum-1) + r'}'
                value = '(' + quoted + '|' + plain + ')'
            else:
                values = spec["enum"]
                value = "(" + "|".join(re.escape(form) for value in values
                                         for form in (value, json.dumps(value))) + ")"
            block = re.escape("<parameter=" + key + ">") + ws + value + ws + re.escape("</parameter>") + ws
            parameters.append((block, key in schema["required"]))
        orders = itertools.permutations(parameters) if parameters else [()]
        for order in orders:
            body = "".join(block if required else "(" + block + ")?" for block, required in order)
            alternatives.append(re.escape("<tool_call>") + ws + re.escape("<function=" + name + ">")
                                + ws + body + re.escape("</function>") + ws + re.escape("</tool_call>"))
    return "(" + "|".join(alternatives) + ")"


def numeric_boundary_allowed(text, limits=None):
    """A numeric parameter may close only when it satisfies the actual contract.

    Keep valid decimal/scientific prefixes available: e.g. 500 can become 500e-2.
    Do not silently clamp or substitute a default for an invalid model value.
    """
    parameter = re.search(r"<parameter=(degrees|moveMagnitude|right|up|ahead)>([^<]*)$", text)
    if not parameter or not parameter[2].strip():
        return True
    function = re.search(r"<function=([A-Za-z]+)>", text)
    try:
        value = json.loads(parameter[2].strip())
        if function[1] == 'MoveHeldObject':
            values = {key:json.loads(raw.strip()) for key,raw in re.findall(
                r'<parameter=(right|up|ahead)>([^<]*)</parameter>',text)}
            values[parameter[1]]=value
            if any(type(v) not in (int,float) or not math.isfinite(v) for v in values.values()):
                return False
            length=math.hypot(*values.values())
            return length<=HAND_MOVE_MAX_METRES and (len(values)<3 or length>0)
        validate_action({"action": function[1], parameter[1]: value})
        spec = (limits or {}).get(function[1], {}).get(parameter[1], {})
        if value < spec.get('minimum', -math.inf) or value > spec.get('maximum', math.inf):
            return False
        step = spec.get("multipleOf")
        if step and not math.isclose(value / step, round(value / step), abs_tol=1e-6):
            return False
        return True
    except (ValueError, TypeError, IndexError):
        return False


@lru_cache(maxsize=32)
def _compiled_native_regex(grammar):
    from lmformatenforcer import RegexParser
    return RegexParser(grammar)


def native_parser(grammar, limits=None):
    """Immutable grammar state plus numeric-domain guards, loaded only by HF."""
    from lmformatenforcer import CharacterLevelParser

    class DomainParser(CharacterLevelParser):
        def __init__(self, core, text="", config=None):
            super().__init__(config)
            self.core, self.text = core, text

        def get_allowed_characters(self):
            allowed = self.core.get_allowed_characters()
            if not numeric_boundary_allowed(self.text, limits):
                allowed = "".join(char for char in allowed if char not in " \t\r\n<")
            return allowed

        def add_character(self, char):
            return DomainParser(self.core.add_character(char), self.text + char, self.config)

        def can_end(self):
            if not self.core.can_end():
                return False
            try:
                parse_native_call(self.text)
                return True
            except ProtocolError:
                return False

        def cache_key(self):
            # Numeric validity depends on prefix content, not just the lexical FSM state.
            return (self.core.cache_key(), self.text)

        @CharacterLevelParser.config.setter
        def config(self, config):
            self._config = config
            self.core.config = config

    return DomainParser(_compiled_native_regex(grammar))


def native_messages(request, scene, current_content, system_extra=""):
    from policy_io import compact_check_feedback
    request = compact_check_feedback(request) if isinstance(request, dict) else {"task": request}
    messages = [{"role": "system", "content": (
        "You control a household robot. Use the supplied tools to advance the user's task. "
        "Return exactly one tool call, without reasoning or prose. Read the CURRENT observation "
        "and actual tool results. Views are observations. A formatting correction changes no physical state. "
        "Copy a current eligible alias for object parameters. Check current state before repeating. "
        + ("The controller ends the task when its postcondition holds. " if request.get("completion_managed_by_controller") else "")
        + system_extra)}, {"role": "user", "content": "Task: " + str(request.get("task", ""))}]
    history = request.get("execution_history")
    if history is None:
        history = [request["previous_execution"]] if request.get("previous_execution") else []
    for index, execution in enumerate(history):
        if execution.get("origin", execution.get("category")) != "policy":
            continue
        action = validate_action(execution["action"])
        call_id = "execution_" + str(index)
        messages.append({"role": "assistant", "content": "", "tool_calls": [{
            "id": call_id, "type": "function", "function": {
                "name": action["action"], "arguments": {key: value for key, value in action.items() if key != "action"}}}]})
        messages.append({"role": "tool", "tool_call_id": call_id,
                         "content": json.dumps({key: execution[key] for key in ("origin", "category", "success", "message", "environment_command_executed") if key in execution})})
    # The native adapter must preserve the same task context as action_json.
    # Images alone do not convey their historical ID/pose mapping, and a saved
    # request is not proof that the model actually received that mapping.
    packet = {key: value for key, value in request.items()
              if key not in {"task", "execution_history", "formatting_correction"}}
    packet["observation"] = observation_packet(scene)
    if request.get("formatting_correction"):
        correction = request["formatting_correction"]
        packet["formatting_correction"] = {
            "error": correction.get("error", "Invalid tool response") if isinstance(correction, dict) else str(correction),
            "physical_state_changed": False,
            "instruction": "Return one complete native tool call with only its declared parameters."}
    # Preserve only the current images. Historical execution facts have their own roles.
    content = [{"type": "text", "text": json.dumps(packet, separators=(",", ":"))}]
    # Drop only the duplicated leading request text. Keep each adjacent identity
    # label with its exact following image; labels contain no semantic answers.
    leading_request=(current_content and current_content[0]['type']=='text'
                     and not current_content[0].get('text','').startswith('RGB image_index='))
    content.extend(current_content[1:] if leading_request else current_content)
    messages.append({"role": "user", "content": content})
    return messages
