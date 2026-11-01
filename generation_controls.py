"""Model/template stop-token alignment and explicit payload framing."""
import hashlib
import json
import math
import re
import time
from decimal import Decimal
from functools import lru_cache

CAMERA_NUMBER_MAX_CHARACTERS = 32


def validate_reasoning_presence_penalty(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 2):
        raise ValueError('Reasoning presence penalty must be finite numeric 0..2')
    return float(value)


def validate_reasoning_repetition_penalty(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 1 <= value <= 2):
        raise ValueError('Reasoning repetition penalty must be finite numeric 1..2')
    return float(value)


def _decimal_signature(value):
    number = Decimal(value)
    digits = ''.join(map(str, number.as_tuple().digits)).rstrip('0')
    return digits, number.adjusted()


@lru_cache(maxsize=128)
def _camera_lattice(step, maximum):
    increment, upper = Decimal(step), Decimal(maximum)
    if increment <= 0 or upper < increment or upper / increment > 10000:
        raise ValueError('Invalid or excessive camera lattice')
    return tuple(_decimal_signature(increment * k) for k in range(1, int(upper / increment) + 1))


def _camera_number_prefix_extendable(number, step, maximum):
    """Any completion to the current stepped camera domain, without choosing one.

    The JSON grammar owns lexical syntax. Mantissa significant digits must be a
    prefix of a legal coefficient, or that whole coefficient plus trailing zeros.
    Once e is emitted the mantissa is fixed; only a compatible exponent may follow.
    """
    domain = _camera_lattice(str(step), str(maximum))
    mantissa, separator, exponent = number.lower().partition('e')
    if not mantissa:
        return True
    if mantissa.startswith('-'):
        return False
    if not separator:
        significant = mantissa.replace('.', '').lstrip('0')
        return not significant or any(digits.startswith(significant) or
            significant.startswith(digits) and not significant[len(digits):].strip('0')
            for digits, _ in domain)
    if not Decimal(mantissa):
        return False
    digits, power = _decimal_signature(mantissa)
    negative = exponent.startswith('-')
    body = exponent[1:] if exponent[:1] in '+-' else exponent
    significant = body.lstrip('0')
    for candidate, adjusted in domain:
        needed = adjusted - power
        if candidate != digits:
            continue
        if exponent and (needed < 0 and not negative or needed > 0 and negative):
            continue
        if not significant or str(abs(needed)).startswith(significant):
            return True
    return False


def camera_number_prefix_allowed(number, step, maximum):
    """A lattice completion must fit the explicit finite JSON camera spelling.

    Only feasibility is returned. No value or suffix is supplied to the model.
    Other action domains and unknown-action field ordering are unchanged.
    """
    if len(number) > CAMERA_NUMBER_MAX_CHARACTERS:
        return False
    if not _camera_number_prefix_extendable(number, step, maximum):
        return False
    if not number:
        return True
    domain = _camera_lattice(str(step), str(maximum))
    mantissa, separator, exponent = number.lower().partition('e')
    if separator:
        digits, power = _decimal_signature(mantissa)
        body = exponent[1:] if exponent[:1] in '+-' else exponent
        if body and not body.isdigit():
            return False
        significant = body.lstrip('0')
        for candidate, adjusted in domain:
            needed = adjusted - power
            if candidate != digits or (needed < 0 and exponent and not exponent.startswith('-')) or (needed > 0 and exponent.startswith('-')):
                continue
            required = str(abs(needed))
            if significant and not required.startswith(significant):
                continue
            suffix = required[len(significant):] if significant else ('' if needed == 0 and body else required)
            sign_length = int(needed < 0 and not exponent)
            if len(number) + sign_length + len(suffix) <= CAMERA_NUMBER_MAX_CHARACTERS:
                return True
        return False
    significant = mantissa.replace('.', '').lstrip('0')
    for digits, adjusted in domain:
        if not significant:
            suffixes = [('.' if mantissa == '0' else '') + digits]
        elif digits.startswith(significant):
            remaining = digits[len(significant):]
            suffixes = [remaining]
            if '.' not in mantissa:
                suffixes.extend(remaining[:i] + '.' + remaining[i:] for i in range(len(remaining)+1))
        elif significant.startswith(digits) and not significant[len(digits):].strip('0'):
            suffixes = ['']
        else:
            continue
        for suffix in suffixes:
            completed_mantissa = mantissa + suffix
            if completed_mantissa.endswith('.'):
                completed_mantissa += '0'
            if not re.fullmatch(r'(?:0|[1-9][0-9]*)(?:\.[0-9]+)?', completed_mantissa):
                continue
            coefficient, power = _decimal_signature(completed_mantissa)
            if coefficient != digits:
                continue
            needed = adjusted - power
            exponent_suffix = '' if needed == 0 else 'e' + str(needed)
            length = len(completed_mantissa) + len(exponent_suffix)
            if needed > 0 and '.' not in completed_mantissa:
                length = min(length, len(completed_mantissa) + needed)
            if length <= CAMERA_NUMBER_MAX_CHARACTERS:
                return True
    return False


def json_camera_prefix_allowed(text, limits):
    active = re.search(r'"degrees"\s*:\s*([0-9.eE+\-]*)$', text)
    name = re.search(r'"action"\s*:\s*"([A-Za-z]+)"', text)
    if active is None or name is None or name[1] not in {'LookUp', 'LookDown'}:
        return True
    spec = limits.get(name[1], {}).get('degrees', {})
    if not spec.get('multipleOf'):
        return True
    return camera_number_prefix_allowed(active[1], spec['multipleOf'], spec['maximum'])


def json_numeric_boundary_allowed(text, limits=None):
    """May the active JSON number close? Never replace or quantize its value."""
    from action_protocol import validate_action, HAND_AXES, HAND_MOVE_MAX_METRES
    active=re.search(r'"(degrees|moveMagnitude|right|up|ahead)"\s*:\s*([^,}\]]*)$',text)
    if active is None or not active[2].strip():
        return True
    name=re.search(r'"action"\s*:\s*"([A-Za-z]+)"',text)
    try:
        value=json.loads(active[2].strip())
        if type(value) not in (int,float) or not math.isfinite(value):
            return False
        key=active[1]
        if key in HAND_AXES:
            values={key:json.loads(raw) for key,raw in re.findall(
                r'"(right|up|ahead)"\s*:\s*([-+0-9.eE]+)\s*[,}]',text)}
            values[key]=value
            if any(type(v) not in (int,float) or not math.isfinite(v) for v in values.values()):
                return False
            distance=math.hypot(*values.values())
            return distance<=HAND_MOVE_MAX_METRES and (len(values)<3 or distance>0)
        # Before action is known use the union of generic numeric domains,
        # preserving legal parameter-first and scientific-number prefixes.
        action_name=name[1] if name else ('RotateRight' if key=='degrees' else 'MoveAhead')
        validate_action({'action':action_name,key:value})
        spec=(limits or {}).get(action_name,{}).get(key,{}) if name else {}
        if value<spec.get('minimum',-math.inf) or value>spec.get('maximum',math.inf):
            return False
        step=spec.get('multipleOf')
        return not step or math.isclose(value/step,round(value/step),abs_tol=1e-6)
    except (ValueError,TypeError):
        return False


def action_json_parser(schema):
    """Real schema grammar plus numeric closure and final availability checks."""
    from lmformatenforcer import JsonSchemaParser,CharacterLevelParser
    from action_protocol import parse_action_response
    from tool_registry import validate_available_action
    tools={s['properties']['action']['enum'][0]:s for s in schema['properties']['action']['anyOf']}
    limits={name:s['properties'] for name,s in tools.items()}

    class DomainParser(CharacterLevelParser):
        def __init__(self,core,text='',config=None):
            self.core,self.text=core,text
            super().__init__(config)
        def get_allowed_characters(self):
            allowed=self.core.get_allowed_characters()
            # Numerical-domain checks constrain JSON numbers, not the contents
            # of a schema-authorized free string. Preserve the library's large-
            # vocabulary shortcut for model-authored reports.
            if self.core.shortcut_key() is not None:
                return allowed
            if not json_numeric_boundary_allowed(self.text,limits):
                allowed=''.join(c for c in allowed if c not in ' \t\r\n,}]')
            allowed=''.join(c for c in allowed if c not in '0123456789.eE+-' or
                            json_camera_prefix_allowed(self.text+c,limits))
            return allowed
        def add_character(self,char):
            return DomainParser(self.core.add_character(char),self.text+char,self.config)
        def can_end(self):
            if not self.core.can_end():return False
            try:
                validate_available_action(parse_action_response(self.text)['action'],tools)
                return True
            except (ValueError,TypeError):return False
        def cache_key(self):return (self.core.cache_key(),self.text)
        def shortcut_key(self):return self.core.shortcut_key()
        @CharacterLevelParser.config.setter
        def config(self,config):
            self._config=config
            self.core.config=config
    return DomainParser(JsonSchemaParser(schema))


def processor_fingerprint(inputs):
    """Hash the exact post-processor CPU tensors for controlled observation replays."""
    import torch
    fields = {}
    for name, value in inputs.items():
        if isinstance(value, torch.Tensor):
            cpu = value.detach().cpu().contiguous()
            raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
            fields[name] = {"dtype": str(cpu.dtype), "shape": list(cpu.shape),
                            "sha256": hashlib.sha256(raw).hexdigest()}
        else:
            fields[name] = {"value": value}
    encoded = json.dumps(fields, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    return {"version": 1, "fields": fields, "sha256": hashlib.sha256(encoded).hexdigest()}


def aligned_eos_ids(model_eos, tokenizer_eos):
    result = []
    for value in (model_eos, tokenizer_eos):
        values = value if isinstance(value, (list, tuple)) else [value]
        for token in values:
            if token is None:
                continue
            if isinstance(token, bool) or not isinstance(token, int) or token < 0:
                raise ValueError("Invalid EOS configuration")
            if token not in result:
                result.append(token)
    return result or None


class GrammarExecutionError(RuntimeError):
    """Infrastructure failure, never a model formatting retry."""


def json_string_boundary_tree(regular_tokens):
    """Tokens missed by the free-text shortcut: text, closing quote, then syntax."""
    root = {'children': {}, 'tokens': []}
    for identity, text, _ in regular_tokens:
        if '"' not in text[1:-1]:
            continue
        node = root
        for char in text:
            node = node['children'].setdefault(char, {'children': {}, 'tokens': []})
        node['tokens'].append(identity)
    return root


def accepted_boundary_tokens(parser, tree):
    """Walk only cached boundary-token prefixes through the complete live grammar."""
    accepted = list(tree['tokens'])
    allowed = parser.get_allowed_characters()
    for char, child in tree['children'].items():
        if char in allowed:
            accepted.extend(accepted_boundary_tokens(parser.add_character(char), child))
    return accepted


def guarded_token_filter(parser, tokenizer_data, builder):
    """Surface parser faults even if the enforcer catches them and forces EOS."""
    from lmformatenforcer.characterlevelparser import CharacterLevelParser
    faults = []

    class GuardedParser(CharacterLevelParser):
        def __init__(self, delegate):
            super().__init__(delegate.config)
            self.delegate = delegate

        def invoke(self, method, *args):
            try:
                return getattr(self.delegate, method)(*args)
            except Exception as exc:
                faults.append({"operation": method, "error": repr(exc)})
                raise

        def add_character(self, char): return GuardedParser(self.invoke("add_character", char))
        def get_allowed_characters(self): return self.invoke("get_allowed_characters")
        def can_end(self): return self.invoke("can_end")
        def shortcut_key(self): return self.invoke("shortcut_key")
        def cache_key(self): return self.invoke("cache_key")

        @property
        def config(self): return self.delegate.config

        @config.setter
        def config(self, value): self.delegate.config = value

    prefix = builder(tokenizer_data, GuardedParser(parser))
    enforcer = getattr(prefix, 'token_enforcer', None)
    boundaries = None
    if enforcer is not None:
        # Per-client CPU tokenizer data only; do not mutate package code, HF
        # assets or the enforcer's cached allowed-token lists.
        boundaries = getattr(tokenizer_data, '_safety_json_boundary_tree', None)
        if boundaries is None:
            boundaries = json_string_boundary_tree(tokenizer_data.regular_tokens)
            tokenizer_data._safety_json_boundary_tree = boundaries

    def checked(batch_id, input_ids):
        result = prefix(batch_id, input_ids)
        if enforcer is not None:
            state = enforcer.prefix_states.get(tuple(input_ids.tolist()))
            if state is not None:
                shortcut = state.parser.shortcut_key()
                if isinstance(shortcut, tuple) and shortcut[0] == 'json_freetext':
                    extra = accepted_boundary_tokens(state.parser, boundaries)
                    if extra:
                        result = list(dict.fromkeys([*result, *extra]))
        if faults:
            raise GrammarExecutionError("Constrained-decoding parser failed: " + repr(faults[0]))
        return result
    return checked


def complete_payload(text, parser):
    try:
        parser(text)
        return True
    except (ValueError, TypeError):
        return False


def final_after_thinking(text, closing="</think>"):
    """Legacy Qwen default; new clients supply their provider's boundary."""
    if closing not in text:
        return None
    return text.split(closing, 1)[1].strip()


def boundary_position(generated, boundary):
    boundary = [boundary] if isinstance(boundary, int) else list(boundary)
    if len(boundary)==1:
        try:return generated.index(boundary[0])
        except ValueError:return None
    return next((i for i in range(len(generated)-len(boundary)+1)
                 if generated[i:i+len(boundary)] == boundary), None)


def bounded_thinking_filter(final_filter, input_length, end_token, vocabulary_size,
                            reasoning_budget, eos_ids=()):
    """Phase filter: None permits reasoning logits; lists constrain the final phase."""
    end_ids = [end_token] if isinstance(end_token, int) else list(end_token)
    if (reasoning_budget < 1 or not end_ids or any(
            not 0 <= token < vocabulary_size or token in eos_ids for token in end_ids)):
        raise ValueError("Invalid thinking boundary or budget")
    state = {"forced_thinking_end": False, "reasoning_token_budget": reasoning_budget,
             'boundary_token_ids': end_ids}

    def prefix(batch_id, input_ids):
        generated = input_ids[input_length:].tolist()
        boundary = boundary_position(generated, end_ids)
        prefix.in_thinking[batch_id] = boundary is None
        if boundary is not None:
            # Closing token is the enforcer's immutable root seed, not payload.
            return final_filter(batch_id, input_ids[input_length + boundary + len(end_ids)-1:]) if final_filter else None
        if len(generated) >= reasoning_budget:
            state["forced_thinking_end"] = True
            # Complete a partly generated boundary, otherwise start it at cutoff.
            if 'forced_boundary_start' not in state:
                partial = max((n for n in range(1, len(end_ids)) if generated[-n:] == end_ids[:n]), default=0)
                state['forced_boundary_start'] = len(generated)-partial
            offset = len(generated)-state['forced_boundary_start']
            return [end_ids[offset]]
        return None
    prefix.in_thinking = {}
    return prefix, state


def thinking_logits_processor(final_filter, input_length, end_token, vocabulary_size,
                              reasoning_budget, eos_ids=(), reasoning_presence_penalty=0, timing=None,
                              reasoning_repetition_penalty=1):
    """Avoid materializing a full-vocabulary allowlist on every reasoning token."""
    import torch
    from transformers import LogitsProcessorList
    penalty = validate_reasoning_presence_penalty(reasoning_presence_penalty)
    repetition = validate_reasoning_repetition_penalty(reasoning_repetition_penalty)
    prefix_boundary_ids = [end_token] if isinstance(end_token, int) else list(end_token)
    prefix, state = bounded_thinking_filter(final_filter, input_length, end_token,
                                            vocabulary_size, reasoning_budget, eos_ids)
    state.update(reasoning_presence_penalty=penalty,
                 presence_scope='Distinct generated reasoning IDs once; prompt and final excluded',
                 reasoning_repetition_penalty=repetition,
                 repetition_scope='Sign-aware multiplicative penalty once per distinct generated reasoning ID; prompt, boundary and final excluded')

    class ThinkingProcessor:
        def __call__(self, input_ids, scores):
            started = time.monotonic()
            for batch_id, row in enumerate(input_ids):
                generated = row[input_length:]
                # The prefix filter already transfers/scans these same IDs.
                # Reuse its phase result instead of synchronizing CUDA twice.
                allowed = prefix(batch_id, row)
                in_thinking = prefix.in_thinking[batch_id]
                if (penalty or repetition != 1) and in_thinking:
                    # Presence, not frequency: repeated occurrences of one ID
                    # receive exactly one subtraction at this generation step.
                    seen = torch.unique(generated)
                    if repetition != 1 and allowed is None:
                        repeated = seen[~torch.isin(seen, torch.tensor(prefix_boundary_ids, device=seen.device))]
                        values = scores[batch_id, repeated]
                        scores[batch_id, repeated] = torch.where(values < 0, values * repetition, values / repetition)
                    if penalty:
                        scores[batch_id, seen] -= penalty
                if allowed is None:
                    scores[batch_id, vocabulary_size:] = -float("inf")
                    if in_thinking:
                        scores[batch_id, list(eos_ids)] = -float("inf")
                else:
                    mask = torch.full_like(scores[batch_id], -float("inf"))
                    mask[allowed] = 0
                    scores[batch_id] += mask
            if timing is not None:
                timing['logits_control_seconds'] += time.monotonic() - started
            return scores
    return LogitsProcessorList([ThinkingProcessor()]), state


def payload_candidate(tokenizer, generated, final_selector=None, thinking_end_ids=None):
    """Do not decode tentative reasoning to look for an executable payload.

    With a verified provider token boundary, defer text decoding until that
    boundary exists. Once present, retain the original complete-decode selector
    semantics. No token budget, reasoning content or final parser is shortened.
    """
    if final_selector is not None and thinking_end_ids is not None:
        if not thinking_end_ids:raise ValueError('Empty thinking boundary')
        ids=generated.tolist() if hasattr(generated,'tolist') else list(generated)
        if boundary_position(ids,thinking_end_ids) is None:return None
        generated=ids
    text=tokenizer.decode(generated,skip_special_tokens=final_selector is None)
    return final_selector(text) if final_selector else text


def payload_stopping_criteria(tokenizer, input_length, parser, final_selector=None,
                              timing=None, on_tokens=None, thinking_end_ids=None):
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    class CompletePayload(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            started = time.monotonic()
            complete = []
            for row in input_ids:
                text = payload_candidate(tokenizer,row[input_length:],final_selector,thinking_end_ids)
                complete.append(text is not None and complete_payload(text, parser))
            result = torch.tensor(complete, dtype=torch.bool, device=input_ids.device)
            if timing is not None:
                timing['payload_control_seconds'] += time.monotonic() - started
            if on_tokens is not None:
                on_tokens(int(input_ids.shape[-1]) - input_length)
            return result

    return StoppingCriteriaList([CompletePayload()])
