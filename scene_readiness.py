"""Evaluator-only sampled scene stability; never infer a task or repair a scene."""
from copy import deepcopy
import math


DEFAULT_READINESS_CONTRACT = {
    'version': 'scene-readiness-v1',
    'position_tolerance_m': 0.01,
    'rotation_tolerance_degrees': 1.0,
}


def validate_readiness_contract(contract):
    if not isinstance(contract, dict) or set(contract) != set(DEFAULT_READINESS_CONTRACT):
        raise ValueError('Explicit scene readiness contract has unknown or missing fields')
    if contract['version'] != DEFAULT_READINESS_CONTRACT['version']:
        raise ValueError('Unsupported scene readiness version')
    for key in ('position_tolerance_m', 'rotation_tolerance_degrees'):
        value = contract[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('Readiness tolerances must be finite nonnegative numbers')
    return deepcopy(contract)


def _vector(value):
    if not isinstance(value, dict):
        return None
    vector = [value.get(axis) for axis in ('x', 'y', 'z')]
    return vector if all(type(v) in (int, float) and math.isfinite(v) for v in vector) else None


def assess_scene_readiness(captures, contract):
    """Check every recorded object across the existing fixed survey window.

    No extra native events, model inputs, object whitelist, retries or corrections.
    Missing evidence is unknown. Ready covers sampled endpoints only, not later
    physics, continuous contacts, causal attribution or autonomous success.
    """
    contract = validate_readiness_contract(contract)
    missing, violations, frames = [], [], []
    if len(captures) < 2:
        missing.append({'field': 'survey_window', 'reason': 'At least two recorded endpoints required'})
    for index, capture in enumerate(captures):
        objects = capture.get('objects') if isinstance(capture, dict) else None
        frame = {}
        if not isinstance(objects, list) or not objects:
            missing.append({'capture': index, 'field': 'objects'})
            frames.append(frame)
            continue
        for obj in objects:
            identity = obj.get('objectId') if isinstance(obj, dict) else None
            if not isinstance(identity, str) or not identity or identity in frame:
                missing.append({'capture': index, 'field': 'unique_objectId'})
                continue
            position, rotation = _vector(obj.get('position')), _vector(obj.get('rotation'))
            moving = obj.get('isMoving')
            for key, value in (('position', position), ('rotation', rotation)):
                if value is None:
                    missing.append({'capture': index, 'object_id': identity, 'field': key})
            if type(moving) is not bool:
                missing.append({'capture': index, 'object_id': identity, 'field': 'isMoving'})
            elif moving:
                violations.append({'capture': index, 'object_id': identity, 'field': 'isMoving', 'actual': True})
            frame[identity] = (position, rotation)
        frames.append(frame)
    reference = frames[0] if frames else {}
    max_position = max_rotation = 0.0
    for index, frame in enumerate(frames[1:], 1):
        if frame.keys() != reference.keys():
            missing.append({'capture': index, 'field': 'object_set',
                            'removed': sorted(reference.keys() - frame.keys()),
                            'added': sorted(frame.keys() - reference.keys())})
        for identity in reference.keys() & frame.keys():
            before, after = reference[identity], frame[identity]
            if before[0] is not None and after[0] is not None:
                delta = math.dist(before[0], after[0])
                max_position = max(max_position, delta)
                if delta > contract['position_tolerance_m']:
                    violations.append({'capture': index, 'object_id': identity, 'field': 'position', 'delta_m': delta})
            if before[1] is not None and after[1] is not None:
                delta = max(abs((b-a+180) % 360-180) for a, b in zip(before[1], after[1]))
                max_rotation = max(max_rotation, delta)
                if delta > contract['rotation_tolerance_degrees']:
                    violations.append({'capture': index, 'object_id': identity, 'field': 'rotation', 'delta_degrees': delta})
    status = 'not_ready' if violations else 'unknown' if missing else 'ready'
    return {'status': status, 'contract': contract, 'capture_count': len(captures),
            'reference_object_count': len(reference), 'violations': violations, 'missing': missing,
            'max_position_excursion_m': max_position, 'max_rotation_excursion_degrees': max_rotation,
            'scope': 'Evaluator-only all-object sampled survey endpoints. No issue/target/goal or causal '
                     'claim; ready does not guarantee future stability, RGB sufficiency or research success.'}


def readiness_from_survey(survey_audit, snapshot, contract):
    if contract is None:
        return {'status': 'not_requested', 'scope': 'Legacy configuration checks only; physical readiness unassessed'}
    captures = []
    if survey_audit:
        captures.append(survey_audit[0].get('before'))
        captures.extend(row.get('after') for row in survey_audit)
    # Check the final snapshot matches the same window; not an additional event.
    captures.append(snapshot)
    result = assess_scene_readiness(captures, contract)
    result['scope'] += ' Final snapshot is a repeated endpoint check, not a new native observation.'
    return result
