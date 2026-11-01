"""Explicit evaluator clock ownership; opt-in, never choose a policy target."""
from copy import deepcopy
import math
from runtime_support import NativeJournalError, NativeDispatchError


SETTLING_CONTRACT = {
    'version': 'sampled-settling-v1', 'time_step': 0.02,
    'max_steps': 200, 'window_steps': 30,
    'position_tolerance_m': 0.001, 'rotation_tolerance_degrees': 0.1,
}


class ClockActionBoundaryError(RuntimeError):
    """Policy outcome is known, but its subsequent clock boundary is incomplete."""
    def __init__(self, boundary, stage, cause):
        self.boundary=deepcopy(boundary)
        self.stage=stage
        self.cause=cause
        super().__init__(str(cause))


def validate_settling_contract(value):
    if not isinstance(value, dict) or set(value) != set(SETTLING_CONTRACT):
        raise ValueError('Explicit complete settling contract required')
    if value['version'] != SETTLING_CONTRACT['version']:
        raise ValueError('Unsupported settling version')
    for key in ('max_steps', 'window_steps'):
        if type(value[key]) is not int or value[key] <= 0:
            raise ValueError('Positive integral step counts required')
    if value['window_steps'] > value['max_steps']:
        raise ValueError('Window exceeds bounded step budget')
    for key in ('time_step', 'position_tolerance_m', 'rotation_tolerance_degrees'):
        v = value[key]
        if type(v) not in (int, float) or not math.isfinite(v) or v <= 0:
            raise ValueError('Finite positive physical sampling parameters required')
    if value['time_step'] > 0.05:
        raise ValueError('Native timestep exceeds supported bound')
    return deepcopy(value)


def sampled_geometry(captures, contract):
    """All pairs/all objects, including cumulative excursion, not just final delta.

    Native moving flags are retained diagnostics; no speed/continuous-rest claim.
    """
    contract = validate_settling_contract(contract)
    missing, frames, moving = [], [], []
    if len(captures) < contract['window_steps'] + 1:
        missing.append('incomplete requested sampling window')
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
            vectors = []
            for field in ('position', 'rotation'):
                value = obj.get(field)
                vector = [value.get(a) for a in ('x', 'y', 'z')] if isinstance(value, dict) else []
                if len(vector) != 3 or any(type(v) not in (int, float) or not math.isfinite(v) for v in vector):
                    missing.append({'capture': index, 'object_id': identity, 'field': field})
                    vector = None
                vectors.append(vector)
            frame[identity] = vectors
            if type(obj.get('isMoving')) is not bool:
                missing.append({'capture': index, 'object_id': identity, 'field': 'isMoving'})
            elif obj['isMoving']:
                moving.append({'capture': index, 'object_id': identity})
        frames.append(frame)
    reference = frames[0] if frames else {}
    for index, frame in enumerate(frames[1:], 1):
        if frame.keys() != reference.keys():
            missing.append({'capture': index, 'field': 'unchanged_object_set'})
    max_position = max_rotation = 0.0
    offenders = []
    for identity in sorted(reference):
        position = rotation = 0.0
        for i, frame in enumerate(frames):
            for other in frames[i + 1:]:
                if identity not in frame or identity not in other:
                    continue
                a, b = frame[identity], other[identity]
                if a[0] is not None and b[0] is not None:
                    position = max(position, math.dist(a[0], b[0]))
                if a[1] is not None and b[1] is not None:
                    rotation = max(rotation, *(abs((x-y+180) % 360-180) for x, y in zip(a[1], b[1])))
        max_position = max(max_position, position)
        max_rotation = max(max_rotation, rotation)
        if position > contract['position_tolerance_m'] or rotation > contract['rotation_tolerance_degrees']:
            offenders.append({'object_id': identity, 'position_excursion_m': position,
                              'rotation_excursion_degrees': rotation})
    return {'status': 'unknown' if missing else 'not_ready' if offenders else 'sampled_ready',
            'missing': missing, 'offenders': offenders, 'moving_flag_observations': moving,
            'capture_count': len(captures), 'object_count': len(reference),
            'max_position_excursion_m': max_position, 'max_rotation_excursion_degrees': max_rotation,
            'scope': 'All-object sampled geometry only;not continuous stability,velocity,contact safety or task success'}


class OwnedPhysicsClock:
    """Fail-closed clock lifecycle. No automatic inverse action after unknown IO.

    Native action-end and later frozen endpoints remain distinct. Callers must
    qualify/integrate that distinction before using this in model feedback.
    """
    def __init__(self, env, record, *, environment_time=False):
        self.env, self.record = env, record
        self.state = 'unowned'
        self.controller_commands = 0
        self.policy_commands = 0
        self.environment_time = environment_time
        self.environment_time_state = None

    def _environment_pause(self,paused):
        after=self._step('physical_clock',{'action':'SetEnvironmentClockPaused','paused':paused})
        info=after.get('actionReturn')
        if (not isinstance(info,dict) or info.get('version')!='unity-scaled-environment-time-v1'
                or info.get('owned') is not True or info.get('paused') is not paused
                or info.get('timeScale')!=(0 if paused else 1)
                or (paused and info.get('physicsAutoSimulation') is not False)):
            raise ValueError('Unknown/refused environment time ownership;no replay')
        self.environment_time_state=deepcopy(info)
        self.state='paused' if paused else 'running'
        return after

    def _step(self, origin, command, allow_known_failure=False, on_received=None):
        if self.state == 'halted':
            raise RuntimeError('Clock halted;unknown/refused command must not be resent')
        if origin == 'physical_clock':
            self.controller_commands += 1
        else:
            self.policy_commands += 1
        before = deepcopy(self.env.snapshot())
        self.state = 'halted'  # IO or persistence failures leave no replayable state.
        self.env._owned_clock_dispatch = True
        try:
            try:
                event = self.env.native_step(origin, **command)
            except NativeJournalError as exc:
                # Use the IO owner's exact endpoint, never the mutable last_event
                # of a different clock/control command. No record or freeze retry.
                if (on_received is not None and exc.origin == origin and exc.command == command
                        and type(exc.endpoint.get('lastActionSuccess')) is bool):
                    on_received(exc.endpoint, exc.command_id)
                raise
        finally:
            self.env._owned_clock_dispatch = False
        after = deepcopy(event.metadata)
        row = {'origin': origin, 'command': deepcopy(command), 'command_id': self.env.last_command_id,
               'before': before, 'after': after}
        if on_received is not None and type(after.get('lastActionSuccess')) is bool:
            on_received(after,self.env.last_command_id)
        self.record(row)
        if type(after.get('lastActionSuccess')) is not bool:
            raise ValueError('Unknown native outcome;stop without further commands')
        if not after['lastActionSuccess'] and not allow_known_failure:
            raise ValueError('Native command refused;stop without retry or inverse cleanup')
        return after

    def acquire(self):
        if self.state != 'unowned':
            raise RuntimeError('Clock ownership can only be acquired once')
        after = self._step('physical_clock', {'action': 'PausePhysicsAutoSim'})
        self.state = 'paused'
        if self.environment_time:after=self._environment_pause(True)
        return after

    def advance(self, time_step):
        if self.state != 'paused':
            raise RuntimeError('Manual advance requires known paused ownership')
        if type(time_step) not in (int, float) or not math.isfinite(time_step) or not 0 < time_step <= 0.05:
            raise ValueError('Invalid timestep;no native command sent')
        if self.environment_time:self._environment_pause(False)
        after = self._step('physical_clock', {'action': 'AdvancePhysicsStep', 'timeStep': time_step,
                                            'simSeconds': time_step, 'allowAutoSimulation': False})
        self.state = 'paused'
        if self.environment_time:after=self._environment_pause(True)
        return after

    def settle(self, contract):
        contract = validate_settling_contract(contract)
        if self.state != 'paused':
            raise RuntimeError('Settling requires paused ownership')
        captures = [deepcopy(self.env.snapshot())]
        assessment = None
        for index in range(contract['max_steps']):
            captures.append(self.advance(contract['time_step']))
            if len(captures) < contract['window_steps'] + 1:
                continue
            assessment = sampled_geometry(captures[-contract['window_steps'] - 1:], contract)
            if assessment['status'] in ('sampled_ready', 'unknown'):
                break
        result = {'contract': contract, 'assessment': assessment, 'steps': index + 1,
                  'requested_nominal_sim_seconds': (index + 1) * contract['time_step'],
                  'actual_simulated_seconds': None,
                  'scope': 'Requested step duration is not independent clock calibration;all intermediate endpoints journaled'}
        self.state = 'halted'
        self.record({'settling_result': result})
        if assessment['status'] == 'sampled_ready':
            self.state = 'paused'
        if assessment['status'] != 'sampled_ready':
            self.state = 'halted'
        return result

    def _paused_control(self,command):
        if self.state != 'paused':
            raise RuntimeError('Wait check requires paused ownership')
        before = deepcopy(self.env.snapshot())
        after = self._step('physical_clock',command)
        # Import-free root implementation uses complete state projection below.
        comparison = paused_comparison(before, after)
        self.record({'wait_barrier': comparison})
        if comparison['status'] != 'unchanged':
            raise ValueError('Paused sampled endpoint changed/unknown;stop without replay')
        self.state = 'paused'
        return after,comparison

    def check_wait(self, *, appearance=False):
        after,comparison=self._paused_control({'action':'GetHeldObjectAppearance' if appearance else
                                             'GetEnvironmentClock' if self.environment_time else 'Pass'})
        if appearance:
            comparison={**comparison,'appearance':deepcopy(after.get('actionReturn'))}
        if self.environment_time:
            state=after
            if appearance:state,_=self._paused_control({'action':'GetEnvironmentClock'})
            info=state.get('actionReturn')
            if (not isinstance(info,dict) or info.get('version')!='unity-scaled-environment-time-v1'
                    or info.get('owned') is not True or info.get('paused') is not True
                    or info.get('timeScale')!=0 or info.get('physicsAutoSimulation') is not False):
                self.state='halted'
                raise ValueError('Environment clock resumed/unknown during wait;no replay')
            self.environment_time_state=deepcopy(info)
            comparison['environment_time']=deepcopy(info)
        return comparison

    def set_held_appearance(self,opacity):
        if type(opacity) not in (int,float) or not math.isfinite(opacity) or not 0<=opacity<=1:
            raise ValueError('Finite RGB opacity in[0,1] required')
        after,_=self._paused_control({'action':'SetHeldObjectOpacity','opacity':opacity})
        return deepcopy(after.get('actionReturn'))

    def action(self, command, origin='qualification_action', allow_known_failure=False):
        if self.state != 'paused' or origin == 'physical_clock':
            raise RuntimeError('Action requires known pause and independent provenance')
        if command.get('action') in {'PausePhysicsAutoSim', 'UnpausePhysicsAutoSim', 'AdvancePhysicsStep',
                                     'SetEnvironmentClockPaused','GetEnvironmentClock'}:
            raise ValueError('Clock actions cannot masquerade as policy actions')
        environment_resume_id=None
        try:
            if self.environment_time:
                self._environment_pause(False)
                environment_resume_id=self.env.last_command_id
            resumed = self._step('physical_clock', {'action': 'UnpausePhysicsAutoSim'})
        except BaseException as exc:
            self.state = 'halted'
            control = dict(exception_type=type(exc).__name__, error=str(exc),
                environment_resume_command_id=environment_resume_id,
                scope='Clock preparation failed before policy IO; control effects are separate')
            if isinstance(exc, NativeDispatchError):
                control['execution'] = deepcopy(exc.execution)
            elif isinstance(exc, NativeJournalError):
                control.update(origin=exc.origin, command=deepcopy(exc.command),
                    command_id=exc.command_id, endpoint=deepcopy(exc.endpoint),
                    end_record_durability='unconfirmed')
            raise NativeDispatchError(origin, command, 'not_sent', exc,
                stage='clock_prepare', control_failure=control) from exc
        resume_id = self.env.last_command_id
        self.state = 'running'
        result = {'resumed_endpoint':resumed,'resume_command_id':resume_id,
                  'native_action_endpoint':None,'action_command_id':None,
                  'frozen_endpoint':None,'freeze_command_id':None,
                  **({'clock_version':'owned-clock-runtime-v2','environment_resume_command_id':environment_resume_id,
                      'physics_freeze_command_id':None} if self.environment_time else {}),
                  'scope':'Action runs with automatic physics;later pause endpoint is separate,not overwritten action feedback'}
        def received(endpoint,identity):
            result.update(native_action_endpoint=deepcopy(endpoint),action_command_id=identity)
        stage='native_action_record'
        try:
            self._step(origin,command,allow_known_failure=allow_known_failure,on_received=received)
            self.state='running'
            stage='physics_freeze'
            frozen=self._step('physical_clock',{'action':'PausePhysicsAutoSim'})
            if self.environment_time:
                result['physics_freeze_command_id']=self.env.last_command_id
                self.state='paused'
                stage='environment_freeze'
                frozen=self._environment_pause(True)
            result.update(frozen_endpoint=frozen,freeze_command_id=self.env.last_command_id)
            stage='action_boundary_record'
            self.record({'action_boundary':result})
        except BaseException as exc:
            self.state='halted'
            if result['native_action_endpoint'] is not None:
                if stage == 'native_action_record' and isinstance(exc, NativeJournalError):
                    stage = 'native_action_journal'
                raise ClockActionBoundaryError(result,stage,exc) from exc
            if isinstance(exc, NativeDispatchError):
                exc.execution['native_dispatch_error']['clock_preparation'] = dict(
                    resume_command_id=resume_id, environment_resume_command_id=environment_resume_id,
                    resumed_endpoint=deepcopy(resumed), clock_state='halted',
                    scope='Control resume acknowledged separately from the policy dispatch status')
            raise
        self.state = 'paused'
        return result

    def description(self):
        return {'version': 'owned-clock-runtime-v2' if self.environment_time else 'owned-clock-runtime-v1', 'state': self.state,
                'controller_commands': self.controller_commands,
                'action_commands': self.policy_commands,
                'actual_simulated_seconds': None,
                **({'environment_time':deepcopy(self.environment_time_state)} if self.environment_time else {}),
                'scope': 'Sampled frozen boundaries;automatic physics during each action;not task success'}


def paused_comparison(before, after):
    """Compare all captured physical/state fields while excluding event diagnostics."""
    fields = ('position', 'rotation', 'isOpen', 'isToggled', 'isBroken', 'isDirty',
              'isPickedUp', 'isSliced', 'isCooked', 'isFilledWithLiquid', 'fillLiquid',
              'temperature', 'parentReceptacles', 'receptacleObjectIds')
    missing, changes = [], []
    frames = []
    for snapshot in (before, after):
        objects = snapshot.get('objects', [])
        frame = {obj['objectId']: obj for obj in objects}
        if not frame or len(frame) != len(objects):
            missing.append('complete unique object set')
        frames.append(frame)
    a, b = frames
    if a.keys() != b.keys():
        missing.append('unchanged object set')
    for identity in sorted(a.keys() & b.keys()):
        for field in fields:
            if field in ('position', 'rotation'):
                for obj in (a[identity], b[identity]):
                    v = obj.get(field)
                    if not isinstance(v, dict) or any(type(v.get(k)) not in (int, float) or not math.isfinite(v[k]) for k in ('x', 'y', 'z')):
                        missing.append(identity + ':' + field)
            if (field in a[identity]) != (field in b[identity]) or a[identity].get(field) != b[identity].get(field):
                changes.append({'object_id': identity, 'field': field})
    for field in ('agent', 'inventoryObjects'):
        if field not in before or field not in after:
            missing.append(field)
        elif before[field] != after[field]:
            changes.append({'field': field})
    return {'status': 'unknown' if missing else 'changed' if changes else 'unchanged',
            'missing': missing, 'changes': changes, 'scope': 'Sampled pause endpoints only'}


def validate_runtime_clock(value):
    version=value.get('version') if isinstance(value,dict) else None
    if version not in {'owned-clock-runtime-v1','owned-clock-runtime-v2'}:raise ValueError('Unknown owned clock version')
    expected = {'version': version, 'settling': SETTLING_CONTRACT}
    if value != expected:
        raise ValueError('Require complete fixed owned clock contract')
    validate_settling_contract(value['settling'])
    return deepcopy(value)


def assess_owned_survey(env, survey_audit, *, expected_version=None):
    """Independent evaluator gate;no native answers enter model requests."""
    clock = env.physics_description()
    settling = getattr(env, 'physics_settling', None)
    if not isinstance(clock, dict) or not isinstance(settling, dict):
        return {'status': 'refused', 'clock': clock, 'initialization': settling,
                'survey_geometry': {'status': 'unknown', 'missing': ['owned clock initialization unavailable']},
                'expected_clock_version': expected_version, 'clock_version_matches': False,
                'scope': 'No owned clock evidence; no legacy fallback or model initialization permitted'}
    baseline = getattr(env, '_physics_ready_snapshot', None)
    captures = ([survey_audit[0]['before']] if survey_audit else
                [deepcopy(baseline)] if baseline is not None else [])
    captures += [row.get('physics_boundary', {}).get('frozen_endpoint', {}) for row in survey_audit]
    captures.append(deepcopy(env.snapshot()))
    contract = {**SETTLING_CONTRACT, 'window_steps': max(1, len(captures)-1)}
    geometry = sampled_geometry(captures, contract)
    version_matches = expected_version is None or clock.get('version') == expected_version
    ready = (version_matches and clock['state'] == 'paused' and settling['assessment']['status'] == 'sampled_ready'
             and geometry['status'] == 'sampled_ready')
    return {'status': 'sampled_ready' if ready else 'refused', 'clock': clock,
            'initialization': settling, 'survey_geometry': geometry,
            'survey_geometry_source': ('recorded_survey_action_boundaries' if survey_audit else
                                      'post_settle_baseline_and_current_paused_endpoint'),
            'post_settle_baseline_command_id': getattr(env, '_physics_ready_command_id', None),
            'expected_clock_version': expected_version, 'clock_version_matches': version_matches,
            'scope': 'Initialization and recorded survey sampled geometry;not readiness-v1/native rest/task success'}
