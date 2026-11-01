"""AI2-THOR access, simulator-assisted observations and complete action capture."""
import math
from copy import deepcopy
import numpy as np
from action_protocol import ALLOWED_ACTIONS, ProtocolError, validate_action, HAND_MOVE_MAX_ATTEMPTS
from evidence_io import CommandJournal
from runtime_support import native_dispatch_feedback, NATIVE_DISPATCH_VERSION, ReceivedExecutionError, NativeJournalError, NativeDispatchError


class ThorWrapper:
    ALLOWED_ACTIONS = ALLOWED_ACTIONS
    supports_command_origin = True
    native_dispatch_contract = NATIVE_DISPATCH_VERSION

    def __init__(self, scene="FloorPlan1", grid_size=0.25, visibility_distance=1.5,
                 render_depth=False, render_instance_seg=False, fov=90,
                 width=600, height=600, platform=None, editor_port=None,
                 local_executable_path=None, command_journal=None):
        self.command_journal = None
        self.last_command_id = None
        self.expected_frame_shape = (height, width, 3)
        self.set_command_journal(command_journal)
        identity = None
        response_logged = False
        try:
            if self.command_journal:
                identity = self.command_journal.begin("initialization", {"operation": "Controller.__init__",
                    "scene": scene, "width": width, "height": height, "grid_size": grid_size,
                    "visibility_distance": visibility_distance, "fov": fov, "platform": platform,
                    "editor_port": editor_port, "local_executable_path": local_executable_path,
                    "render_depth": render_depth, "render_instance_seg": render_instance_seg})
            self.last_command_id = identity
            from ai2thor.controller import Controller
            options = dict(scene=scene, gridSize=grid_size, visibilityDistance=visibility_distance,
                           renderDepthImage=render_depth, renderInstanceSegmentation=render_instance_seg,
                           fieldOfView=fov, width=width, height=height,
                           maxUpwardLookAngle=30, maxDownwardLookAngle=60)
            self.camera_contract = {"min_horizon": -30, "max_horizon": 60, "degree_increment": 0.1}
            if platform == "cloud":
                from ai2thor.platform import CloudRendering
                options["platform"] = CloudRendering
            if editor_port is not None:
                options.update(start_unity=False, port=editor_port, host="127.0.0.1")
            if local_executable_path:
                options["local_executable_path"] = local_executable_path
            self.controller = Controller(**options)
            self.last_event = self.controller.last_event
            if self.command_journal:
                self.command_journal.end(identity, self.last_event.metadata)
            response_logged = True
            self.id_to_alias = {}
            self.alias_to_id = {}
            self.audit = []
            self.affordance_rejections = []
            self.hand_motion_attempts = 0
            self.hand_motion_halted = False
            self._frame_buffer = []
            self.frame_info = []
            self._capture()
        except BaseException as exc:
            try:
                if self.command_journal and identity is not None and not response_logged:
                    self.command_journal.error(identity, exc)
            finally:
                try:
                    if hasattr(self, "controller"):
                        self.controller.stop()
                finally:
                    if self.command_journal:
                        self.command_journal.close()
            raise

    def set_command_journal(self, path):
        current = getattr(self, "command_journal", None)
        if current is not None:
            current.close()
        self.command_journal = CommandJournal(path) if path is not None else None

    def start_physics_clock(self, record, contract, *, environment_time=False):
        """Acquire after evaluator setup;discard pre-settle RGB before baseline/survey."""
        from physics_time import OwnedPhysicsClock
        if getattr(self, 'physics_clock', None) is not None:
            raise RuntimeError('Clock already owned;no reacquisition or retry')
        if self.snapshot().get('lastActionSuccess') is not True:
            raise ValueError('Initialization/setup not successful;no clock acquisition')
        self.physics_clock = OwnedPhysicsClock(self, record,environment_time=environment_time)
        self.physics_clock.acquire()
        self.physics_settling = self.physics_clock.settle(contract)
        if self.physics_clock.state != 'paused':
            raise ValueError('Initialization sampled settling refused;no model or scene repair')
        self._frame_buffer, self.frame_info = [], []
        self._capture()
        # Evaluator-only baseline for a zero-motion survey. Never part of model input.
        self._physics_ready_snapshot = deepcopy(self.snapshot())
        self._physics_ready_command_id = self.last_command_id
        return deepcopy(self.physics_settling)

    def check_physics_wait(self):
        clock = getattr(self, 'physics_clock', None)
        return clock.check_wait() if clock is not None else None

    def physics_description(self):
        clock = getattr(self, 'physics_clock', None)
        return clock.description() if clock is not None else None

    def native_step(self, origin, **command):
        if getattr(self, '_native_journal_halted', False):
            raise RuntimeError('Native journal failed; no further commands or replay')
        if getattr(self, 'physics_clock', None) is not None and not getattr(self, '_owned_clock_dispatch', False):
            raise RuntimeError('Owned clock forbids direct native IO;no bypass or automatic retry')
        journal = getattr(self, "command_journal", None)
        try:
            identity = journal.begin(origin, command, self.last_event.metadata) if journal else None
        except BaseException as exc:
            self._native_journal_halted = True
            raise NativeDispatchError(origin, command, 'not_sent', exc, stage='journal_begin') from exc
        self.last_command_id = identity
        try:
            event = self.controller.step(**command)
        except BaseException as exc:
            self._native_journal_halted = True
            retention_error = None
            if journal:
                try:
                    journal.error(identity, exc)
                except BaseException as secondary:
                    retention_error = secondary
            raise NativeDispatchError(origin, command, 'outcome_unknown', exc, identity,
                stage='controller_step', retention_error=retention_error) from exc
        self.last_event = event  # Preserve received state even if the end-record write fails.
        if journal:
            try:
                journal.end(identity, event.metadata)
            except BaseException as exc:
                self._native_journal_halted = True
                # Do not append command_error: that record means unknown native IO.
                # end may have partially written; do not retry it either.
                raise NativeJournalError(origin, command, identity, event.metadata, exc) from exc
        return event

    def _alias(self, object_id):
        if object_id not in self.id_to_alias:
            self._alias_sequence = max(getattr(self,'_alias_sequence',0),len(self.id_to_alias))+1
            alias = f"obj_{self._alias_sequence:03d}"
            self.id_to_alias[object_id] = alias
            self.alias_to_id[alias] = object_id
        return self.id_to_alias[object_id]

    def _capture(self):
        from datetime import datetime, timezone
        import time
        frame = self.last_event.frame
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise RuntimeError("Unity did not deliver native uint8 RGB")
        expected = getattr(self, "expected_frame_shape", None)
        self.render_contract = {"native_width": frame.shape[1], "native_height": frame.shape[0],
                                "requested_shape": expected, "matching": expected is None or frame.shape == expected}
        if expected is not None and frame.shape != expected:
            raise RuntimeError(f"Unity RGB shape {frame.shape} differs from requested {expected}; configure the actual player viewport before inference")
        self._frame_buffer.append(frame.copy())
        agent = self.last_event.metadata.get("agent", {})
        self._observation_sequence = getattr(self, "_observation_sequence", 0) + 1
        self.frame_info.append({"frame": len(self._frame_buffer)-1,
                                "observation_id": f"obs-{self._observation_sequence:06d}",
                                "captured_utc": datetime.now(timezone.utc).isoformat(),
                                "monotonic_seconds": time.monotonic(),
                                "position": deepcopy(agent.get("position")),
                                "rotation": deepcopy(agent.get("rotation")),
                                "yaw": agent.get("rotation", {}).get("y"),
                                "horizon": agent.get("cameraHorizon"),
                                "action": self.last_event.metadata.get("lastAction")})
        clock = getattr(self, 'physics_clock', None)
        if clock is not None:
            if clock.state != 'paused':
                raise RuntimeError('RGB capture requires a known frozen clock boundary')
            self.frame_info[-1]['physical_time'] = {
                'version': clock.description()['version'], 'boundary': 'frozen_endpoint',
                'endpoint_command_id': self.last_command_id,
                'action_command_id': getattr(self, 'last_action_command_id', None),
                'actual_simulated_seconds': None,
                **({'waiting_time_scope':'Physics and scaled Unity timers paused during model waits;actions resume both. Sampled endpoints,not exact simulation-time calibration.'}
                   if clock.environment_time else {})}

    def get_discovery_summary(self):
        """Neutral current identity/geometry for all visible types; no native state."""
        rows = []
        masks = getattr(self.last_event, "instance_masks", {})
        for obj in self.snapshot().get("objects", []):
            if not obj.get("visible"):
                continue
            mask = masks.get(obj["objectId"])
            yy, xx = np.nonzero(mask) if mask is not None else ([], [])
            rows.append({"alias": self._alias(obj["objectId"]), "type": obj.get("objectType"),
                         "bbox_xyxy": [int(min(xx)), int(min(yy)), int(max(xx)) + 1, int(max(yy)) + 1] if len(xx) else None,
                         "mask_pixels": len(xx)})
        rows.sort(key=lambda row: (-row["mask_pixels"], row["alias"]))
        inventory = self.snapshot().get("inventoryObjects", [])
        return {"objects": rows[:12], "omitted_visible_count": max(0, len(rows)-12),
                "selection": "largest current visible masks across all types; no problem priority", "heldObject":
                self._alias(inventory[0]["objectId"]) if inventory else None,
                "information_source": "simulator visibility/identity/2D masks and physical inventory only; no Boolean state"}

    def get_scene_state(self):
        meta = self.last_event.metadata
        inventory = meta.get("inventoryObjects", [])
        held_id = inventory[0].get("objectId") if inventory else None
        objects = []
        for obj in meta.get("objects", []):
            if not obj.get("visible"):
                continue
            properties = {}
            capabilities = {"pickupable":"isPickedUp", "openable":"isOpen", "toggleable":"isToggled",
                            "sliceable":"isSliced", "cookable":"isCooked", "dirtyable":"isDirty",
                            "breakable":"isBroken", "canFillWithLiquid":"isFilledWithLiquid", "receptacle":None}
            for capability, state in capabilities.items():
                if obj.get(capability):
                    properties[capability] = True
                    if state in obj:
                        properties[state] = obj[state]
            if "isInteractable" in obj:
                properties["isInteractable"] = obj["isInteractable"]
            if not any(properties.get(key) for key in capabilities):
                continue
            item = {"objectId": self._alias(obj["objectId"]), "objectType": obj["objectType"],
                    "distance": round(obj.get("distance", 0), 2), **properties}
            masks = getattr(self.last_event, "instance_masks", None)
            depth = getattr(self.last_event, "depth_frame", None)
            if masks is not None and depth is not None and obj["objectId"] in masks:
                values = depth[masks[obj["objectId"]]]
                values = values[np.isfinite(values) & (values > 0)]
                if values.size:
                    item["depth_median_m"] = round(float(np.median(values)), 3)
            objects.append(item)
        state = {"heldObject": self._alias(held_id) if held_id else None, "objects": objects}
        state['handMotion']={'remaining_attempts':0 if getattr(self,'hand_motion_halted',False) else max(0,HAND_MOVE_MAX_ATTEMPTS-getattr(self,'hand_motion_attempts',0)),
                             'max_attempts_per_episode':HAND_MOVE_MAX_ATTEMPTS}
        if held_id:
            held = next((obj for obj in meta.get("objects", []) if obj["objectId"] == held_id), inventory[0])
            state["heldObjectDetails"] = {"objectType": held.get("objectType"), "physicallyHeld": True}
            visible = {obj["objectId"]: obj for obj in meta.get("objects", []) if obj.get("visible")}
            feedback = []
            for row in getattr(self, "affordance_rejections", []):
                target = visible.get(row["target_id"])
                if (row["held_id"] == held_id and row["held_type"] == held.get("objectType")
                        and target is not None and target.get("objectType") == row["target_type"]):
                    feedback.append({"action": "PutObject", "heldAlias": self._alias(held_id),
                        "targetAlias": self._alias(row["target_id"]), "category": "incompatible_target",
                        "source": "observed_native_rejection", "scope": "same held object and receiver types; not a spatial failure"})
            if feedback:
                state["affordanceFeedback"] = feedback
        horizon = meta.get("agent", {}).get("cameraHorizon")
        if horizon is not None and hasattr(self, "camera_contract"):
            state["camera"] = {"horizon": round(float(horizon), 1), **self.camera_contract}
        return state

    def get_visible_objects_summary(self, frame_window=8):
        import json
        recent = self.frame_info[-frame_window:]
        return json.dumps({**self.get_scene_state(), "recentFrames": recent,
                           "informationSource": "simulator visibility, inventory and optional depth; not RGB-only"})

    def snapshot(self):
        return self.last_event.metadata

    def coverage_reachable_positions(self):
        """Privileged planning query, journaled and charged to episode action budget."""
        before = self.snapshot()
        event = self.native_step("coverage_map", action="GetReachablePositions")
        self._capture()
        success = bool(event.metadata["lastActionSuccess"])
        self.audit.append({"action": {"action": "GetReachablePositions"}, "success": success,
                           "before": before, "after": event.metadata, "origin": "coverage_map",
                           "command_id": self.last_command_id, "frame": len(self._frame_buffer)-1})
        return {"success": success, "positions": event.metadata.get("actionReturn", []) if success else [],
                "command_id": self.last_command_id, "source": "native_reachable_positions"}

    def sanitize_error(self, error):
        for object_id, alias in self.id_to_alias.items():
            error = error.replace(object_id, alias)
        return error

    def execute_action(self, action_dict, origin="policy"):
        if isinstance(action_dict,dict) and 'selection_report' in action_dict:
            return {'success':False,'category':'precondition_not_met',
                    'error':'Controller selection metadata must be consumed before native dispatch',
                    **native_dispatch_feedback('not_sent')}
        try:
            action = validate_action(action_dict)
        except ProtocolError as exc:
            return {"success": False, "error": str(exc), "message": str(exc), **native_dispatch_feedback('not_sent')}
        if action["action"] == "Done":
            return {"success": False, "error": "Done requires the independent completion verifier.", "message": "Done request", **native_dispatch_feedback('not_sent')}
        if action['action'] == 'RequestReplan':
            return {'success': False, 'category': 'model_replan_request', 'error': action['reason'],
                    **native_dispatch_feedback('not_sent'),
                    'scope': 'Model-authored planning feedback only; never a Unity command or proof of impossibility'}
        if action['action']=='MoveHeldObject':
            if (len(self.snapshot().get('inventoryObjects',[]))!=1 or getattr(self,'hand_motion_attempts',0)>=HAND_MOVE_MAX_ATTEMPTS
                    or getattr(self,'hand_motion_halted',False)):
                return {'success':False,'category':'precondition_not_met',**native_dispatch_feedback('not_sent'),
                        'error':'Hand translation requires one held object and a remaining attempt'}
            # Charge attempts before native IO; a failed/unknown move is not free.
            self.hand_motion_attempts=getattr(self,'hand_motion_attempts',0)+1
        if "objectId" in action:
            target = action["objectId"]
            if target in self.alias_to_id:
                action["objectId"] = self.alias_to_id[target]
            else:
                return {"success": False, "error": "Unknown candidate alias; inspect before acting.", "message": "Unknown target", **native_dispatch_feedback('not_sent')}
        before = self.last_event.metadata
        clock = getattr(self, 'physics_clock', None)
        boundary = None
        if clock is None:
            try:
                event = self.native_step(origin, **action)
            except NativeJournalError as exc:
                endpoint = exc.endpoint
                if type(endpoint.get('lastActionSuccess')) is not bool:
                    raise
                self.last_action_command_id = exc.command_id
                success = endpoint['lastActionSuccess']
                received = dict(success=success, error=endpoint.get('errorMessage','') if not success else '',
                    category='physical_success' if success else 'physical_failure',
                    executed_action=deepcopy(exc.command), command_id=exc.command_id,
                    **native_dispatch_feedback('outcome_received', exc.command_id))
                raise ReceivedExecutionError(received, 'native_journal_record', exc) from exc
            action_id = self.last_command_id
        else:
            from physics_time import ClockActionBoundaryError
            try:
                boundary = clock.action(action, origin, allow_known_failure=True)
            except ClockActionBoundaryError as exc:
                boundary=exc.boundary
                endpoint=boundary['native_action_endpoint']
                action_id=boundary['action_command_id']
                self.last_action_command_id=action_id
                success=endpoint['lastActionSuccess']
                raw_error=endpoint.get('errorMessage','') if not success else ''
                received=dict(success=success,error=raw_error,
                    category='physical_success' if success else 'physical_failure',
                    executed_action=deepcopy(action),command_id=action_id,
                    **native_dispatch_feedback('outcome_received',action_id),
                    physical_time=dict(version=clock.description()['version'],
                        native_action_command_id=action_id,
                        frozen_endpoint_command_id=boundary['freeze_command_id'],
                        observation_boundary='unavailable',result_boundary='native_action_endpoint',
                        clock_state='halted',failure_stage=exc.stage,
                        scope='Known policy outcome;clock boundary failed;no further commands or replay'))
                try:
                    self.audit.append(dict(action=deepcopy(action),success=success,error=raw_error,
                        category=received['category'],command_id=action_id,before=before,
                        after=deepcopy(endpoint),frame=None,physics_boundary=boundary,
                        post_execution_observation_error=str(exc)))
                except BaseException as audit_error:
                    received['audit_retention_error'] = str(audit_error)
                failure_stage = 'native_journal_record' if exc.stage == 'native_action_journal' else 'physical_clock_boundary'
                raise ReceivedExecutionError(received,failure_stage,exc.cause) from exc
            action_id = boundary['action_command_id']
            # last_event remains the frozen RGB/state. Native feedback is independent.
            event = type('ActionEndpoint', (), {'metadata': boundary['native_action_endpoint']})()
        self.last_action_command_id = action_id
        # Capture the IO-owner result before aliases, observations or audits can fail.
        success = event.metadata.get('lastActionSuccess')
        if type(success) is not bool:
            raise ValueError('Native endpoint lacks a Boolean outcome; do not infer success')
        raw_error = event.metadata.get('errorMessage', '') if not success else ''
        received = dict(success=success, error=raw_error,
            category='physical_success' if success else 'physical_failure',
            executed_action=deepcopy(action), command_id=action_id,
            **native_dispatch_feedback('outcome_received', action_id))
        if boundary is not None:
            received['physical_time'] = dict(native_action_command_id=action_id,
                frozen_endpoint_command_id=boundary['freeze_command_id'],
                observation_boundary='frozen_endpoint', result_boundary='native_action_endpoint')
        stage = 'native_observation_capture'
        try:
            self._capture()
            stage = 'native_result_projection'
            result, audit = self._project_action_result(action, before, event, action_id, boundary)
            stage = 'native_audit_record'
            self.audit.append(audit)
            return result
        except BaseException as exc:
            # Never retry a failed audit append. The command journal remains the
            # native original; the received error crosses runner/task/operation.
            if stage != 'native_audit_record':
                try:
                    self.audit.append(dict(action=deepcopy(action), success=success, error=raw_error,
                        category=received['category'], command_id=action_id, before=before,
                        after=deepcopy(event.metadata), frame=None,
                        post_execution_processing_error=str(exc),
                        **({'post_execution_observation_error':str(exc)} if stage=='native_observation_capture' else {}),
                        **({'physics_boundary':boundary} if boundary is not None else {})))
                except BaseException as audit_error:
                    received['audit_retention_error'] = str(audit_error)
            raise ReceivedExecutionError(received, stage, exc) from exc

    def _project_action_result(self, action, before, event, action_id, boundary):
        """Project a known endpoint without dispatching any additional native IO."""
        success = bool(event.metadata["lastActionSuccess"])
        raw_error = event.metadata.get("errorMessage", "") if not success else ""
        error = self.sanitize_error(raw_error)
        effect = None
        if action["action"] == "PutObject" and before.get("inventoryObjects"):
            objects = {obj["objectId"]: obj for obj in before.get("objects", [])}
            held_id = before["inventoryObjects"][0]["objectId"]
            effect = {"heldBefore": {"alias": self._alias(held_id), "objectType": objects.get(held_id, {}).get("objectType")},
                      "receiver": {"alias": self._alias(action["objectId"]), "objectType": objects.get(action["objectId"], {}).get("objectType")},
                      "inventoryEmptyAfter": not event.metadata.get("inventoryObjects", [])}
        category = "physical_success" if success else "physical_failure"
        constraint = None
        hand_audit = None
        if action['action']=='MoveHeldObject':
            from runtime_support import verify_hand_motion
            hand_audit=verify_hand_motion(before,self.snapshot(),before['inventoryObjects'][0]['objectId'])
            if boundary is not None:
                native_guard=verify_hand_motion(before,event.metadata,before['inventoryObjects'][0]['objectId'])
                hand_audit['native_action_endpoint_audit']=native_guard
                if native_guard['status']=='violation' or (native_guard['status']=='unknown' and hand_audit['status']=='complete'):
                    hand_audit['status']=native_guard['status']
            effect={'hand_motion_attempts_used':self.hand_motion_attempts,
                    'hand_motion_preservation':hand_audit['status'],
                    'scope':hand_audit['scope']}
            if hand_audit['status']!='complete':
                self.hand_motion_halted=True
                success=False
                category='preservation_violation' if hand_audit['status']=='violation' else 'preservation_unknown'
                error='Hand-motion endpoint preservation '+hand_audit['status']+'; stop without automatic retry'
                constraint={'category':category,'scope':hand_audit['scope']}
        lower = raw_error.lower()
        if not success and action["action"] == "PutObject":
            if "cannot be placed in" in lower or "not a valid object type to be placed in" in lower:
                category = "incompatible_target"
                inventory = before.get("inventoryObjects", [])
                objects = {obj["objectId"]: obj for obj in before.get("objects", [])}
                if inventory and action["objectId"] in objects:
                    held_id = inventory[0]["objectId"]
                    held = objects.get(held_id, inventory[0])
                    target = objects[action["objectId"]]
                    rejection = {"held_id": held_id, "held_type": held.get("objectType"),
                                 "target_id": target["objectId"], "target_type": target.get("objectType")}
                    if not hasattr(self, "affordance_rejections"):
                        self.affordance_rejections = []
                    if rejection not in self.affordance_rejections:
                        self.affordance_rejections.append(rejection)
                    constraint = {"action": "PutObject", "heldAlias": self._alias(held_id),
                                  "targetAlias": self._alias(target["objectId"]), "category": category,
                                  "source": "observed_native_rejection"}
                    error = (f"Receiver {constraint['targetAlias']} ({target.get('objectType')}) cannot accept the held "
                             f"{held.get('objectType')}. No placement occurred. Choose a different receiver; turning does not change type compatibility.")
            elif "receptacle is closed" in lower:
                category = "target_closed"
            elif "is full right now" in lower:
                category = "target_full"
        if not success and category == "physical_failure" and any(word in lower for word in ("collision", "collid", "blocked")):
            category = "collision"
        if not success:
            if 'fpscontroller' in lower and action['action'] in {'OpenObject', 'CloseObject'}:
                error = ('The moving door collided with the robot\'s own collision body '
                         '(native name: FPSController); the requested opening/closing did not complete. '
                         'A successful robot movement does not by itself verify clearance for this door. '
                         'Native report: ' + raw_error)
            elif "open/close" in raw_error.lower() and action["action"] in {"OpenObject", "CloseObject"}:
                error += " Door movement may be blocked; step back or choose a clear position while keeping it visible."
            elif action["action"] == "PutObject" and category == "physical_failure":
                error += " Target the receiving container; open it if closed and check reachability/compatibility."
            elif category in {"physical_failure", "collision"}:
                error += " Check visibility, distance and the action's required object state."
        audit = {"action": action, "success": success, "error": raw_error, "category": category,
                           "command_id": action_id,
                           "before": before, "after": event.metadata,
                           **({'physics_boundary': boundary} if boundary is not None else {}),
                           "frame": len(self._frame_buffer)-1,
                           **({'hand_motion_audit':hand_audit} if hand_audit is not None else {})}
        return {"success": success, "error": error, "category": category,
                **native_dispatch_feedback('outcome_received', action_id),
                "message": f"{action['action']}: {'success' if success else error}",
                "executed_action": action, "command_id": action_id,
                **({'physical_time': {'version': self.physics_clock.description()['version'],
                     'native_action_command_id': action_id,
                     'frozen_endpoint_command_id': boundary['freeze_command_id'],
                     'observation_boundary': 'frozen_endpoint',
                     'result_boundary': 'native_action_endpoint',
                     'scope': 'Known native outcome;later frozen RGB/inventory;not goal completion'}} if boundary is not None else {}),
                **({"constraint": constraint} if constraint else {}),
                **({"effect": effect} if effect else {})}, audit

    def get_recent_frames(self, n=8):
        if n < 1:
            raise ValueError("Frame window must be positive")
        return self._frame_buffer[-n:]

    def get_all_frames(self):
        return self._frame_buffer

    def _seed_frame_buffer(self):
        self._frame_buffer = []
        self.frame_info = []
        if getattr(self, 'physics_clock', None) is None:
            self.last_event = self.native_step("observation", action="Pass")
        else:
            self.check_physics_wait()
        self._capture()

    def reset(self, scene="FloorPlan1"):
        clock = getattr(self, 'physics_clock', None)
        if clock is not None:
            if clock.state != 'paused':
                raise RuntimeError('Unknown/halted clock cannot reset automatically')
            clock.state = 'halted'  # Reset IO may fail; never leave old ownership usable.
        journal = getattr(self, "command_journal", None)
        identity = journal.begin("reset", {"operation": "Controller.reset", "scene": scene}, self.snapshot()) if journal else None
        self.last_command_id = identity
        try:
            self.last_event = self.controller.reset(scene)
        except BaseException as exc:
            if journal:
                journal.error(identity, exc)
            raise
        if journal:
            journal.end(identity, self.last_event.metadata)
        self.physics_clock = None
        self._physics_ready_snapshot = None
        self._physics_ready_command_id = None
        self.last_action_command_id = None
        self.id_to_alias.clear()
        self.alias_to_id.clear()
        self._alias_sequence = 0
        self.audit.clear()
        self.affordance_rejections = []
        self.hand_motion_attempts = 0
        self.hand_motion_halted = False
        self._frame_buffer = []
        self.frame_info = []
        self._capture()

    def setup_problem(self, scenario):
        """Evaluator-controlled initial setup, never a model-selected action."""
        self.setup_audit = []

        def setup_step(**action):
            event = self.native_step("setup", **action)
            self.last_event = event
            entry = {"action": action, "success": bool(event.metadata["lastActionSuccess"]),
                     "error": event.metadata.get("errorMessage", "")}
            if action["action"] != "GetReachablePositions":
                entry["after"] = event.metadata
            self.setup_audit.append(entry)
            return event
        object_type, field = {"faucet-off": ("Faucet", "isToggled"),
                              "fridge-closed": ("Fridge", "isOpen")}[scenario]
        target = next((obj for obj in self.snapshot()["objects"] if obj["objectType"] == object_type), None)
        if target is None:
            raise ValueError(f"Scene has no {object_type} for {scenario}")
        action = "ToggleObjectOn" if field == "isToggled" else "OpenObject"
        setup_event = setup_step(action=action, objectId=target["objectId"], forceAction=True)
        if not setup_event.metadata["lastActionSuccess"] and scenario == "fridge-closed":
            # Some scene spawn poses obstruct opening the door before policy begins.
            # Move the fixture agent away, retry once, then establish the usual view.
            preflight = setup_step(action="GetReachablePositions")
            candidates = preflight.metadata.get("actionReturn")
            if preflight.metadata["lastActionSuccess"] and candidates:
                position = target["position"]
                far = max(candidates, key=lambda p: (p["x"]-position["x"])**2 + (p["z"]-position["z"])**2)
                moved = setup_step(action="TeleportFull", position=far,
                                   rotation={"x": 0, "y": 0, "z": 0}, horizon=15,
                                   standing=True, forceAction=True)
                if moved.metadata["lastActionSuccess"]:
                    setup_event = setup_step(action=action, objectId=target["objectId"], forceAction=True)
        if not setup_event.metadata["lastActionSuccess"]:
            raise RuntimeError(setup_event.metadata.get("errorMessage"))
        reachable = setup_step(action="GetReachablePositions")
        points = reachable.metadata["actionReturn"]
        position = target["position"]
        point = min(points, key=lambda p: (p["x"]-position["x"])**2 + (p["z"]-position["z"])**2)
        yaw = math.degrees(math.atan2(position["x"]-point["x"], position["z"]-point["z"])) % 360
        event = setup_step(action="TeleportFull", position=point,
                                     rotation={"x": 0, "y": yaw, "z": 0}, horizon=15,
                                     standing=True, forceAction=True)
        if not event.metadata["lastActionSuccess"]:
            raise RuntimeError(event.metadata.get("errorMessage"))
        clearance_event = None
        if scenario == "fridge-closed":
            # The nearest reachable point can lie inside the door's closing sweep.
            # This is explicit evaluator setup, before the policy takes control.
            clearance_event = setup_step(action="MoveBack", moveMagnitude=0.25)
            if not clearance_event.metadata["lastActionSuccess"]:
                raise RuntimeError("Cannot establish fridge door clearance: " + clearance_event.metadata.get("errorMessage", ""))
            event = clearance_event
        self.last_event = event
        self._frame_buffer = []
        self.frame_info = []
        self._capture()
        condition = {"objectId": target["objectId"], "field": field, "equals": False}
        setup = {"scenario": scenario, "setupAction": action, "target": target["objectId"],
                 "initialPosition": event.metadata["agent"]["position"],
                 "initialYaw": yaw, "simulatorAssistedSetup": True,
                 "events": self.setup_audit}
        if clearance_event is not None:
            setup["events"][-1]["purpose"] = "Explicit evaluator setup for door sweep clearance"
        task = "Turn off the running faucet." if field == "isToggled" else "Close the open refrigerator door."
        return task, [condition], setup

    def setup_held_problem(self, reference_path, scenario, opacity=1):
        from placement_task import apply_held_fixture
        kind = {"held-mug-counter": "Mug", "held-egg-counter": "Egg"}[scenario]
        return apply_held_fixture(self, reference_path, kind, opacity)

    def close(self):
        try:
            self.controller.stop()
        finally:
            journal = getattr(self, "command_journal", None)
            if journal:
                journal.close()
