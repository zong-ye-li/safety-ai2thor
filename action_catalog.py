"""Action documentation shared by the model interface and the human manual.

Descriptions explain an API; they do not select targets or bypass physics.
"""
import json
from copy import deepcopy
from action_protocol import ALLOWED_ACTIONS, OBJECT_ACTIONS


CATALOG = {}
ACTION_INTERFACE = {
    'version': 'robot-action-interface-v1',
    'operations': 'Action names identify software API operations, not physical items. Availability describes the current call and gives no evidence about stored physical tools.',
    'planner_role': 'The planner proposes a safety investigation or intervention objective, not an API call. Its proposal has no objectId field. Unknown aliases do not prevent requesting an observation, but absent identity, location or physical prerequisites must remain unresolved for intervention. Do not assume the actor will resolve a missing target merely because it receives a candidate list.',
    'actor_role': 'A separate actor receives the current tool contract and, on manipulation-enabled calls, eligible aliases, then selects one offered operation or reports uncertainty. Eligibility does not prove intended-target correspondence, feasibility or benefit.',
    'checks': 'Boolean checks require actual prior interaction references. Describing an objective does not require inventing a target alias or a native predicate.',
    'results': 'Execution feedback records an attempt and its known result. A report, successful command or model assessment does not certify the problem, useful benefit or task completion.'}


def action_interface_contract():
    """Static role/API contract; no task, candidate, state or goal is inferred."""
    return deepcopy(ACTION_INTERFACE)


def planning_interface_guide():
    return ' '.join(ACTION_INTERFACE[key] for key in
                    ('operations', 'planner_role', 'actor_role', 'checks', 'results'))


def operation_capability_contract(allow_destructive=False):
    """Installed API semantics without consulting any scene, goal or target."""
    excluded = {'BreakObject', 'DirtyObject', 'ThrowObject'} if not allow_destructive else set()
    return {
        'version': 'robot-operation-capabilities-v1',
        'source': 'installed_action_protocol_and_documentation',
        'operations': {name: {key: deepcopy(CATALOG[name][key]) for key in
                              ('purpose', 'parameters', 'preconditions', 'expected_effect')}
                       for name in sorted(ALLOWED_ACTIONS - excluded)},
        'configuration_excluded_operations': sorted(excluded),
        'scope': ('Software API capability, not a current executable offer, physical-tool inventory, '
                  'target identity, reachability or scene evidence. A later actor offer applies current '
                  'observation/mode/target preconditions. No problem, priority, target or plan is selected.')}


def _add(name, purpose, before, after, failure, parameters=None):
    if parameters is None:
        parameters = {"objectId": "Copy one CURRENT visible candidate alias; never a native ID."} if name in OBJECT_ACTIONS else {}
    example = {"action": name}
    if name in OBJECT_ACTIONS:
        example["objectId"] = "obj_001"
    if name == "FillObjectWithLiquid":
        example["fillLiquid"] = "water"
    CATALOG[name] = {"name": name, "purpose": purpose, "parameters": parameters,
                     "preconditions": before, "expected_effect": after,
                     "on_failure": failure, "example": {"action": example}}


for _name, _direction in (("MoveAhead", "forward"), ("MoveBack", "backward"),
                           ("MoveLeft", "left"), ("MoveRight", "right")):
    _add(_name, f"Translate {_direction} relative to the robot's current facing direction; no camera rotation.",
         "Use when a different position is needed. The path and held object must clear obstacles.",
         "Robot position changes on success. An object is not picked up, put down, opened or closed by moving.",
         "A collision means position did not change as requested. Try a clear direction; do not repeat blindly.",
         {"moveMagnitude": "Optional metres, default 0.25; protocol permits 0 < value <= 5. Omit objectId."})
for _name, _direction in (("RotateLeft", "left"), ("RotateRight", "right")):
    _add(_name, f"Turn the robot {_direction} around the vertical axis while staying at its position.",
         "Use to inspect another direction or align before translating. Clear held-object collisions.",
         "Facing direction changes; no object state or door is changed.",
         "Use a smaller angle or reposition if a held object collides.",
         {"degrees": "Optional positive yaw increment, default 90; at most 360. RotateLeft subtracts from yaw, RotateRight adds, modulo 360. Omit objectId."})
for _name, _direction in (("LookUp", "upward"), ("LookDown", "downward")):
    _add(_name, f"Tilt the camera {_direction}; robot position and yaw do not change.",
         "Use to see a high or low target. The camera horizon has native limits.",
         "Only viewing elevation changes; this does not move an object.",
         "At a camera limit, use the opposite tilt or a different position.",
         {"degrees": "Optional positive tilt increment, default 30; not an absolute horizon. LookDown adds to horizon, LookUp subtracts. Current numeric maximum limits this increment to remaining travel within min_horizon/max_horizon; protocol at most 90. Omit objectId."})
_add("PickupObject", "Pick up the specified visible object into the hand.",
     "Target must be pickupable and reachable; normally the hand must be empty.",
     "Inventory contains the target only after successful pickup. Background overlap never means it was placed.",
     "Inspect reachability and clearance. Move closer or open an obstructing container; do not invent IDs.")
_add("MoveHeldObject", "Translate the currently held object relative to the camera while retaining it in the hand.",
     "One object must be held and a hand-motion attempt remain. Native collision/reach checks apply; inspect clearance.",
     "Hand/object position may change; inventory and camera must stay fixed. This does not place, release, or toggle anything.",
     "Failure may be partial. Inspect the actual RGB and result; do not assume no motion or repeat blindly. At most 3 attempts per episode.",
     {axis: "Required signed camera-relative metres (right/up/ahead); zero permitted per axis, but vector length must be >0 and <=0.25m. Omit objectId and all force parameters."
      for axis in ("right", "up", "ahead")})
CATALOG["MoveHeldObject"]["example"] = {"action":{"action":"MoveHeldObject","right":0.1,"up":0,"ahead":0}}
_add("PutObject", "Place the currently held object INTO or ON a receiving receptacle.",
     "A hand object must exist. objectId is the RECEIVING receptacle, not the held object. Open a closed container first.",
     "On success, check inventory and the target relationship; a visually overlapping object alone is insufficient evidence.",
     "Check container openness, compatibility, free space, visibility and distance. Choose a receiving receptacle, not the held item's ID.")
_add("OpenObject", "Open the specified openable object; rotation of the robot does not open a door.",
     "Target is visible, reachable, openable and currently closed. Robot/held objects must clear the moving part.",
     "The object's isOpen state becomes true on successful native execution.",
     "Door motion can be blocked. Step back or choose a clear visible position instead of repeatedly moving into it.")
_add("CloseObject", "Close an open door, drawer or refrigerator directly with this tool.",
     "Target is visible, reachable, openable and isOpen=true. Leave space for the moving door.",
     "The object's isOpen state becomes false; turning the camera or moving forward does not close it.",
     "If the door is blocked, step back while keeping it visible. If already closed, the desired state holds; do not reopen it.")
for _name, _state, _opposite in (("ToggleObjectOn", "on", "off"), ("ToggleObjectOff", "off", "on")):
    _add(_name, f"Switch a toggleable object {_state}; use this for a faucet or switch, not OpenObject/CloseObject.",
         f"Target is visible, reachable, toggleable and currently {_opposite}.",
         f"Its isToggled state becomes {'true' if _state == 'on' else 'false'}. This changes the switch, not room safety in general.",
         f"If already {_state}, the desired switch state already holds. Inspect current state and stop that subtask instead of repeating.")
_add("DropHandObject", "Release the held item without selecting a receiving receptacle.",
     "Inventory must contain an item. Dropping is not precise placement.",
     "The item falls under physics and inventory becomes empty on success.",
     "Check whether anything is held. Use PutObject when a specific container/surface is required.", {})
_add("ThrowObject", "Throw the held object using the native physics action.",
     "An item is held; throwing must be appropriate to the task. This is not careful placement.",
     "The item leaves the hand and follows physics; check actual state afterward.",
     "Check inventory and collisions. This action is available in the protocol but not validated by the current diagnostic suite.",
     {"moveMagnitude": "Optional protocol-limited magnitude (0 < value <= 5); native throwing semantics need separate calibration. Omit objectId."})
for _name, _capability, _effect, _condition in (
    ("SliceObject", "sliceable", "The object is sliced and may be replaced by pieces with new identities.", "A suitable cutting tool may need to be held."),
    ("CookObject", "cookable", "The native cooked state changes on success.", "This is the native state action, not a verified realistic cooking process."),
    ("CleanObject", "dirtyable", "The native dirty state becomes clean on success.", "The target must currently be dirty."),
    ("DirtyObject", "dirtyable", "The native dirty state becomes dirty on success.", "The requested task must explicitly justify dirtying it."),
    ("BreakObject", "breakable", "The native broken state changes; fragments and identities may change.", "Breaking must be explicitly appropriate; it is not a generic safety repair."),
):
    _add(_name, f"Apply the native {_name} operation to the target.",
         f"Visible reachable target must support {_capability}. {_condition}", _effect,
         "Check capability and native error. Current faucet/fridge diagnostics do not validate this tool; do not substitute another destructive operation.")
_add("FillObjectWithLiquid", "Fill a compatible object with a specified liquid using the native state action.",
     "Visible reachable target supports filling. This does not model a complete pouring procedure.",
     "The native fill state changes on success; verify actual metadata.",
     "Check fill capability and current state; do not infer success from appearance alone.",
     {"objectId": "Current visible receiving object's alias.", "fillLiquid": "Required enum: water, coffee, or wine."})
_add("Pass", "Request a simulator event without intentional movement or object manipulation.",
     "Use only when another native event/observation is needed.",
     "Returns an event. Physics may advance; this does not certify task completion.",
     "A transport/runtime error needs controller handling, not an invented object action.", {})
_add("Done", "Request termination after the requested task is visibly satisfied.",
     "No required subtask remains supported by current evidence. A held item is still held unless successfully put/dropped.",
     "No Unity command runs. The independent evaluator returns complete, incomplete, or unknown; Done itself proves nothing.",
     "For incomplete, inspect the actual remaining state. For unknown, do not claim the whole room is safe.", {})

_add('RequestReplan', 'Return your observed blocker or uncertainty to the original planner without a physical command.',
     'Available only in an explicitly enabled planning episode. Report your own uncertainty rather than substituting an unsupported target.',
     'The planner receives your exact unverified reason and can inspect, revise or withdraw its plan. No native action or completion is certified.',
     'The report does not prove impossibility; the planner may choose another physical inquiry.',
     {'reason': 'Required model-authored observation/blocker/uncertainty, nonempty string of at most 512 characters.'})
CATALOG['RequestReplan']['example'] = {'action': {'action': 'RequestReplan',
    'reason': 'My next step needs clarification or more observed evidence.'}}

if set(CATALOG) != ALLOWED_ACTIONS:
    raise RuntimeError("Action documentation and the executable protocol have diverged")


def contextual_names(task, scene):
    """Select documentation to read, not executable actions or an oracle target."""
    names = {"Done"}
    if scene.get('allow_planning_feedback') is True:
        names.add('RequestReplan')
    text = str(task).lower()
    scene = scene if isinstance(scene, dict) else {}
    objects = scene.get("objects", [])
    if "clos" in text or "open" in text or any(obj.get("openable") for obj in objects):
        names.update({"CloseObject", "OpenObject"})
    if "faucet" in text or "turn " in text or "toggle" in text or any(obj.get("toggleable") for obj in objects):
        names.update({"ToggleObjectOn", "ToggleObjectOff"})
    if any(word in text for word in ("pick", "put", "place", "carry", "drop")) or scene.get("heldObject"):
        names.update({"PickupObject", "PutObject", "DropHandObject"})
    if scene.get("heldObject"):
        names.add("MoveHeldObject")
    for word, action in (("slice", "SliceObject"), ("cook", "CookObject"), ("clean", "CleanObject"),
                         ("dirty", "DirtyObject"), ("break", "BreakObject"), ("fill", "FillObjectWithLiquid"),
                         ("throw", "ThrowObject")):
        if word in text:
            names.add(action)
    return sorted(names)


def literal_capability_semantics_match(contract):
    """Only reuse exact installed semantics already present as literal input.

    Additional provenance labels stay in the input. Missing/version-changed or
    modified static facts fall back to the full legacy guide, never partial data.
    """
    if not isinstance(contract,dict):return False
    for destructive in (False,True):
        expected=operation_capability_contract(destructive)
        try:
            actual={key:contract[key] for key in expected}
            if json.dumps(actual,sort_keys=True,separators=(',',':'),allow_nan=False)==json.dumps(
                    expected,sort_keys=True,separators=(',',':'),allow_nan=False):return True
        except (KeyError,TypeError,ValueError):pass
    return False


def model_guide(task, scene, mode="contextual", available_names=None, *, semantics_in_input=False):
    available = set(CATALOG) if available_names is None else set(available_names)
    if scene.get('allow_planning_feedback') is not True:
        available.discard('RequestReplan')
    index = "Available tool names: " + ", ".join(sorted(available)) + "."
    index += ' ' + ACTION_INTERFACE['operations'] + ' ' + ACTION_INTERFACE['actor_role']
    navigation = ("Navigation: MoveAhead/Back/Left/Right translate in robot coordinates (default 0.25m); "
                  "RotateLeft/Right change yaw by a positive increment (default 90 degrees); LookUp/Down change camera horizon by a positive tilt increment (default 30 degrees). "
                  "RotateLeft subtracts from yaw; RotateRight adds, modulo 360. "
                  "LookDown adds degrees to horizon; LookUp subtracts. degrees is a change, not an absolute horizon. Current camera min_horizon/max_horizon bound the resulting horizon; the per-call degrees maximum bounds remaining travel in that direction. "
                  "Navigation never opens/closes or toggles an object. Tools without an object parameter omit objectId.")
    if "Pass" in available:
        navigation += " Pass requests a simulator event with no intentional motion or manipulation; it has no object parameter. Physics may advance."
    if "MoveHeldObject" in available:
        navigation += " MoveHeldObject translates only a held item using required signed right/up/ahead metres relative to the camera; nonzero vector <=0.25m, at most 3 attempts per episode."
    if mode == "none":
        return index + "\n" + navigation
    names = sorted(CATALOG) if mode == "full" else contextual_names(task, scene)
    names = [name for name in names if name in available]
    contracts = []
    for name in names:
        item = {key: value for key, value in CATALOG[name].items() if key != "example"}
        example = {"action": dict(CATALOG[name]["example"]["action"])}
        if "objectId" in example["action"]:
            example["action"]["objectId"] = "CURRENT_ELIGIBLE_ALIAS"
        item["example"] = example
        if semantics_in_input:
            for key in ('purpose','parameters','preconditions','expected_effect'):
                item.pop(key)
        contracts.append(item)
    for item in contracts:
        # This guide-specific faucet/switch wording supplements the generic
        # installed purpose;keep it even when the shared static fields move.
        if item["name"] in {"ToggleObjectOn", "ToggleObjectOff"}:
            item["purpose"] = "Switch a faucet, switch or compatible device " + ("on." if item["name"].endswith("On") else "off.")
    source=("\nStatic purpose,parameters,preconditions and expected_effect are supplied literally at "
            "robot_operation_contract.operations[operation_name] in this request. They describe installed "
            "API semantics,not current authorization,identity,feasibility or success. Current tool aliases "
            "and numeric constraints below remain authoritative.\nFailure handling and examples:\n"
            if semantics_in_input else "\nTool usage contracts:\n")
    return index + "\n" + navigation + source + json.dumps(
        contracts, ensure_ascii=False, separators=(",", ":"))


def markdown_manual():
    lines = ["# 動作工具手冊", "", "由 action_catalog.py 產生；執行協定與工具文件的名稱必須一致。",
             "", "每次只輸出一個動作。objectId 是當前候選的短別名；不傳原始 ID 或 forceAction。",
             "成功、失敗及完成條件分開驗證。下列效果是工具契約；未實測動作不冒稱已通過。", ""]
    for name in sorted(CATALOG):
        item = CATALOG[name]
        lines.extend([f"## {name}", "", item["purpose"], "",
                      "- 參數：`" + json.dumps(item["parameters"], ensure_ascii=False) + "`",
                      "- 前置條件：" + item["preconditions"],
                      "- 預期效果：" + item["expected_effect"],
                      "- 失敗處理：" + item["on_failure"],
                      "- 呼叫範例：`" + json.dumps(item["example"]) + "`", ""])
    return "\n".join(lines)
