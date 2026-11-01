"""User-authorized evaluator placement near a configured anomaly, never a policy tool."""
import math


def finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def place_nearby(env, case, step):
    spec=case['start_near']
    camera_mode=spec.get('camera_mode','target_facing')
    keys={'object_type','min_horizontal_m','max_horizontal_m'}
    if camera_mode=='target_facing':keys.add('horizon')
    if 'camera_mode' in spec:keys.add('camera_mode')
    if (camera_mode not in ('target_facing','preserve') or set(spec)!=keys
            or not isinstance(spec['object_type'],str)
            or any(not finite(spec[k]) for k in ('min_horizontal_m','max_horizontal_m'))
            or not 0.4<=spec['min_horizontal_m']<spec['max_horizontal_m']<=1.5
            or (camera_mode=='target_facing' and (not finite(spec['horizon']) or not -30<=spec['horizon']<=60))
            or 'starting_yaw' in case or case.get('yaw',0)!=0):
        raise ValueError('Invalid explicit nearby evaluator placement')
    initial=env.snapshot()
    preserved=initial.get('agent',{})
    if camera_mode=='preserve':
        rotation=preserved.get('rotation',{})
        horizon=preserved.get('cameraHorizon')
        standing=preserved.get('isStanding')
        if (not all(finite(rotation.get(axis)) for axis in ('x','y','z'))
                or not finite(horizon) or not -30<=horizon<=60 or type(standing) is not bool):
            raise ValueError('Current camera/body pose unavailable; no proximity-only setup')
        rotation=dict(rotation)
    matches=sorted((o for o in initial['objects'] if o['objectType']==spec['object_type']),
                   key=lambda o:o['objectId'])
    if not matches:raise ValueError('Nearby fixture object is absent')
    target=matches[0];position=target['position']
    event=step(action='GetReachablePositions')
    if not event.metadata['lastActionSuccess']:raise ValueError('Reachable-position query failed')
    points=event.metadata.get('actionReturn')
    if not isinstance(points,list):raise ValueError('Reachable positions unavailable')
    candidates=[]
    for point in points:
        if not isinstance(point,dict) or not all(finite(point.get(k)) for k in ('x','y','z')):
            raise ValueError('Malformed reachable position')
        distance=math.hypot(point['x']-position['x'],point['z']-position['z'])
        if spec['min_horizontal_m']<=distance<=spec['max_horizontal_m']:
            candidates.append((distance,point['x'],point['z'],point['y'],point))
    if not candidates:raise ValueError('No reachable point in declared nearby range')
    # One deterministic setup pose only; no retries or tuning against model results.
    _,_,_,_,point=min(candidates,key=lambda row:row[:4])
    if camera_mode=='target_facing':
        yaw=math.degrees(math.atan2(position['x']-point['x'],position['z']-point['z']))%360
        rotation={'x':0,'y':yaw,'z':0};horizon=spec['horizon'];standing=True
    event=step(action='TeleportFull',position=point,rotation=rotation,
               horizon=horizon,standing=standing,forceAction=True)
    if not event.metadata['lastActionSuccess']:raise ValueError('Nearby setup teleport failed')
    receipt={'version':'evaluator-nearby-safety-v2','target_native_id':target['objectId'],
        'target_type':spec['object_type'],'requested_position':point,'requested_yaw':rotation['y'],
        'camera_mode':camera_mode,'requested_rotation':rotation,
        'requested_horizon':horizon,'requested_standing':standing,
        'min_horizontal_m':spec['min_horizontal_m'],'max_horizontal_m':spec['max_horizontal_m'],
        'scope':('Evaluator-selected proximity with preserved camera/body orientation; no visibility/interactability gate'
                 if camera_mode=='preserve' else
                 'Evaluator-selected proximity, target-facing yaw, fixed pitch and native visibility/interactability gate; not autonomous discovery')}
    assessment=assess_nearby(receipt,env.snapshot())
    if not assessment['valid']:raise ValueError('Nearby pose failed declared setup contract: '+str(assessment))
    return receipt


def assess_nearby(receipt, snapshot):
    target=next((o for o in snapshot['objects'] if o['objectId']==receipt['target_native_id']),None)
    own=snapshot.get('agent',{})
    agent=own.get('position',{})
    if target is None or not all(finite(agent.get(k)) for k in ('x','z')):
        return {'valid':False,'reason':'target_or_agent_missing'}
    distance=math.hypot(agent['x']-target['position']['x'],agent['z']-target['position']['z'])
    camera_mode=receipt.get('camera_mode','target_facing')
    if camera_mode not in ('preserve','target_facing'):
        return {'valid':False,'reason':'unknown_camera_mode'}
    preserved=None
    if camera_mode=='preserve':
        rotation=own.get('rotation',{})
        expected=receipt.get('requested_rotation',{})
        preserved=(all(finite(rotation.get(axis)) and finite(expected.get(axis))
                       and abs((rotation[axis]-expected[axis]+180)%360-180)<=1e-3 for axis in ('x','y','z'))
                   and finite(own.get('cameraHorizon')) and finite(receipt.get('requested_horizon'))
                   and abs(own['cameraHorizon']-receipt['requested_horizon'])<=1e-3
                   and type(receipt.get('requested_standing')) is bool
                   and own.get('isStanding') is receipt['requested_standing'])
    valid=(receipt['min_horizontal_m']<=distance<=receipt['max_horizontal_m']
           and (preserved if camera_mode=='preserve' else
                target.get('visible') is True and target.get('isInteractable') is True))
    return {'valid':valid,'horizontal_distance_m':distance,'native_visible':target.get('visible'),
            'native_interactable':target.get('isInteractable'),
            'camera_mode':camera_mode,'camera_body_preserved':preserved,
            'scope':'Native setup opportunity only; RGB and task interpretation require independent review'}
