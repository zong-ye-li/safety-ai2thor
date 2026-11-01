"""Episode RGB-only held appearance configuration, independent of policy goals."""
import math
import os

VERSION='episode-held-rgb-v1'
DEFAULT_OPACITY = .5
SCOPE=('Held objects remain physically present,in inventory and collision geometry. RGB-only opacity reveals background; '
       'the partly visible held foreground can overlap background surfaces and is not a separate world object. '
       'depth and instance masks retain original geometry and do not describe the same visible surfaces.')


def public_contract(receipt):
    if not receipt or receipt.get('status') not in {'configured','verified_held_rgb'}:return None
    return {'version':VERSION,'rgb_opacity':receipt['requested_opacity'],'scope':SCOPE}


def _information(metadata):
    if metadata.get('lastActionSuccess') is not True:
        raise ValueError('Held appearance interface refused: '+str(metadata.get('errorMessage','unknown')))
    info=metadata.get('actionReturn')
    if (not isinstance(info,dict) or info.get('renderMode')!='own_rgb_camera_only'
            or info.get('supported') is not True or info.get('supportError') or info.get('lastError')):
        raise ValueError('Held RGB capability unsupported/unknown: '+str(info))
    return info


def configure(env,args,save=lambda:None):
    mode=getattr(args,'held_appearance','auto');value=getattr(args,'held_opacity',None)
    custom=bool(getattr(args,'editor_port',None) is not None or getattr(args,'local_executable',None))
    fixture=getattr(args,'scenario',None) in {'held-mug-counter','held-egg-counter'}
    requested=(1.0 if fixture else DEFAULT_OPACITY) if value is None else value
    if mode=='opaque':requested=1.0
    if type(requested) not in (int,float) or not math.isfinite(requested) or not 0<=requested<=1:
        raise ValueError('Held opacity must be finite and in[0,1]')
    receipt={'version':VERSION,'mode':mode,'requested_opacity':requested,'status':'starting',
             'custom_player_declared':custom,'player_sha256':os.environ.get('SAFETY_AI2THOR_PLAYER_SHA256'),
             'player_manifest_sha256':os.environ.get('SAFETY_AI2THOR_PLAYER_MANIFEST_SHA256') or None,
             'scope':SCOPE,'native_setup_commands':[],'verified_boundaries':0}
    env.held_appearance_contract=receipt;save()
    if requested==1:
        receipt.update(status='explicit_opaque',reason='Declared opaque control/configuration');save();return receipt
    if not custom:
        receipt.update(status='unavailable_backend',reason='No custom player declared;stock Cloud build lacks held appearance extension')
        save()
        if mode=='required' or value is not None:
            raise ValueError('Requested RGB transparency requires a qualified custom Unity player;no model started')
        return receipt
    def command(action,**fields):
        event=env.native_step('held_rgb_configuration',action=action,**fields)
        receipt['native_setup_commands'].append({'action':action,'command_id':env.last_command_id})
        return _information(event.metadata)
    command('GetHeldObjectAppearance')
    configured=command('SetHeldObjectOpacity',opacity=requested)
    if configured.get('requestedOpacity')!=requested:raise ValueError('Held opacity configuration not acknowledged')
    receipt['status']='configured';save()
    env._capture()
    verify(env,save)
    return receipt


def verify(env,save=lambda:None):
    receipt=getattr(env,'held_appearance_contract',None)
    if public_contract(receipt) is None:return
    clock=getattr(env,'physics_clock',None)
    try:
        if clock is not None:
            barrier=clock.check_wait(appearance=True);info=barrier['appearance']
        else:
            info=_information(env.native_step('held_rgb_verification',action='GetHeldObjectAppearance').metadata)
    except BaseException as exc:
        receipt.update(status='failed_runtime_verification',error=repr(exc));save()
        raise
    inventory=env.snapshot().get('inventoryObjects',[])
    held=inventory[0].get('objectId') if inventory else None
    if (not isinstance(info,dict) or info.get('requestedOpacity')!=receipt['requested_opacity']
            or info.get('heldObjectId')!=held or info.get('supported') is not True
            or info.get('supportError') or info.get('lastError')):
        receipt['status']='failed_runtime_verification';save()
        raise ValueError('Held RGB configuration/identity/support changed;no silent opaque fallback')
    if held is not None and info.get('lastRgbOpacity')!=receipt['requested_opacity']:
        receipt['status']='failed_runtime_verification';save()
        raise ValueError('Current held object RGB opacity not applied')
    receipt.update(status='verified_held_rgb' if held is not None else 'configured',
                   verified_boundaries=receipt['verified_boundaries']+1,
                   last_verification_command_id=env.last_command_id,
                   last_rgb_opacity=info.get('lastRgbOpacity'),physically_held=held is not None)
    save()
