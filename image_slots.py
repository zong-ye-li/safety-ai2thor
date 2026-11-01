"""Exact per-call model raster sharing; observation identities never merge."""
import base64
from copy import deepcopy
import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image

VERSION='exact-model-raster-slots-v1'
SCOPE=('Only exact pixels share image_index. Observation IDs,time,pose,commands,current and region parents stay independent; '
       'equal RGB does not prove unchanged world state. Cite observation_id when shared. Regions are NOT_A_NEW_CAPTURE.')


def enabled(request):
    value=request.get('image_slot_contract') if isinstance(request,dict) else None
    return isinstance(value,dict) and value.get('version')==VERSION


def raster(frame):
    if isinstance(frame,str):
        with Image.open(io.BytesIO(base64.b64decode(frame,validate=True))) as image:
            value=np.array(image.convert('RGB'))
        policy='decoded_encoded_RGB'
    else:
        value=np.asarray(frame);policy='native_RGB'
        if value.dtype!=np.uint8 or value.ndim!=3 or value.shape[2]!=3:
            raise ValueError('Expected actual uint8 RGB model input')
    if min(value.shape[:2])<1:raise ValueError('Empty model raster')
    return value,policy


def groups(request,image_count,key):
    views=request.get(key)
    if not isinstance(views,list):raise ValueError('Missing image views')
    result=[[] for _ in range(image_count)];seen=set()
    for view in views:
        index=view.get('image_index');identity=view.get('observation_id')
        if (type(index) is not int or not 0<=index<image_count
                or not isinstance(identity,str) or not 1<=len(identity)<=32 or identity in seen):
            raise ValueError('Invalid image-slot observation mapping')
        seen.add(identity);result[index].append(view)
    if any(not group for group in result):raise ValueError('Unreferenced model image slot')
    if enabled(request):
        contract=request['image_slot_contract']
        currents=[v['observation_id'] for v in views if v.get('current')]
        if len(currents)>1 or contract.get('current_observation_id')!=(currents[0] if currents else None):
            raise ValueError('Current observation binding differs')
        if contract.get('image_count')!=image_count or contract.get('observation_count')!=len(views):
            raise ValueError('Image-slot capacity receipt differs')
    elif len(views)!=image_count or any(len(group)!=1 for group in result):
        raise ValueError('Shared indices require the image-slot contract')
    return result


def share(request,frames,key):
    """Input frames are already fitted/encoded. No mutation,resize or approximation."""
    if enabled(request):raise ValueError('Model images have already been mapped')
    views=request.get(key)
    if not isinstance(views,list) or len(views)!=len(frames) or len(views)>8:
        raise ValueError('Expected at most eight observation/image pairs')
    result=deepcopy(request);unique=[];entries=[];indices=[]
    for i,(view,frame) in enumerate(zip(views,frames)):
        if type(view.get('image_index')) is not int or view['image_index']!=i:
            raise ValueError('Expected original ordered image mapping')
        pixels,policy=raster(frame);raw=pixels.tobytes()
        identity=(policy,pixels.shape,hashlib.sha256(raw).digest())
        # Hash filters candidates; byte equality is required even on collision.
        index=next((j for j,(key_,data) in enumerate(entries) if key_==identity and data==raw),None)
        if index is None:
            index=len(unique);unique.append(frame);entries.append((identity,raw))
        indices.append(index);result[key][i]['image_index']=index
    currents=[v['observation_id'] for v in views if v.get('current')]
    if len(currents)>1:raise ValueError('Multiple current captures')
    result['image_slot_contract']={'version':VERSION,'image_count':len(unique),
        'observation_count':len(views),'current_observation_id':currents[0] if currents else None,
        'scope':SCOPE}
    groups(result,len(unique),key)
    return result,unique


def recorded_view_images(call,root):
    """Validate archive bytes and reconstruct every observation's submitted raster."""
    request=call['request'];key='views' if 'views' in request else 'image_views'
    views=request.get(key,[]);refs=call['images'];root=Path(root).resolve()
    if enabled(request):groups(request,len(refs),key)
    elif len(views)!=len(refs):raise ValueError('Legacy view/image count differs')
    result=[]
    for i,view in enumerate(views):
        ref=refs[view['image_index'] if enabled(request) else i]
        path=(root/ref['path']).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest()!=ref['sha256']:
            raise ValueError('Recorded image provenance differs')
        with Image.open(path) as image:pixels=np.array(image.convert('RGB'))
        if enabled(request):
            if ([ref['width'],ref['height']]!=[pixels.shape[1],pixels.shape[0]]
                    or hashlib.sha256(pixels.tobytes()).hexdigest()!=ref['raster_sha256']):
                raise ValueError('Recorded model raster differs')
        result.append((deepcopy(view),pixels))
    return result
