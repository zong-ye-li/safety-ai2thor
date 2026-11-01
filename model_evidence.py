"""Persist exact model image inputs and every call's start/end/error before execution."""
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import time


class ModelEvidence:
    def __init__(self, output):
        self.root = Path(output)
        self.images = self.root / "model_inputs"
        self.images.mkdir(exist_ok=False)
        self.stream = (self.root / "model_calls.jsonl").open("x", encoding="utf-8")
        self.sequence = 0
        self.wait_boundary = None

    def _write(self, value):
        self.stream.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
        self.stream.flush()

    def _frames(self, frames, views=None):
        from PIL import Image
        refs = []
        if views is not None and len(views)!=len(frames):
            raise ValueError('Image derivation mapping must match actual input images')
        for index,frame in enumerate(frames):
            if isinstance(frame, str):
                content = base64.b64decode(frame, validate=True)
                with Image.open(io.BytesIO(content)) as image:
                    size, kind = image.size, image.format
                    image.verify()
                extension = "jpg" if kind == "JPEG" else "png"
                representation = "exact_encoded_network_image"
            else:
                image = Image.fromarray(frame).convert("RGB")
                size = image.size
                data = io.BytesIO()
                image.save(data, format="PNG")
                content, extension = data.getvalue(), "png"
                representation = "lossless_native_RGB"
            digest = hashlib.sha256(content).hexdigest()
            path = self.images / f"{digest}.{extension}"
            if not path.exists():
                with path.open("xb") as stream:
                    stream.write(content)
            ref={"path": path.relative_to(self.root).as_posix(), "sha256": digest,
                 "width": size[0], "height": size[1], "representation": representation}
            if views is not None and 'visual_inspection' in views[index]:
                from copy import deepcopy
                ref['representation']='derived_recorded_RGB_region_'+('encoded' if isinstance(frame,str) else 'lossless')
                ref['visual_inspection']=deepcopy(views[index]['visual_inspection'])
            refs.append(ref)
        return refs

    def invoke(self, phase, step, attempt, request, observation, frames, operation, statistics, *, on_error=None):
        if self.wait_boundary is not None:
            self.wait_boundary()
        views=None
        if isinstance(request,dict):
            candidate=request.get('views',request.get('image_views'))
            if candidate and any('visual_inspection' in v for v in candidate):views=candidate
        from image_slots import enabled as shared_images,groups
        if shared_images(request):
            from copy import deepcopy
            from image_slots import raster
            key='views' if 'views' in request else 'image_views'
            bindings=groups(request,len(frames),key)
            refs=self._frames(frames)
            for ref,group,frame in zip(refs,bindings,frames):
                pixels,policy=raster(frame)
                ref.update(observation_bindings=deepcopy(group),processing_policy=policy,
                           raster_sha256=hashlib.sha256(pixels.tobytes()).hexdigest())
        else:
            refs = self._frames(frames,views)
        self.sequence += 1
        call_id = f"call-{self.sequence:06d}"
        common = {"call_id": call_id, "phase": phase, "step": step, "attempt": attempt}
        from input_references import enabled as shared_input_enabled
        if shared_input_enabled(request):
            from policy_io import model_input_projection
            wire,profile=model_input_projection(request,observation if phase=='action' else None)
            common['model_input_encoding']={**profile,'wire_packet':wire}
        self._write({**common, "kind": "call_start", "utc": datetime.now(timezone.utc).isoformat(),
                     "request": request, "observation": observation, "images": refs})
        started = time.monotonic()
        try:
            value = operation()
        except BaseException as exc:
            self._write({**common, "kind": "call_error", "error": repr(exc),
                         "elapsed_seconds": time.monotonic()-started, "stats": statistics()})
            if on_error is not None:on_error(call_id,repr(exc))
            raise
        self._write({**common, "kind": "call_end", "raw_response": value[1] if isinstance(value, tuple) else value,
                     "elapsed_seconds": time.monotonic()-started, "stats": statistics()})
        # Persist the received model response before testing the environment barrier.
        # A changed/unknown endpoint prevents execution,never repeats the model call.
        if self.wait_boundary is not None:
            self.wait_boundary()
        return value

    def store_images(self, frames):
        """Record survey evidence without fabricating an inference call."""
        return self._frames(frames)

    def close(self):
        self.stream.close()
