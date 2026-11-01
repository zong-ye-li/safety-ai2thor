"""Weight-only loading and physical resident-storage evidence (not model qualification)."""
import importlib.metadata


def install_gemma_vision_precision(model):
    """Keep the overflowing patch projection through its next norm in FP32.

    Resident weights stay unchanged/quantized on CUDA. FP32 dequantized projection
    weights are temporary, and normalized activations return to the model dtype.
    """
    from types import MethodType
    import torch.nn.functional as F
    from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedVisionEmbedder
    embedders = [(name, module) for name, module in model.named_modules()
                 if type(module) is Gemma4UnifiedVisionEmbedder]
    if len(embedders) != 1:
        raise ValueError('Expected one pinned Gemma vision embedder')
    name, embedder = embedders[0]
    dense, norm = embedder.patch_dense, embedder.patch_ln2
    if any('forward' in module.__dict__ for module in (dense, norm)):
        raise ValueError('Vision precision adapter already installed or overridden')
    output_dtype = dense.weight.dtype

    def project(module, x):
        weight = module.weight.dequantize().float()
        bias = module.bias.float() if module.bias is not None else None
        return F.linear(x.float(), weight, bias)

    def normalize(module, x):
        weight = module.weight.float() if module.weight is not None else None
        bias = module.bias.float() if module.bias is not None else None
        return F.layer_norm(x, module.normalized_shape, weight, bias, module.eps).to(output_dtype)

    dense.forward = MethodType(project, dense)
    norm.forward = MethodType(normalize, norm)
    return {'version':'gemma-vision-projection-fp32-v1',
            'modules':[name+'.patch_dense', name+'.patch_ln2'],
            'intermediate_dtype':'torch.float32', 'normalized_output_dtype':str(output_dtype),
            'resident_weights_changed':False,
            'scope':'FP32 transient projection and following normalization; no clamp, offload or persistent full-precision weights'}


class StorageAdmissionError(ValueError):
    """Carry the complete rejected audit; callers must preserve it, not retry."""
    def __init__(self, evidence):
        self.evidence = evidence
        super().__init__('Storage admission rejected: ' + '; '.join(evidence['violations']))


def quantization_kwargs(quantization, dtype, device_map):
    if quantization == 'none':
        return {}
    if quantization not in {'quanto-int8', 'quanto-float8'} or dtype != 'float16':
        raise ValueError('Weight-only Quanto requires explicit int8/float8 and float16 compute')
    if device_map != {'': 0}:
        raise ValueError('Quantized loader requires explicit single CUDA placement; no offload')
    from transformers import QuantoConfig
    return {'quantization_config': QuantoConfig(weights=quantization.split('-')[1])}


def place_quanto_buffers(model):
    """HF Quanto leaves activation-scale scalars on CPU during weight-only load.

    Move only these identified buffers to their own frozen weight's CUDA device;
    do not call model.to(), cast weights, permit offload or exempt other storage.
    """
    import torch
    from optimum.quanto import QModuleMixin
    moved = []
    for name, module in model.named_modules():
        if not isinstance(module, QModuleMixin):
            continue
        weight = module.weight
        if not module.frozen or weight.device.type != 'cuda':
            continue  # The subsequent strict audit rejects any invalid weights.
        for key in ('input_scale', 'output_scale'):
            value = module._buffers.get(key)
            if value is None or value.device == weight.device:
                continue
            if value.device.type != 'cpu' or value.ndim != 0 or not value.is_floating_point():
                raise ValueError('Unexpected Quanto activation-scale buffer: '+name+'.'+key)
            placed = value.to(device=weight.device)
            if placed.dtype != value.dtype or not torch.equal(placed.cpu(), value):
                raise ValueError('Quanto scale placement changed its value or dtype')
            setattr(module, key, placed)
            moved.append({'name':name+'.'+key,'from':'cpu','to':str(weight.device),
                          'dtype':str(value.dtype),'bytes':value.untyped_storage().nbytes(),
                          'value_preserved':True})
    return moved


def storage_evidence(model, quantization):
    """Flatten tensor subclasses to real storage; count aliased storage exactly once."""
    import torch
    seen, totals, entries, quantized, unquantized = set(), {}, [], [], []
    violations = []

    def visit(name, tensor, role):
        flatten = getattr(tensor, '__tensor_flatten__', None)
        if flatten is not None:
            names, _ = flatten()
            if not names:
                raise ValueError('Opaque tensor subclass has no physical storage children: ' + name)
            for child in names:
                visit(name + '.' + child, getattr(tensor, child), role)
            return
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        shared = key in seen
        seen.add(key)
        entries.append({'name': name, 'role': role, 'dtype': str(tensor.dtype), 'device': str(tensor.device),
                        'shape': list(tensor.shape), 'storage_bytes': storage.nbytes(), 'shared_storage': shared})
        if not shared:
            totals[str(tensor.device)] = totals.get(str(tensor.device), 0) + storage.nbytes()

    for name, module in model.named_modules():
        weight = getattr(module, 'weight', None)
        if quantization != 'none' and hasattr(module, 'frozen') and hasattr(module, 'weight_qtype') and not module.frozen:
            violations.append('Unfrozen quantized module retains full precision weights: ' + name)
        if isinstance(weight, torch.Tensor):
            if hasattr(weight, '_data') and hasattr(weight, '_scale'):
                data, scale = weight._data, weight._scale
                if quantization != 'none':
                    valid = data.dtype == torch.int8 if quantization == 'quanto-int8' else str(data.dtype).startswith('torch.float8_')
                    if not valid or data.element_size() != 1 or data.device.type != 'cuda' or scale.device.type != 'cuda':
                        violations.append('Quantized resident weights/scales differ from requested storage: ' + name)
                quantized.append({'module': name, 'class': type(module).__name__,
                                  'weight_class': type(weight).__name__, 'qtype': str(getattr(weight, 'qtype', None)),
                                  'storage_dtype': str(data.dtype), 'scale_dtype': str(scale.dtype)})
            else:
                unquantized.append({'module': name, 'class': type(module).__name__,
                                    'dtype': str(weight.dtype), 'numel': weight.numel()})
    for role, values in [('parameter', model.named_parameters()), ('buffer', model.named_buffers())]:
        for name, value in values:
            try:
                visit(name, value, role)
            except Exception as exc:
                violations.append('Cannot inspect physical storage: '+name+': '+repr(exc))
    non_cuda = [entry for entry in entries if not entry['device'].startswith('cuda:')]
    if quantization != 'none':
        if not quantized:
            violations.append('No actual quantized CUDA weights detected; refusing fallback')
        if non_cuda:
            violations.append('Non-CUDA physical storage found: '+', '.join(
                entry['name']+' ('+entry['role']+', '+entry['device']+')' for entry in non_cuda))
    versions = {}
    for package in ('torch', 'transformers', 'accelerate', 'optimum-quanto'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    evidence = {'version': 'physical-weight-storage-v2', 'requested_quantization': quantization,
            'actual_quantization_config': (model.config.quantization_config.to_dict()
                if hasattr(getattr(model.config, 'quantization_config', None), 'to_dict')
                else getattr(model.config, 'quantization_config', None)),
            'storage_bytes_by_device': totals, 'tensors': entries,
            'quantized_modules': quantized, 'unquantized_modules': unquantized, 'dependencies': versions,
            'non_cuda_storage': non_cuda, 'violations': violations,
            'admission': 'rejected' if violations else 'accepted'}
    if violations:
        raise StorageAdmissionError(evidence)
    return evidence


def move_processor_inputs(inputs, device, dtype):
    """Cast only floats; retain integer IDs/masks and image order/shape."""
    import torch
    return {name: value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
            if isinstance(value, torch.Tensor) else value for name, value in inputs.items()}
