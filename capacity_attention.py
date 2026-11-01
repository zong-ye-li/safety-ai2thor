"""Request-scoped, CUDA memory-efficient SDPA with explicit grouped-head expansion."""
from contextlib import contextmanager


def install_gemma_mlp_chunking(model, chunk_size=1024):
    """Bound token-independent inference intermediates without changing context.

    Only the pinned dense Gemma text MLP is adapted. We call its original forward
    on every token, keeping original weights/activation and output ordering.
    """
    import hashlib
    import inspect
    from types import MethodType
    import torch
    from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedTextMLP

    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError('MLP chunk size must be a positive integer')
    original = Gemma4UnifiedTextMLP.forward
    modules = [(name, module) for name, module in model.named_modules()
               if type(module) is Gemma4UnifiedTextMLP]
    if not modules or any('forward' in module.__dict__ for _, module in modules):
        raise ValueError('Expected unmodified Gemma dense MLP modules')

    def chunked(module, x):
        if x.ndim != 3:
            raise ValueError('Gemma MLP expects batch, sequence, hidden dimensions')
        if x.shape[1] <= chunk_size:
            return original(module, x)
        if module.training or torch.is_grad_enabled():
            raise ValueError('Chunked Gemma MLP is inference-only')
        output = torch.empty_like(x)
        for start in range(0, x.shape[1], chunk_size):
            stop = min(start + chunk_size, x.shape[1])
            output[:, start:stop, :] = original(module, x[:, start:stop, :])
        return output

    for _, module in modules:
        module.forward = MethodType(chunked, module)
    return {'version': 'gemma-dense-token-chunk-v1', 'token_chunk_size': chunk_size,
            'modules': [name for name, _ in modules],
            'original_forward_sha256': hashlib.sha256(inspect.getsource(original).encode()).hexdigest(),
            'scope': 'All tokens and original dense MLP weights/activation retained; inference only'}


@contextmanager
def attention_execution(implementation, backend='auto', query_chunk_size=1024):
    if type(query_chunk_size) is not int or query_chunk_size < 1:
        raise ValueError('Query chunk size must be a positive integer')
    if backend not in {'auto', 'efficient'}:
        raise ValueError('Unknown SDPA backend contract')
    if backend == 'auto':
        yield
        return
    if implementation != 'sdpa':
        raise ValueError('Efficient-only attention requires the sdpa implementation')
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    import torch

    class ExpandedHeads:
        num_key_value_groups = 1
        def __init__(self, original):
            self.original = original
        def __getattr__(self, name):
            return getattr(self.original, name)

    def efficient(module, query, key, value, *args, **kwargs):
        q_heads, k_heads, v_heads = query.shape[1], key.shape[1], value.shape[1]
        if k_heads != v_heads or q_heads % k_heads:
            raise ValueError('Invalid grouped attention head counts')
        if q_heads != k_heads:
            groups = q_heads // k_heads
            key = key.repeat_interleave(groups, dim=1)
            value = value.repeat_interleave(groups, dim=1)
        # The installed Transformers5.20 SDPA enables native GQA without a mask.
        # V100's efficient kernel rejects unequal heads; expanding first and
        # exposing groups=1 avoids that fallback without changing any weights.
        # A full explicit mask preserves every query/key relation. Split only
        # query rows to bound efficient-SDPA mask/workspace materialization;
        # never truncate keys, context, images or the model's sliding masks.
        mask = args[0] if args else kwargs.get('attention_mask')
        qlen = query.shape[-2]
        if (qlen > query_chunk_size and mask is not None and mask.ndim == 4
                and mask.shape[-2] in (1, qlen)):
            chunks = []
            for start in range(0, qlen, query_chunk_size):
                stop = min(start + query_chunk_size, qlen)
                part = mask if mask.shape[-2] == 1 else mask[..., start:stop, :]
                options = dict(kwargs)
                if args:
                    positional = (part, *args[1:])
                else:
                    positional = ()
                    options['attention_mask'] = part
                result, _ = sdpa_attention_forward(ExpandedHeads(module), query[..., start:stop, :],
                    key, value, *positional, **options)
                chunks.append(result)
            return torch.cat(chunks, dim=1), None
        return sdpa_attention_forward(ExpandedHeads(module), query, key, value, *args, **kwargs)

    previous = ALL_ATTENTION_FUNCTIONS['sdpa']
    try:
        ALL_ATTENTION_FUNCTIONS.register('sdpa', efficient)
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            yield
    finally:
        ALL_ATTENTION_FUNCTIONS.register('sdpa', previous)
