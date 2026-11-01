"""Explicit template contracts. Adapter prefill is never canonical-template evidence."""
import hashlib

GEMMA12B = 'google/gemma-4-12B-it'
GEMMA12B_REVISION = '707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7'


def resolve_provider(provider, model_name):
    if provider == 'auto':
        return 'gemma4' if model_name == GEMMA12B else 'qwen' if model_name.startswith('Qwen/') else 'generic'
    if provider not in {'qwen', 'gemma4', 'generic'}:
        raise ValueError('Unknown model provider')
    return provider


class ThinkingContract:
    def __init__(self, provider, tokenizer, template, enabled):
        self.provider, self.enabled = provider, enabled
        self.close = '<channel|>' if provider == 'gemma4' else '</think>'
        self.prefill = '<|channel>thought\n' if provider == 'gemma4' and enabled else ''
        self.end_ids = tuple(tokenizer.encode(self.close, add_special_tokens=False)) if enabled else ()
        if enabled:
            if provider == 'generic' or not self.end_ids or tokenizer.decode(self.end_ids, skip_special_tokens=False) != self.close:
                raise ValueError('Unverified thinking boundary for provider')
            if any(i == tokenizer.unk_token_id for i in self.end_ids):
                raise ValueError('Thinking boundary contains unknown tokens')
        self.evidence = {'version': 'provider-thinking-v1', 'provider': provider,
            'enabled': enabled, 'end_token_ids': list(self.end_ids),
            'template_sha256': hashlib.sha256(template.encode()).hexdigest(),
            'adapter_prefill': self.prefill, 'adapter_contract': 'gemma-thought-prefill-v1' if self.prefill else None,
            'canonical_prefills_thought': provider == 'qwen' and enabled}

    def prepare(self, text):
        if self.enabled:
            if self.provider == 'gemma4':
                if not text.endswith('<|turn>model\n') or '<|think|>\n' not in text:
                    raise ValueError('Pinned Gemma thinking template changed')
                return text + self.prefill
            if not text.rstrip().endswith('<think>'):
                raise ValueError('Qwen thinking template must prefill <think>')
        return text

    def final(self, text):
        return text.split(self.close, 1)[1].strip() if self.close in text else None
