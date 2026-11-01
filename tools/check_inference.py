"""Check installed inference imports and optional cached files without loading weights or using a GPU."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir')
    parser.add_argument('--check-cache', action='store_true')
    parser.add_argument('--ssh', action='store_true', help='Also check POSIX/tmux and mailbox write access on this host')
    parser.add_argument('--session-root')
    args = parser.parse_args()
    if args.session_root is not None and not args.ssh:
        parser.error('--session-root requires --ssh')
    import torch
    import torchvision
    import accelerate
    import ninja
    from optimum.quanto import QModuleMixin
    from transformers import AutoProcessor, QuantoConfig
    from transformers.models.gemma4_unified.modeling_gemma4_unified import (
        Gemma4UnifiedForConditionalGeneration, Gemma4UnifiedTextMLP, Gemma4UnifiedVisionEmbedder)
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
    from model_provider import GEMMA12B, GEMMA12B_REVISION
    from tools.serve_inference import mailbox_root
    packages = ('torch', 'torchvision', 'transformers', 'accelerate', 'optimum-quanto',
                'ninja', 'lm-format-enforcer', 'huggingface-hub')
    result = {'imports': 'passed', 'packages': {p: importlib.metadata.version(p) for p in packages},
              'torch_cuda_build': torch.version.cuda,
              'scope': 'Imports and optional cached file presence only; no weights loaded, GPU kernel, model generation or simulator tested'}
    if args.ssh:
        import fcntl
        if shutil.which('tmux') is None:
            raise RuntimeError('Install tmux and expose it on the non-interactive SSH PATH')
        directory = mailbox_root(ROOT, args.session_root)
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        with tempfile.TemporaryFile(dir=directory) as stream:
            stream.write(b'write-check')
        result['mailbox_root'] = str(directory)
    if args.check_cache:
        from huggingface_hub import snapshot_download
        # Hub completeness checks also include repository prose by default.
        # Check runtime assets; a missing README is not a missing model shard.
        snapshot = Path(snapshot_download(GEMMA12B, revision=GEMMA12B_REVISION,
                                         cache_dir=args.cache_dir, local_files_only=True,
                                         allow_patterns=['*.json', '*.safetensors', '*.model', '*.jinja', '*.txt']))
        AutoProcessor.from_pretrained(str(snapshot), local_files_only=True)
        index = snapshot/'model.safetensors.index.json'
        files = set(json.loads(index.read_text())['weight_map'].values()) if index.is_file() else {'model.safetensors'}
        if not files:
            raise ValueError('The checkpoint index contains no weight files')
        for name in files:
            path = snapshot/name
            if Path(name).is_absolute() or '..' in Path(name).parts or not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError('Missing/invalid cached checkpoint shard: '+name)
        result['cache'] = {'snapshot': str(snapshot), 'processor': 'loaded locally',
                           'weight_files_present': len(files), 'weight_contents_validated': False}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
