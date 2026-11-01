"""Offline delivery integrity and command checks; no player/model/network."""
import ast
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]

def package_files(root):
    """Ignore documented local environments/outputs, but verify every source file."""
    for directory, names, files in os.walk(root):
        names[:] = [name for name in names
                    if name not in {'__pycache__', '.git', 'runs', 'build', 'inference-sessions'}
                    and not name.startswith('.venv')]
        for name in files:
            path = Path(directory)/name
            if path != root/'MANIFEST.json' and path.suffix != '.pyc':
                yield path

def main():
    manifest = json.loads((ROOT/'MANIFEST.json').read_text(encoding='utf-8'))
    actual = {p.relative_to(ROOT).as_posix() for p in package_files(ROOT)}
    assert actual == set(manifest['files']), 'Delivery file set differs'
    for name, expected in manifest['files'].items():
        p = ROOT/name
        assert not p.is_symlink(), name
        assert hashlib.sha256(p.read_bytes()).hexdigest() == expected, name
        if p.suffix == '.py':
            ast.parse(p.read_text(encoding='utf-8'), filename=name)
    sys.path.insert(0, str(ROOT))
    from run_profiles import PROFILES
    from physics_time import validate_runtime_clock
    from validation.scripts.public_fixture_contract import validate_recovery_memory_entry
    cases = json.loads((ROOT/'examples/cases.json').read_text())
    assert [c['id'] for c in cases['cases']] == ['01', '02', '03']
    assert [c['states'][0]['type'] for c in cases['cases']] == ['Faucet', 'Fridge', 'Microwave']
    validate_recovery_memory_entry(cases['cases'])
    validate_runtime_clock(cases['physical_time'])
    profile = PROFILES['gemma12b-int8-rp105-t08']
    assert (profile['temperature'], profile['reasoning_repetition_penalty']) == (.8, 1.05)
    assert (profile['max_input_tokens'], profile['reasoning_token_budget'], profile['max_new_tokens']) == (32768, 16384, 20481)
    assert profile['frame_width'] == profile['frame_height'] == 448
    common = json.loads((ROOT/'examples/common-arguments.json').read_text())
    assert '--planning-record-tables' in common and '--planning-shared-inputs' in common
    assert '--llm' not in common and '--ssh-cuda-device' not in common and '--profile' not in common
    assert '--export-video' not in common
    assert not list((ROOT/'assets').glob('*.mp4'))
    for document in (ROOT/name for name in actual if name.endswith('.md')):
        for link in re.findall(r'\]\(([^)]+)\)', document.read_text(encoding='utf-8')):
            if '://' in link or link.startswith('#'):
                continue
            target = (document.parent/link.split('#')[0]).resolve()
            assert target.is_relative_to(ROOT) and target.exists(), (document, link)
    from discovery_protocol import GOAL_PROTOCOLS, goal_schema, goal_parser, goal_prompt
    for protocol in GOAL_PROTOCOLS:
        assert goal_schema(protocol) and callable(goal_parser(protocol)) and goal_prompt(protocol)
    for resolver in (goal_schema, goal_parser, goal_prompt):
        try:
            resolver('diagnostic:legacy')
        except ValueError:
            pass
        else:
            raise AssertionError('Unshipped diagnostic protocol must be rejected')
    patch = ROOT/'unity_patches/public-upstream-24f79883.patch'
    spec = json.loads((ROOT/'examples/unity-base-input-hashes.json').read_text())
    assert hashlib.sha256(patch.read_bytes()).hexdigest() == spec['patch_sha256']
    with tempfile.TemporaryDirectory(prefix='safety-ai2thor-') as temp:
        fixture = Path(temp)/'inventory'
        fixture.mkdir()
        (fixture/'source.py').write_text('pass\n')
        for folder in ('.venv', 'runs', 'build', 'inference-sessions', '.git'):
            (fixture/folder).mkdir()
            (fixture/folder/'generated.md').write_text('[external](missing.md)')
        assert {p.name for p in package_files(fixture)} == {'source.py'}
        (fixture/'unexpected.py').write_text('pass\n')
        assert {p.name for p in package_files(fixture)} == {'source.py', 'unexpected.py'}
        from tools.serve_inference import worker_environment_arguments, mailbox_root
        from inference_rpc import RPCError
        assert mailbox_root(fixture) == fixture/'inference-sessions'
        assert mailbox_root(fixture, str(fixture/'custom mailbox')) == fixture/'custom mailbox'
        try:
            mailbox_root(fixture, 'relative/path')
        except RPCError:
            pass
        else:
            raise AssertionError('Relative mailbox root was accepted')
        environment = worker_environment_arguments({'HOME': '/home/test', 'HF_HOME': '/cache with spaces',
                                                     'CUDA_VISIBLE_DEVICES': '3'})
        assert 'HF_HOME=/cache with spaces' in environment and 'HOME=/home/test' in environment
        assert 'CUDA_VISIBLE_DEVICES=3' in environment and 'HF_HUB_OFFLINE=1' in environment
        assert any(environment[i:i+2] == ['-u', 'HF_HUB_CACHE'] for i in range(len(environment)-1))
        for backend in ('huggingface', 'ssh'):
            for case in ('01', '02', '03'):
                for condition in ('M0', 'M1', 'M2'):
                    output = Path(temp)/(backend+case+condition)
                    args = [sys.executable, '-B', str(ROOT/'tools/run_public_study.py'),
                            '--executable', 'D:/safety-ai2thor/build/safety-ai2thor.exe', '--player-manifest', 'D:/safety-ai2thor/player-manifest.json',
                            '--source', 'D:/safety-ai2thor/upstream', '--output', str(output), '--llm', backend,
                            '--cache-dir', '/model-cache', '--case', case, '--condition', condition, '--print-command']
                    if backend == 'ssh':
                        args += ['--ssh-host', 'example-host', '--ssh-project', '/srv/safety-ai2thor',
                                 '--ssh-python', '/srv/safety-ai2thor/.venv/bin/python', '--ssh-session-root', '/data/mailbox with spaces']
                    cmd = json.loads(subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.PIPE))
                    assert '--planning-record-tables' in cmd and 'gemma12b-int8-rp105-t08' in cmd
                    assert cmd.count('--llm') == 1 and cmd[cmd.index('--llm')+1] == backend
                    assert ('--ssh-host' in cmd) == (backend == 'ssh')
                    if backend == 'ssh':
                        assert cmd[cmd.index('--ssh-session-root')+1] == '/data/mailbox with spaces'
                    assert ('--discovery-patrol-context' in cmd) == (condition == 'M1')
                    assert ('--discovery-end-review' in cmd) == (condition == 'M2')
                    assert not output.exists() and not output.with_name(output.name+'-case.json').exists()
        # Check the actual default and argument errors, without loading a model/player.
        local_args = args[:args.index('--llm')] + args[args.index('--llm')+2:]
        local_args = local_args[:local_args.index('--ssh-host')]
        cmd = json.loads(subprocess.check_output(local_args, cwd=ROOT, text=True, stderr=subprocess.PIPE))
        assert cmd[cmd.index('--llm')+1] == 'huggingface'
        default_cache_args = list(local_args)
        cache_index = default_cache_args.index('--cache-dir')
        del default_cache_args[cache_index:cache_index+2]
        default_command = json.loads(subprocess.check_output(default_cache_args, cwd=ROOT, text=True, stderr=subprocess.PIPE))
        assert '--cache-dir' not in default_command
        assert cmd[cmd.index('--cache-dir')+1] == '/model-cache'
        baseline_args = local_args+['--profile', 'gemma12b-int8']
        baseline = json.loads(subprocess.check_output(baseline_args, cwd=ROOT, text=True, stderr=subprocess.PIPE))
        assert baseline[baseline.index('--profile')+1] == 'gemma12b-int8'
        for invalid in (local_args+['--llm', 'ssh'], local_args+['--ssh-host', 'unexpected'],
                        local_args+['--ssh-session-root', '/unexpected']):
            result = subprocess.run(invalid, cwd=ROOT, capture_output=True, text=True)
            assert result.returncode == 2, result.stderr
        Path(cmd[cmd.index('--output')+1]).mkdir()
        result = subprocess.run(local_args, cwd=ROOT, capture_output=True, text=True)
        assert result.returncode != 0 and 'FileExistsError' in result.stderr
    print(f"PASS: {len(actual)} file hashes; Python syntax; internal Markdown links; public protocols; fixture/config contracts; 18 backend/case/condition commands; default/error/fresh-output checks. No Unity/GPU validation.")

if __name__ == '__main__':
    main()
