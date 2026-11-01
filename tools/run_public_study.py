"""Run a fresh public case with local Hugging Face or optional SSH inference."""
import argparse
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ('executable', 'player-manifest', 'source', 'output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--cache-dir',
                        help='Hugging Face Hub cache root on the inference host; omit for its HF environment/default cache. Not a snapshot or --local-dir folder.')
    parser.add_argument('--llm', choices=('huggingface', 'ssh'), default='huggingface',
                        help='Local GPU inference by default; SSH is optional')
    parser.add_argument('--profile', choices=('gemma12b-int8', 'gemma12b-int8-rp105-t08'),
                        default='gemma12b-int8-rp105-t08',
                        help='Latest engineering candidate by default; select gemma12b-int8 for the baseline decoder settings')
    for key in ('ssh-host', 'ssh-project', 'ssh-python'):
        parser.add_argument('--' + key)
    parser.add_argument('--ssh-cuda-device', default='0')
    parser.add_argument('--ssh-session-root', help='Absolute writable remote mailbox root; default: <ssh-project>/inference-sessions')
    parser.add_argument('--port', type=int, default=8201)
    parser.add_argument('--condition', choices=('M0', 'M1', 'M2'), default='M0')
    parser.add_argument('--case', choices=('01', '02', '03'), default='01')
    parser.add_argument('--print-command', action='store_true')
    args = parser.parse_args()
    ssh_values = (args.ssh_host, args.ssh_project, args.ssh_python)
    if args.llm == 'ssh' and not all(ssh_values):
        parser.error('--llm ssh requires --ssh-host, --ssh-project and --ssh-python')
    if args.llm != 'ssh' and (any(value is not None for value in ssh_values) or args.ssh_session_root is not None):
        parser.error('SSH connection arguments require --llm ssh')
    if args.ssh_session_root is not None and not PurePosixPath(args.ssh_session_root).is_absolute():
        parser.error('--ssh-session-root must be an absolute path on the remote host')
    cases = json.loads((ROOT / 'examples/cases.json').read_bytes())
    cases['cases'] = [case for case in cases['cases'] if case['id'] == args.case]
    output = Path(args.output).resolve()
    casefile = output.with_name(output.name + '-case.json')
    if output.exists() or casefile.exists():
        raise FileExistsError('Fresh output and case receipt required; unknown launch effects must not be resent')
    common = json.loads((ROOT / 'examples/common-arguments.json').read_bytes())
    command = [sys.executable, '-B', str(ROOT / 'validation/scripts/run_custom_episode.py'),
               '--driver', 'public-nearby-study', '--executable', args.executable,
               '--player-manifest', args.player_manifest, '--source', args.source,
               '--output', str(output), '--port', str(args.port), '--timeout', '43200',
               '--', *common, '--llm', args.llm, '--profile', args.profile,
               '--cases', str(casefile)]
    if args.cache_dir:
        command.extend(['--cache-dir', args.cache_dir])
    if args.llm == 'ssh':
        command.extend(['--ssh-host', args.ssh_host, '--ssh-project', args.ssh_project,
                        '--ssh-python', args.ssh_python, '--ssh-cuda-device', args.ssh_cuda_device])
        if args.ssh_session_root:
            command.extend(['--ssh-session-root', args.ssh_session_root])
    if args.condition == 'M1':
        command.append('--discovery-patrol-context')
    if args.condition == 'M2':
        command.append('--discovery-end-review')
    location = 'local machine' if args.llm == 'huggingface' else 'SSH inference host'
    cache = args.cache_dir or 'HF_HUB_CACHE, then HF_HOME/hub, then the Hugging Face user cache'
    print(f'Model preparation: see README.md, "Download the model". Backend: {args.llm}; cache on {location}: {cache}.', file=sys.stderr)
    print(f'Profile: {args.profile}. Historical examples do not validate the exported latest configuration.', file=sys.stderr)
    print('Download the pinned revision before launching; --cache-dir is the Hub cache root, not its snapshots subdirectory.', file=sys.stderr)
    if args.print_command:
        print(json.dumps(command, indent=2))
        return
    casefile.parent.mkdir(parents=True, exist_ok=True)
    with casefile.open('x', encoding='utf-8') as stream:
        json.dump(cases, stream, indent=2)
    subprocess.run(command, check=True, cwd=ROOT)


if __name__ == '__main__':
    main()
