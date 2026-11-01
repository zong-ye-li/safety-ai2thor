"""Portable SSH configuration; no private deployment defaults."""
def resolve_ssh_runtime(args, root=None):
    if args.llm == 'ssh' and not all((args.ssh_host, args.ssh_project, args.ssh_python)):
        raise ValueError('Supply --ssh-host, --ssh-project and --ssh-python explicitly')
    return args
