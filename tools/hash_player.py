"""Write a complete player manifest outside the build directory."""
import argparse
import hashlib
import json
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--executable', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    exe, out = a.executable.resolve(), a.output.resolve()
    if not exe.is_file() or out.is_relative_to(exe.parent):
        p.error('Existing executable and manifest outside its build directory required')
    if out.exists():
        raise FileExistsError(out)
    rows = []
    for f in sorted(exe.parent.rglob('*')):
        if f.is_symlink():
            raise ValueError('Symlinks are not supported in the player bundle')
        if f.is_file():
            with f.open('rb') as s:
                rows.append(dict(path=f.relative_to(exe.parent).as_posix(), bytes=f.stat().st_size,
                                 sha256=hashlib.file_digest(s, 'sha256').hexdigest()))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as s:
        json.dump(rows, s, indent=2)
    print(f'{len(rows)} build files recorded')

if __name__ == '__main__':
    main()
