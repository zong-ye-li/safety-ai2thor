"""Check a clean declared clone; explicit --apply assembles source only, never opens Unity."""
import argparse,hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(project,*args):return subprocess.check_output(['git','-C',str(project),*args],text=True).strip()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--apply',action='store_true');a=p.parse_args();project=a.project.resolve()
 spec=json.loads((ROOT/'examples/unity-base-input-hashes.json').read_bytes());patch=ROOT/'unity_patches/public-upstream-24f79883.patch'
 if project==ROOT or git(project,'rev-parse','HEAD')!=spec['upstream_commit'] or git(project,'status','--porcelain','--untracked-files=all'):raise ValueError('A separate clean clone of the exact declared upstream commit is required')
 if sha(patch)!=spec['patch_sha256']:raise ValueError('Declared patch changed')
 for name,h in spec['base_inputs'].items():
  target=(project/name).resolve()
  if not target.is_relative_to(project) or sha(target)!=h:raise ValueError('Upstream input differs (exact LF checkout required): '+name)
 extensions=list((ROOT/'unity_extensions').rglob('*.cs'))
 for src in extensions:
  if (project/('unity/Assets/Editor' if src.parent.name=='Editor' else 'unity/Assets/Scripts')/src.name).exists():raise FileExistsError(src.name)
 subprocess.run(['git','apply','--check',str(patch)],cwd=project,check=True)
 if not a.apply:print('Exact upstream inputs and patch applicability passed; no files changed');return
 subprocess.run(['git','apply',str(patch)],cwd=project,check=True)
 for src in extensions:
  target=project/('unity/Assets/Editor' if src.parent.name=='Editor' else 'unity/Assets/Scripts')/src.name
  target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('xb') as stream:stream.write(src.read_bytes())
 print('Explicitly assembled source in the supplied clone; no Unity build or native/model qualification performed')
if __name__=='__main__':main()
