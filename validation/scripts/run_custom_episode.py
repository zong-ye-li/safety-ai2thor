"""Launch an owned custom Unity player and main.py; keep the user's Editor intact."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from run_profiles import PROFILES
from evidence_io import atomic_json_write


def finish_owned_launch(report_path, report, processes, primary_error=None):
    """Attempt every owned cleanup; never replace the original launch failure."""
    errors=[]
    states={}
    def retain(role, process, stage, exc):
        errors.append(dict(role=role,pid=process.pid if process is not None else None,
                           stage=stage,error=repr(exc),exception_type=type(exc).__name__))
    for role,process in processes:
        if process is None:
            attempted=report.get(role+'_launch_attempted',False)
            states[role]={'created':None if attempted else False,'exited':not attempted,
                          'creation_status':'unconfirmed' if attempted else 'not_attempted'}
            continue
        state=dict(created=True,pid=process.pid,exited=False,returncode=None)
        states[role]=state
        try:
            running=process.poll() is None
        except BaseException as exc:
            retain(role,process,'poll',exc)
            continue
        if running:
            try:
                process.terminate()
                process.wait(timeout=10)
            except BaseException as exc:
                retain(role,process,'terminate_wait',exc)
                try:
                    if process.poll() is None:
                        process.kill()
                        process.wait(timeout=10)
                except BaseException as secondary:
                    retain(role,process,'kill_wait',secondary)
        try:
            state['returncode']=process.poll()
            state['exited']=state['returncode'] is not None
        except BaseException as exc:
            retain(role,process,'final_poll',exc)
    report.update(owned_processes=states,
                  owned_processes_exited=all(state['exited'] for state in states.values()),
                  cleanup_errors=errors)
    if errors and primary_error is None:
        report['status']='launcher_cleanup_error'
    try:
        atomic_json_write(report_path,report)
    except BaseException as exc:
        retain('launcher',None,'final_receipt_write',exc)
        # A failed durable write cannot certify cleanup. Keep a distinct stderr
        # fallback in the invoking handle; do not rewrite a frozen worker result.
        try:
            print(json.dumps(dict(type='launcher_final_receipt_unconfirmed',report=report)),file=sys.stderr,flush=True)
        except BaseException:
            # Even an unavailable fallback stream cannot replace the primary
            # failure (or the receipt-write failure when no primary exists).
            pass
        if primary_error is None:
            raise
    if errors and primary_error is None:
        raise RuntimeError('Owned launcher cleanup failed; inspect retained cleanup_errors')


def requested_resolution(arguments):
    selector=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    selector.add_argument("--profile",choices=tuple(PROFILES),default="weak-2b")
    selector.add_argument("--width",type=int)
    selector.add_argument("--height",type=int)
    selected,_=selector.parse_known_args(arguments)
    width=selected.width if selected.width is not None else PROFILES[selected.profile]["width"]
    height=selected.height if selected.height is not None else PROFILES[selected.profile]["height"]
    if min(width,height)<1:
        raise ValueError("Player resolution must be positive")
    return width,height


def redacted_arguments(arguments):
    result=[]
    redact_next=False
    for value in arguments:
        if redact_next:
            result.append("[redacted]")
            redact_next=False
        elif value=="--api-key":
            result.append(value)
            redact_next=True
        elif value.startswith("--api-key="):
            result.append("--api-key=[redacted]")
        else:
            result.append(value)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable",type=Path,required=True)
    parser.add_argument("--player-manifest",type=Path,help="Exact full build artifacts manifest; exe alone is not scene identity")
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--port",type=int,default=8201)
    parser.add_argument("--timeout",type=float,default=900)
    parser.add_argument("--driver",choices=["episode","public-nearby-study"],default="episode")
    parser.add_argument("episode_args",nargs=argparse.REMAINDER)
    args=parser.parse_args()
    extra=args.episode_args
    if extra and extra[0]=="--":
        extra=extra[1:]
    width,height=requested_resolution(extra)
    if any(key.split("=",1)[0] in {"--output-dir","--editor-port","--ai2thor-source","--local-executable"} for key in extra):
        parser.error("Player ownership, source, output and port are controlled by this launcher")
    if not args.executable.is_file() or not (args.source/"ai2thor"/"controller.py").is_file():
        parser.error("A compiled player and AI2-THOR source checkout are required")
    manifest_sha=None
    if args.player_manifest is not None:
        records=json.loads(args.player_manifest.read_bytes());base=args.executable.resolve().parent
        expected={}
        for record in records:
            path=(base/record['path']).resolve()
            if not path.is_relative_to(base) or path in expected:
                raise ValueError('Invalid/duplicate artifact path')
            expected[path]=record
            with path.open('rb') as artifact:
                digest=hashlib.file_digest(artifact,'sha256').hexdigest()
            if path.stat().st_size!=record['bytes'] or digest!=record['sha256']:
                raise ValueError('Player artifact changed: '+record['path'])
        actual={p.resolve() for p in base.rglob('*') if p.is_file()}
        if actual!=set(expected):raise ValueError('Player bundle file set changed')
        manifest_sha=hashlib.sha256(args.player_manifest.read_bytes()).hexdigest()
    if not 1<=args.port<=65535 or args.timeout<=0:
        parser.error("Port and timeout must be positive")
    output=args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    def listening():
        try:
            with socket.create_connection(("127.0.0.1",args.port),timeout=0.2):
                return True
        except OSError:
            return False
    if listening():
        raise RuntimeError("Player port occupied; refusing to attach an unknown server")
    flags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0
    player=worker=None
    primary_error=None
    started=time.monotonic()
    report={"type":"owned custom Unity player episode launcher","executable":str(args.executable.resolve()),
            "launcher_pid":os.getpid(),
            "player_manifest_sha256":manifest_sha,
            "executable_sha256":hashlib.sha256(args.executable.read_bytes()).hexdigest(),"source":str(args.source.resolve()),
            "episode_args":redacted_arguments(extra),"output":str(output),"status":"starting",
            "requested_resolution":{"width":width,"height":height},
            "phase":"claiming_launch","launch_attempts":1,
            "worker_launch_attempted":False,"worker_start_confirmed":False,
            "player_launch_attempted":False,"player_start_confirmed":False}
    report_path=output.parent/(output.name+"-launcher.json")
    if report_path.exists():
        raise FileExistsError(report_path)
    # Claim this launch exclusively before creating a child; a competing launcher
    # cannot overwrite ownership evidence in its failure cleanup.
    with report_path.open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(report,stream,indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        with (output.parent/(output.name+"-launcher.log")).open("xb") as log:
            root=Path(__file__).resolve().parents[2]
            driver={"episode":root/"main.py","public-nearby-study":root/"validation/scripts/run_open_discovery.py"}[args.driver]
            worker_environment=dict(os.environ,SAFETY_AI2THOR_PLAYER_SHA256=report["executable_sha256"],
                                    SAFETY_AI2THOR_PLAYER_MANIFEST_SHA256=manifest_sha or '')
            report.update(phase='worker_dispatch',driver=str(driver),
                          driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),
                          launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            atomic_json_write(report_path,report)
            report['worker_launch_attempted']=True
            worker=subprocess.Popen([sys.executable,"-B","-u",str(driver),
                "--ai2thor-source",str(args.source.resolve()),"--editor-port",str(args.port),"--output-dir",str(output),*extra],
                stdout=log,stderr=subprocess.STDOUT,creationflags=flags,env=worker_environment)
            report.update(worker_pid=worker.pid,status="worker_started",phase='worker_port_wait',
                          worker_start_confirmed=True)
            atomic_json_write(report_path,report)
            deadline=time.monotonic()+min(120,args.timeout)
            while not listening():
                if worker.poll() is not None:
                    report['worker_exit_code']=worker.returncode
                    raise RuntimeError("Episode worker exited before binding its local port; read launcher log")
                if time.monotonic()>deadline:
                    raise TimeoutError("Episode worker did not bind its port")
                time.sleep(0.1)
            report.update(phase='player_dispatch',
                          startup_elapsed_seconds=time.monotonic()-started)
            atomic_json_write(report_path,report)
            environment=dict(os.environ,AI2THOR_PORT=str(args.port),AI2THOR_HOST="127.0.0.1",
                             AI2THOR_SERVER_TYPE="WSGI",AI2THOR_SERVER_SIDE_SCREENSHOT="True")
            startup=None
            if os.name=="nt":
                startup=subprocess.STARTUPINFO()
                startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW
                startup.wShowWindow=0
            report['player_launch_attempted']=True
            player=subprocess.Popen([str(args.executable.resolve()),"-batchmode","-screen-width",str(width),
                "-screen-height",str(height),"-logFile",str(output/"Player.log")],
                env=environment,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=flags,startupinfo=startup)
            report.update(worker_pid=worker.pid,player_pid=player.pid,status="player_started",
                          phase='worker_completion_wait',player_start_confirmed=True)
            atomic_json_write(report_path,report)
            code=worker.wait(timeout=args.timeout)
            report["worker_exit_code"]=code
            report["status"]="worker_finished" if code==0 else "worker_failed"
            if code:
                raise RuntimeError("Episode failed; read manifest and launcher log")
    except BaseException as exc:
        primary_error=exc
        report.update(status="launcher_error",error=repr(exc),traceback=traceback.format_exc(),
                      failure_stage=report['phase'],elapsed_seconds=time.monotonic()-started)
        raise
    finally:
        finish_owned_launch(report_path,report,(('player',player),('worker',worker)),primary_error)
    print("Episode saved:",output,flush=True)


if __name__=="__main__":
    main()
