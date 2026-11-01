# Build the AI2-THOR source extension

The extension adds held-object appearance controls, owned simulation time and scene-light inspection. A small patch makes the upstream base controller partial, protects the owned time scale, provides batch camera rendering and supplies the necessary Standard Assets assembly references. Partial controller methods must compile into the same assembly as AI2-THOR; a standalone plugin DLL cannot add them to an already compiled stock player. This source overlay avoids redistributing an entire modified simulator or Unity assets.

## Assemble a separate upstream clone

The commands use `D:/safety-ai2thor` as an example workspace root. Replace it consistently with your chosen location; the upstream source is under `upstream`, the player output under `build`, and the executable is named `safety-ai2thor.exe`.

Use a clean clone at commit `24f79883b4889e3f0e6f4ae301808b9025872dfc`. Its `unity/ProjectSettings/ProjectVersion.txt` specifies **2020.3.25f1**. Keep that Editor version; no upgrade is part of this recipe. Use a short absolute checkout path: upstream contains long asset filenames, and a deeply nested Windows path can leave checkout incomplete. Keep LF line endings so the declared base hashes match:

```powershell
git init D:/safety-ai2thor/upstream
git -C D:/safety-ai2thor/upstream config core.autocrlf false
git -C D:/safety-ai2thor/upstream config core.longpaths true
git -C D:/safety-ai2thor/upstream remote add origin https://github.com/allenai/ai2thor.git
git -C D:/safety-ai2thor/upstream fetch --depth 1 origin 24f79883b4889e3f0e6f4ae301808b9025872dfc
git -C D:/safety-ai2thor/upstream checkout --detach FETCH_HEAD
.venv/Scripts/python -B tools/prepare_unity_clone.py --project D:/safety-ai2thor/upstream
.venv/Scripts/python -B tools/prepare_unity_clone.py --project D:/safety-ai2thor/upstream --apply
```

Stop if checkout reports errors; the first helper must pass before applying anything. The first helper invocation checks without modifying files. The second checks the exact commit, clean worktree, base and patch SHA values, then applies the patch and copies the five runtime extensions plus the Editor build helper. **It refuses changed or already assembled sources.** `examples/unity-base-input-hashes.json` pins the base files, including the Editor version file.

## Build only the two example scenes

Install Unity **2020.3.25f1** with Windows Mono build support and ensure that this exact Editor can obtain a valid Unity license on the build machine.

```powershell
$env:SAFETY_AI2THOR_EXE = 'D:/safety-ai2thor/build/safety-ai2thor.exe'
$build = Start-Process -FilePath 'C:/Program Files/Unity/Hub/Editor/2020.3.25f1/Editor/Unity.exe' -ArgumentList @('-batchmode', '-quit', '-projectPath', 'D:/safety-ai2thor/upstream/unity', '-executeMethod', 'SafetyAi2ThorBuild.Build', '-logFile', 'D:/safety-ai2thor/build.log') -WindowStyle Hidden -Wait -PassThru
if ($build.ExitCode -ne 0) { throw "Unity build failed with exit code $($build.ExitCode); inspect D:/safety-ai2thor/build.log" }
if (-not (Test-Path D:/safety-ai2thor/build/safety-ai2thor.exe)) { throw 'Build produced no player; inspect the build log' }
.venv/Scripts/python -B tools/hash_player.py --executable D:/safety-ai2thor/build/safety-ai2thor.exe --output D:/safety-ai2thor/player-manifest.json
```

`D:/safety-ai2thor/build` must not exist before building. The Editor helper selects FloorPlan25_physics and FloorPlan26_physics and sets Mono in this separate clone. A successful process launch alone does not mean the build succeeded: **inspect the exit code, build log, and produced executable**. Hash the complete fresh output directory. Keep its manifest outside that directory, since the launcher verifies the full set of files and all content hashes.

After a successful build, run the README command to verify actual RGB, fixture placement, held rendering, and owned-clock pause checks before interpreting model outcomes. The runner refuses failed setup and does not silently fall back to legacy time. Compilation and player/model execution remain separate validation steps. See the release-check table in README.md for the tested scope and outstanding build/runtime gates.