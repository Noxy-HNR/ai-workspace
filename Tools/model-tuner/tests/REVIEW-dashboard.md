# Security and Robustness Review: Dashboard Model Tuner Integration

Findings ranked most severe first.

---

## 1. Path Traversal in `Start-ModelTrial` and `/api/models/load` (persist:false)

**Files:** `dashboard-server.ps1` lines 730–751 (`Start-ModelTrial`), lines 1159–1171 (`/api/models/load` route)

**What goes wrong:** The `FileName` parameter is joined with `$ModelsDir` via `Join-Path` and only validated by `Test-Path`. `Join-Path` does not normalize `..` segments, so a filename like `..\..\windows\system32\calc.exe` resolves outside `C:\AI\models`. If such a file exists, `Test-Path` returns true and the server attempts to load it as a GGUF model.

**Concrete scenario:** Attacker sends POST to `/api/models/load` with `{"filename": "..\\..\\secret\\data.gguf", "persist": false, "settings": {}}`. If `C:\AI\secret\data.gguf` exists, llama-server starts with that file.

**Suggested fix:** Add the same traversal check used in `Invoke-TunerAction` (line 471) to `Start-ModelTrial` and `Apply-ModelSettings`:
```powershell
if ($FileName -match '[\/:]' -or $FileName -match '\.\.' -or -not (Test-Path (Join-Path $ModelsDir $FileName))) {
    return @{ success = $false; message = "Unknown model: $FileName" }
}
```

---

## 2. Race Condition: Two Concurrent `start` Requests Can Both Pass the Running Check

**Files:** `dashboard-server.ps1` lines 467–485 (`Invoke-TunerAction` 'start' case), line 449–455 (`Test-TunerRunning`)

**What goes wrong:** `Test-TunerRunning` reads `current.json` and checks the PID. The check and the subsequent `Start-TunerProcess` call are not atomic. Two simultaneous POSTs to `/api/optimizer` with `action=start` can both see "not running" and both launch a tuner process.

**Concrete scenario:** User double-clicks "Start tuning" quickly, or a script fires two requests. Two `python -m modeltuner run ...` processes start, both writing to `state/current.json` and `state/runs/...`, corrupting state and potentially loading two models simultaneously.

**Suggested fix:** Use a file-based mutex (e.g., `state/tuner.lock` created with `New-Item -ItemType File -Force -Path $lock -ErrorAction Stop` which fails if the file exists) around the check-and-start sequence, or have the tuner itself enforce single-instance via a PID file with atomic creation.

---

## 3. XSS in Services Tab: `svcData.error` Rendered Without Escaping

**Files:** `dashboard.html` line 917 (inside `renderServices`)

**What goes wrong:** The error string from the status API is interpolated directly into `innerHTML` without escaping. If any service returns an error containing HTML/JS, it executes in the dashboard origin.

**Concrete scenario:** A compromised or misconfigured service returns `error: "<img src=x onerror=alert(1)>"`. When the Services tab renders, the script runs.

**Suggested fix:** Apply `escapeHtml` (already defined at line 1259) to `svcData.error`:
```javascript
${svcData.error ? `<div class="service-detail-row"><span style="color:var(--accent-red)">Error</span><span style="font-size:11px;color:var(--accent-red)">${escapeHtml(svcData.error)}</span></div>` : ''}
```

---

## 4. `extraArgs` Injection into `llama-server` via Trial Loads

**Files:** `dashboard-server.ps1` lines 738–747 (`Start-ModelTrial`), line 295 (`Start-LlamaNative` calls `Build-LlamaArgs`)

**What goes wrong:** The `extraArgs` field is user-controllable via the Models tab and `/api/models/load` (persist:false). `Start-ModelTrial` appends `--load-mode none --cache-ram 0` but does not sanitize the preceding user-supplied string. If `Build-LlamaArgs` passes `extraArgs` verbatim to `Start-Process -ArgumentList`, a value like `--model ../../malicious.gguf --load-mode none --cache-ram 0` could override the intended model path (last-wins semantics in llama.cpp).

**Concrete scenario:** Attacker saves a model config with `extraArgs: "--model ../../../exfil.gguf"`, then triggers a trial load. The appended safety flags come after, but llama.cpp may honor the earlier `--model`.

**Suggested fix:** In `Start-ModelTrial`, parse `extraArgs` into an array, filter out any flag that sets a model path or other dangerous options (`--model`, `--model-draft`, `--lora`, `--lora-scaled`, `--mmproj`), then re-serialize. Or whitelist known-safe flags.

---

## 5. UI State Mismatch: Headline Shows "Running" While Start Button Is Enabled

**Files:** `dashboard-server.ps1` lines 513–540 (`Get-TunerStatus`), `dashboard.html` lines 1161–1176 (`tunerRefresh`)

**What goes wrong:** `Get-TunerStatus` returns `current.json` content unconditionally. `tunerRefresh` uses `s.running` (from `Test-TunerRunning`, which verifies the PID) for button state, but uses `cur.status` from `current.json` for the headline. If the tuner process dies but `current.json` still says `"status": "running"`, the headline reads "Running: ..." while the Start button is enabled.

**Concrete scenario:** Tuner crashes or is killed externally. `current.json` retains `"status": "running"`. Dashboard polls status: `s.running = false` (PID gone), `cur.status = "running"`. UI shows "Running" headline + enabled Start button — confusing.

**Suggested fix:** In `Get-TunerStatus`, if `Test-TunerRunning` returns false, clear or mark `current.json` as stale (e.g., set `status: "crashed"`), or have `tunerRefresh` derive the headline from `s.running` instead of `cur.status`.

---

## 6. `run_id` Used in `Join-Path` Without Validation in `Get-TunerStatus`

**Files:** `dashboard-server.ps1` line 520 (`Get-TunerStatus`)

**What goes wrong:** `Get-TunerStatus` reads `run_id` from `current.json` (written by the tuner) and uses it directly in `Join-Path $TunerState ("runs\" + $result.current.run_id)`. A malicious or buggy tuner could write `run_id: "..\..\exfil"` causing directory traversal when reading events/report.

**Concrete scenario:** Compromised tuner binary writes `current.json` with `run_id: "..\..\windows\system32"`. Dashboard status poll triggers `Get-ChildItem` on that path, potentially exposing filesystem info.

**Suggested fix:** Validate `run_id` with the same regex used in `Invoke-TunerAction` 'apply' (`^[\w.\-]+$`) before using it in a path.

---

## 7. `plan` Action Does Not Check `Test-TunerRunning`

**Files:** `dashboard-server.ps1` lines 491–496 (`Invoke-TunerAction` 'plan' case)

**What goes wrong:** The 'plan' action starts a tuner process (`plan` subcommand) without checking if a tuning run is already in progress. The plan command may read the model file and hardware state concurrently with a running `run` command, causing contention or corrupted reads.

**Concrete scenario:** User starts a full tuning run, then immediately clicks "Plan on paper". Both processes probe the GGUF and GPU simultaneously.

**Suggested fix:** Add `if (Test-TunerRunning) { return @{ success = $false; message = 'A tuning run is already in progress.' } }` at the start of the 'plan' case, or make the plan command safe for concurrent execution (documented).

---

## 8. Concurrent Trial Loads (`/api/models/load` persist:false) Can Interfere

**Files:** `dashboard-server.ps1` lines 1159–1171 (`/api/models/load`), lines 230–248 (`Stop-LlamaNative`), lines 250–315 (`Start-LlamaNative`)

**What goes wrong:** Two simultaneous trial-load requests both call `Start-ModelTrial` → `Start-LlamaNative` → `Stop-LlamaNative`. The second request may stop the server the first request just started, leaving the first request's trial in a broken state.

**Concrete scenario:** Tuner issues two rapid trial loads (different configs). Request A starts loading model X. Request B stops llama-server (killing A's load), then starts its own load. Request A's subsequent verification fails.

**Suggested fix:** Serialize trial loads with a lock (e.g., a `state/trial.lock` file) or queue them in the tuner backend so only one trial runs at a time.

---

## 9. `run_id` Validation in `apply` Uses Regex That Allows `.` and `-` but Not Path Separators — Adequate

**Files:** `dashboard-server.ps1` line 499

**Status:** Not a defect. The regex `^[\w.\-]+$` prevents path traversal. No fix needed.

---

## 10. Argument Injection in `Start-TunerProcess` — Not Exploitable

**Files:** `dashboard-server.ps1` lines 457–465 (`Start-TunerProcess`), lines 480–484 (`Invoke-TunerAction`)

**Status:** Not a defect. `$TunerArgs` is an array passed to `Start-Process -ArgumentList`, which bypasses the shell. Model, mode, and ctx are validated/whitelisted before being added to the array.

---

---

## Proposed Tests (under `proposed_tests/`)

| Finding | Test File | Currently Fails? |
|---------|-----------|------------------|
| 1. Path traversal in `Start-ModelTrial` | `proposed_tests/test_path_traversal.py` | Yes |
| 2. Race condition on start | `proposed_tests/test_start_race.py` | Yes |
| 3. XSS in services tab | `proposed_tests/test_xss_services.py` | Yes (JS test via Node/Jest) |
| 4. `extraArgs` injection | `proposed_tests/test_extra_args_injection.py` | Yes |
| 5. UI state mismatch | `proposed_tests/test_ui_state_mismatch.py` | Yes |
| 6. `run_id` traversal in `Get-TunerStatus` | `proposed_tests/test_run_id_traversal.py` | Yes |
| 7. `plan` without running check | `proposed_tests/test_plan_concurrent.py` | Yes |
| 8. Concurrent trial loads | `proposed_tests/test_concurrent_trials.py` | Yes |

> Note: Tests use fakes/mocks per ground rules — no real llama-server, tuner, or dashboard process is started.

---

## Summary

Reviewed the Model Tuner integration points in `dashboard-server.ps1` and `dashboard.html`. Found 8 real defects/risks:

- **Critical (2):** Path traversal in trial loads (#1), race condition on start (#2)
- **High (2):** XSS in services tab (#3), `extraArgs` injection (#4)
- **Medium (3):** UI state mismatch (#5), `run_id` traversal in status (#6), `plan` without running check (#7)
- **Low (1):** Concurrent trial loads interference (#8)

All proposed tests are written to `proposed_tests/` and currently fail (they demonstrate the defects). Existing test suite unchanged and still passes.