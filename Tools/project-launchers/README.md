# Desktop project launchers

`install-shortcuts.ps1` creates nine named shortcuts on the Windows user's actual Desktop (including OneDrive redirection). Existing unrelated shortcuts, including Notes, are preserved.

- Chemistry Workbench, Research Library, Spectroscopy Explorer, AI Evaluation Lab: start/reuse Science Studio on port 8780 and open the selected workspace.
- Lecture Notes Dashboard: start/reuse the read/search/corrections dashboard on port 8770. Recording remains available through the existing Notes shortcut.
- IR Camera Viewer: open the native camera viewer with the project's Python environment.
- Local AI Stack: run the existing `C:\AI\start.ps1` launcher for the model, dashboard, and cptr.
- Personal Health Service: run the existing health launcher, including its normal startup/update behavior.
- TI-84 Chemistry - Transfer Files: open `release/` for calculator transfer; these programs run on the TI-84, not desktop Python.

Web launchers wait for readiness, reuse recognized running servers, and serialize startup for shared workspaces. They do not stop processes occupying a port. Background startup logs are in `C:\AI\Logs\project-launchers`. The AI Stack and Health shortcuts retain their existing interactive startup windows. Local model features in Science Studio need a running model endpoint; the Local AI Stack shortcut starts the primary server.

Recreate shortcuts:

```powershell
& C:\AI\Tools\project-launchers\install-shortcuts.ps1
```

Validate paths without opening projects:

```powershell
& C:\AI\Tools\project-launchers\launch-project.ps1 -Project chemistry -ValidateOnly
```

`-NoBrowser` starts/checks a web server without opening a browser and reports errors in the terminal. Supported project keys: chemistry, research, spectra, evaluation, lecture, ir.

All nine shortcuts have matching custom icons in `icons/`, with seven embedded sizes
from 16 to 256 pixels. `update-icons.ps1` changes only icon assignments and refreshes
the Windows shell; it verifies that target paths, arguments, working directories,
and window styles remain unchanged. The shortcut installer also uses these icons.
The source generators are `icons/create_project_icons.py` and `icons/create_ir_icon.py`;
they run with the existing IRViewer Python environment (OpenCV and NumPy).

Verified on 2026-09-13: all shortcut targets and project entrypoints exist; Science Studio server reuse; Lecture Notes cold start and readiness; subsequent Lecture Notes reuse; correct Research workspace deep link in the browser; JavaScript syntax. Camera, microphone recording, model loading, and health synchronization were not activated during shortcut verification.
