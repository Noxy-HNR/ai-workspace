# UI refresh — September 12, 2026

Implemented across every active project interface. Research has no application UI; the old standalone TI editions remain reference copies.

- **AI workspace:** graphite surfaces and teal accents, clearer service/model cards, compact activity disclosure (errors open it), keyboard-accessible navigation, labeled settings, quieter action buttons, responsive grids, improved chat/log surfaces, shorter utility descriptions, and a direct path from the unavailable optimizer to model settings. The overview heading appears only on Services so tools have more working space.
- **Lecture Notes:** shared dark/light styling across the library, recordings, search, study guides, flashcards, diagnostics, and corrections; stronger reading surfaces and page hierarchy; phone layouts; visible focus; accessible flashcard reveal controls; styled download/correction links; three clearly numbered correction steps. Diagnostics no longer marks “unavailable” as healthy.
- **IRViewer:** separate header, image area, telemetry, and keyboard controls; live/stalled indicator; active control highlighting; snapshot feedback; restyled histogram. The sensor image and snapshot exports remain unobscured.
- **TI chemistry suite:** coordinated colors, high-contrast element selection, softer block colors, clearer menu selection, and consistent header/footer accents in PTABLE, ORBITAL, and RYDBERG. Root and release copies rebuilt from chem-suite. No practice mode added.

## Validation

- Lecture suite: 74 tests passed.
- Chemistry suite: 9 tests passed; transfer bundle and SHA-256 manifest verified.
- IRViewer suite: 7 tests passed, including image preservation across scales and histogram states.
- Embedded JavaScript syntax checked for all four web pages.
- Browser QA used synthetic local fixtures: workspace tabs and settings, lecture reading/study views, correction selection and regeneration preview, diagnostics, and 390-pixel layouts. Light mode inspected as well as dark mode.
- Synthetic IR previews and calculator drawing previews visually inspected. Physical camera/calculator testing remains a hardware check.

## Viewing the changes

Refresh the lecture dashboards. Restart an already-running AI dashboard server to reload its cached HTML. Reopen IRViewer to load its new interface. Transfer all four files from `Projects/ti84evo-chem/release/` to update the calculator.

Preview images: `ui-previews/ir-viewer.png`, `ui-previews/ir-stalled.png`, and `../../Projects/ti84evo-chem/previews/` (paths relative to this file).

`preview_workspace.py` and `preview_corrections.py` provide synthetic browser fixtures on ports 18772 and 18771. They are development previews, not replacements for the actual applications. `modernize_ui.py` and `finish_ui.py` were one-time editing scripts and should not be rerun.
