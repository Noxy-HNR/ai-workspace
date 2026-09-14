# Project improvements delivered

Scope: the projects reviewed under `C:/AI/Projects`. TI-84 practice mode was excluded as requested. Research remains empty.

## Lecture Notes

- Capture writes and flushes WAV blocks before transcription; failed inference leaves recoverable audio.
- Backlogged transcription stores file offsets, reducing memory usage.
- Recording filenames cannot overwrite existing WAVs.
- Audio-relative JSONL segments support timestamped playback from search results.
- The recordings library streams WAVs with HTTP byte-range seeking.
- Recovery processes audio in chunks, refreshes timestamps, and replaces the matching session section.
- Final diarization and condensation preserve other sessions on the same date.
- Markdown replacement is atomic, with prior revisions and a restoration command.
- Word mirrors are rebuilt without internal session markers.
- Note lookups are confined to the library; same-date navigation anchors are unique.
- Shutdown closes audio resources even if final processing fails.
- The backup list includes recordings that have no transcript yet.

Legacy unmarked notes cannot be assigned to sessions reliably. They are preserved; the first recovery may coexist with them. Later recoveries replace the new marked section.

## TI-84 Evo chemistry

- `chem-suite/` is authoritative; `build_release.py` generates root copies and `release/` with a SHA-256 manifest.
- Legacy standalone editions are clearly labeled and retained.
- Graphical periodic table starts with symbols visible and keeps manual atomic-number lookup.
- Element details include ion charge adjustment, full configuration scrolling, and subshell-by-subshell orbital diagrams.
- Ion configurations implement a bounded textbook isolated-ion model; they do not claim measured ground states for every ion or model ligand fields.
- Orbital animation replay now returns the user's replay choice correctly.
- Checked-in tests cover chemistry invariants, reference cases, table layout/navigation, ion controls, replay, and Rydberg rendering.
- Desktop visual previews are in `previews/`.

Transfer all four `.py` files in `C:/AI/Projects/ti84evo-chem/release/` together.

## IRViewer

- Stale frames show their age and a red border; R releases and reconnects the camera.
- Stale images cannot be saved as fresh snapshots.
- H toggles the raw histogram; L locks contrast.
- CLI and keyboard snapshots share raw/display/both behavior and JSON metadata.
- Failed exports report errors; existing files are protected.
- Frame, bitmap, reader, and capture resources are explicitly released.
- Dependency requirements and a lock file reflect the installed environment.

## Verification

- Lecture Notes: 59 tests passed in the full run, then all 11 durability tests passed after adding the audio-only recovery listing test (60 distinct tests total).
- Chemistry: 9 tests passed; generated release and manifest checks passed.
- IRViewer: 6 hardware-free tests passed.
- Browser: verified source-linked search layout, timestamped playback, and float32 WAV decoding using synthetic silent recordings.
- Visually inspected desktop periodic table and ion detail renderings.
- Temporary browser tabs and the fixture server were closed.

Physical calculator, microphone/GPU recording, and IR camera reconnect behavior still require hardware smoke tests. No actual lecture data was reformatted or recovered during verification. Existing unrelated edits were preserved; no commits were made.
