# Bot queue

Runs build jobs through OpenCode models unattended (for example while Claude is rate-limited). Each job is a folder in
`state/botqueue/jobs/<name>/` with `job.json` (models to try in order, check command, timeout) and `TASK.md`. The runner
seeds a fresh copy of the project for every attempt, runs `opencode run --auto` in it, runs the job's check (pytest by
default), and falls back to the next model on failure. Nothing a bot writes touches the project until someone reviews
it and copies it in.

```bash
python devtools/jobs_batch1.py          # create a batch of jobs (idempotent)
python devtools/botqueue.py run         # start (use pythonw via WMI to keep it running after the session ends)
python devtools/botqueue.py status      # per-job state, model, attempts, test results
```

Stop it by creating `state/botqueue/STOP`: running attempts finish, nothing new starts. It also exits after 14 hours.

Local-model jobs (`llama-local/...`) run one at a time, use whatever model is loaded unless the job names one, and wait
while a Model Tuner run is using llama-server or the laptop is on battery. Cloud jobs run three at a time; launches are
spaced 15 s apart because OpenCode's standalone servers collide when started together.
