# C:\AI Local AI Dashboard

This dashboard is designed for the current C:\AI stack:

- llama-server runs natively on Windows (no Docker/WSL) on `http://127.0.0.1:10000`, using the CUDA build in `Tools\llama-native\bin`.
- cptr runs natively on Windows on `http://127.0.0.1:8000`.
- The dashboard server runs on `http://127.0.0.1:9090`.

Running llama-server natively instead of in Docker Desktop/WSL2 removes
several GB of idle RAM overhead and starts noticeably faster.

## Start

```powershell
cd C:\AI
.\start.ps1
```

This starts llama-server (with the active model from `Config\models.json`),
the dashboard server, and cptr, then opens the dashboard at `http://127.0.0.1:9090`.

To just run the dashboard server on its own:

```powershell
cd C:\AI
powershell -ExecutionPolicy Bypass -File .\dashboard-server.ps1
```

## Models

Available models are scanned from `C:\AI\models\*.gguf`. Per-model startup
parameters (context size, GPU layers, batch size, etc.) are stored in
`Config\models.json` and edited from the dashboard's Models tab. Loading a
model there stops any running llama-server process and starts a new one with
that model's saved settings.

The dashboard API returns a `services` object with the exact service names
used by the UI, including `llama-server`, so the old `undefined.llama-server`
error is avoided.

The dashboard does not modify your `.env`, model files, or cptr installation.

## Known gap

The Auto-Optimizer tab still assumes Docker (it benchmarks configs by
spinning up temporary containers) and currently fails with a clear message
rather than running. It needs a native rewrite that launches temporary
llama-server.exe processes on an alternate port instead. Manual tuning via
the Models tab settings panel works normally in the meantime.
