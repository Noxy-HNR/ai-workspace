$ErrorActionPreference = 'Stop'
$helper = 'C:\AI\Tools\cptr-safety-profile.ps1'
$root = Join-Path $env:TEMP ('cptr-safety-test-' + [guid]::NewGuid().ToString('N'))
$config = Join-Path $root 'config.toml'
$verify = Join-Path $root 'verify.py'
New-Item -ItemType Directory -Path $root | Out-Null
try {
    $fixture = @'
[server]
host = "127.0.0.1"
port = 18000
[app_config]
"tool_approval.default_builtin_approval" = "allow"
"tool_approval.builtin_tools" = "{}"
"browser.provider" = "browser_use"
"unrelated.setting" = "keep-this"
'@
    Set-Content -LiteralPath $config -Value $fixture -Encoding utf8

    $verifyCode = @'
import json, pathlib, sys, tomllib
p = pathlib.Path(sys.argv[1])
expected = sys.argv[2]
d = tomllib.loads(p.read_text(encoding="utf-8-sig"))
a = d["app_config"]
overrides = json.loads(a["tool_approval.builtin_tools"])
assert a["tool_approval.default_builtin_approval"] == ("review" if expected == "applied" else "allow")
assert a["browser.provider"] == ("local" if expected == "applied" else "browser_use")
assert a["unrelated.setting"] == "keep-this"
if expected == "applied":
    assert all(overrides.get(name) == "review" for name in ("delegate_task", "update_memory", "update_tasks"))
else:
    assert overrides == {}
print(expected + " TOML verification passed")
'@
    Set-Content -LiteralPath $verify -Value $verifyCode -Encoding utf8

    $applyOutput = & $helper -Action Apply -ConfigPath $config
    $applyOutput | Write-Output
    $backupLine = $applyOutput | Where-Object { $_ -match '^Applied cptr safety profile\. Backup: ' } | Select-Object -First 1
    if (-not $backupLine) { throw 'Apply did not report a backup path.' }
    $backup = $backupLine -replace '^Applied cptr safety profile\. Backup: ', ''
    & python $verify $config applied
    if ($LASTEXITCODE -ne 0) { throw 'Applied config did not parse or match the profile.' }

    & $helper -Action Rollback -ConfigPath $config -BackupPath $backup | Write-Output
    & python $verify $config rolledback
    if ($LASTEXITCODE -ne 0) { throw 'Rollback did not preserve the original settings.' }

    Write-Output 'PASS: apply, TOML parse, exact policy values, targeted rollback, and unrelated-setting preservation.'
} finally {
    if (Test-Path -LiteralPath $root) { Remove-Item -LiteralPath $root -Recurse -Force }
}
