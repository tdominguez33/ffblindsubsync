$ErrorActionPreference = "Stop"

$guiDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $guiDir
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

$version = (git -C $repoRoot describe --tags --always --dirty 2>$null)
if (-not $version) { $version = "0.0.0+local" }
$version = $version.Trim()

$frozenMod = Join-Path $repoRoot "ffsubsync\_frozen_version.py"
$versionFile = Join-Path $repoRoot "__version__"
Set-Content -Path $frozenMod -Value "FFSUBSYNC_VERSION = `"$version`"`n" -Encoding UTF8
Set-Content -Path $versionFile -Value "$version`n" -Encoding UTF8

$outExe = Join-Path $guiDir "dist\FFblindsubsync.exe"
$legacyDir = Join-Path $guiDir "dist\FFblindsubsync"
$running = Get-Process -Name "FFblindsubsync" -ErrorAction SilentlyContinue
if ($running) {
    throw "FFblindsubsync.exe is currently running (PID $($running.Id -join ', ')). Close it before rebuilding."
}
if (Test-Path $legacyDir) {
    Remove-Item -Recurse -Force $legacyDir
}

try {
    & $python -m PyInstaller --clean -y `
        --distpath (Join-Path $guiDir "dist") `
        --workpath (Join-Path $guiDir "build") `
        (Join-Path $guiDir "desktop.spec")
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    $sizeMb = [math]::Round((Get-Item $outExe).Length / 1MB, 1)
    Write-Host "Built single-file Windows app at: $outExe ($sizeMb MB)"
} finally {
    Remove-Item -Force $frozenMod, $versionFile -ErrorAction SilentlyContinue
}
