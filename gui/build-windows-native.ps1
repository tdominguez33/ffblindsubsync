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

$outAppDir = Join-Path $guiDir "dist\FFblindsubsync"
$running = Get-Process -Name "FFblindsubsync" -ErrorAction SilentlyContinue
if ($running) {
    throw "FFblindsubsync.exe is currently running (PID $($running.Id -join ', ')). Close it before rebuilding."
}

$backupBins = @()
foreach ($binName in @("ffmpeg.exe", "ffprobe.exe")) {
    $src = Join-Path $outAppDir $binName
    if (Test-Path $src) {
        $bak = Join-Path $env:TEMP ("ffs_bak_" + $binName)
        Move-Item -Force $src $bak
        $backupBins += @{ Name = $binName; Bak = $bak }
    }
}

try {
    & $python -m PyInstaller --clean -y `
        --distpath (Join-Path $guiDir "dist") `
        --workpath (Join-Path $guiDir "build") `
        (Join-Path $guiDir "desktop.spec")
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    Write-Host "Built portable Windows app at: $(Join-Path $outAppDir 'FFblindsubsync.exe')"
} finally {
    if (Test-Path $outAppDir) {
        foreach ($item in $backupBins) {
            if (Test-Path $item.Bak) {
                Move-Item -Force $item.Bak (Join-Path $outAppDir $item.Name)
            }
        }
    }
    Remove-Item -Force $frozenMod, $versionFile -ErrorAction SilentlyContinue
}
