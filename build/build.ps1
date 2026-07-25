# Build PropSim end to end: executable, then installer.
#
#   powershell -ExecutionPolicy Bypass -File build\build.ps1
#
# Run from the repo root on Windows with Python 3.11+ and Inno Setup 6.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "==> checking the rule table before packaging it" -ForegroundColor Cyan
# Ship nothing that fails its own self-check: the rule data is the one asset a
# user makes funding decisions on.
python prop_rules.py
python sim.py --selfcheck

Write-Host "==> PyInstaller" -ForegroundColor Cyan
python -m PyInstaller --noconfirm --clean build\PropSim.spec

$exe = Join-Path $root "dist\PropSim\PropSim.exe"
if (-not (Test-Path $exe)) { throw "build failed: $exe missing" }
$mb = [math]::Round((Get-ChildItem "dist\PropSim" -Recurse |
                     Measure-Object Length -Sum).Sum / 1MB, 1)
Write-Host "    dist\PropSim  ($mb MB)" -ForegroundColor Green

Write-Host "==> Inno Setup" -ForegroundColor Cyan
$iscc = @(
  "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
  "${env:ProgramFiles}\Inno Setup 6\ISCC.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $iscc) {
  Write-Warning "Inno Setup 6 not found — the executable is built, the installer is not."
  Write-Warning "Get it from https://jrsoftware.org/isdl.php and re-run."
  exit 0
}
& $iscc "build\PropSim.iss"
if ($LASTEXITCODE -ne 0) { throw "ISCC failed with $LASTEXITCODE" }

Get-ChildItem "dist_installer\*.exe" | ForEach-Object {
  Write-Host ("    {0}  ({1} MB)" -f $_.Name, [math]::Round($_.Length / 1MB, 1)) `
    -ForegroundColor Green
}
Write-Host "done" -ForegroundColor Green
