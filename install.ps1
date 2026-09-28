# Sahaay installer for Windows on Snapdragon (HP OmniBook, EliteBook, any Snapdragon X / X2 PC).
# Run from an unzipped release or a git checkout:   powershell -ExecutionPolicy Bypass -File install.ps1
# Installs ARM64 Python 3.12 if missing, creates a virtual environment, installs dependencies,
# downloads the models once, and puts a Start Menu shortcut in place. Afterwards Sahaay is fully offline.
param([switch]$NoGenAI, [switch]$NoShortcut)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

function Find-Py312 {
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        $arch = & py -3.12 -c "import platform;print(platform.machine())" 2>$null
        if ($arch -eq "ARM64") { return "py -3.12" }
    }
    return $null
}

Write-Host "== Sahaay installer ==" -ForegroundColor Green
if ($env:PROCESSOR_ARCHITECTURE -ne "ARM64") {
    Write-Host "This PC is not ARM64. Sahaay needs a Snapdragon-powered PC for the Hexagon NPU." -ForegroundColor Yellow
}
$py = Find-Py312
if (-not $py) {
    Write-Host "Installing Python 3.12 (ARM64) via winget..."
    winget install --id Python.Python.3.12 --architecture arm64 --exact --silent --accept-source-agreements --accept-package-agreements
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
    $py = Find-Py312
    if (-not $py) { throw "Python 3.12 ARM64 not found after install. Install it from python.org and re-run." }
}
Write-Host "Python: $py"

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..."
    Invoke-Expression "$py -m venv .venv"
}
$venvPy = Join-Path $here ".venv\Scripts\python.exe"
& $venvPy -m pip install --upgrade pip --quiet
Write-Host "Installing dependencies (this can take a few minutes)..."
& $venvPy -m pip install -r requirements.txt --quiet

Write-Host "Downloading models (one time)..."
$args = @()
if ($NoGenAI) { $args += "--no-genai" }
& $venvPy tools\get_models.py @args

if (-not $NoShortcut) {
    $startMenu = [Environment]::GetFolderPath("Programs")
    $lnk = Join-Path $startMenu "Sahaay.lnk"
    $ws = New-Object -ComObject WScript.Shell
    $s = $ws.CreateShortcut($lnk)
    $s.TargetPath = (Join-Path $here "run.bat")
    $s.WorkingDirectory = $here
    $s.IconLocation = (Join-Path $here "docs\media\sahaay.ico")
    $s.Description = "Sahaay: hands-free and eyes-free Windows, on the NPU"
    $s.Save()
    Write-Host "Start Menu shortcut created: $lnk"
}
Write-Host "Done. Launch with run.bat or the Start Menu shortcut. Say 'what's on my screen' to begin." -ForegroundColor Green
