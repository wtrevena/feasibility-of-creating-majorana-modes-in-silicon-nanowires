# ---------------------------------------------------------------------------
# reproduce.ps1 -- one-command reproduction of every figure, number, and check
# (Windows; see reproduce.sh for the POSIX version).
#
# Runs the test suites, then every analysis script in dependency order, then
# tools/manifest.py (which fails if any headline number quoted in the paper
# no longer matches the ledger output/data/key_numbers.json).
#
# NOTE on runtime: all heavy scripts checkpoint per-section to
# output/data/*.json (keyed by parameter signatures) and resume, so a full
# pass over the committed checkpoints is mostly cache reads (minutes).
# Delete a script's checkpoint JSON in output/data/ to force genuine
# recomputation of that section (hours for the largest scans).
#
# Override the interpreter with:  $env:PYTHON = "C:\path\to\python.exe"
# ---------------------------------------------------------------------------
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if ($env:PYTHON) { $py = $env:PYTHON }
elseif (Test-Path ".\venv\Scripts\python.exe") { $py = ".\venv\Scripts\python.exe" }
else { $py = "python" }
Write-Host "Using interpreter: $py"

function Invoke-Step {
    param([string[]]$CmdArgs)
    Write-Host ""
    Write-Host "== $($CmdArgs -join ' ') =="
    & $py @CmdArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $($CmdArgs -join ' ')" -ForegroundColor Red
        exit 1
    }
}

# 1. test suites (every exact claim is pinned here)
Invoke-Step @("tests.py")
Invoke-Step @("compat/test_shim.py")

# 2. analysis scripts, dependency order
Invoke-Step @("run_analysis.py", "--fig", "all")   # figs 1-11 + key_numbers tags
Invoke-Step @("transport.py")                      # fig12, transport_numbers.json
Invoke-Step @("transport_valley.py")               # valley/Anderson transport
Invoke-Step @("qp_poisoning.py")                   # QP poisoning bounds
Invoke-Step @("convergence.py")                    # fig13 + convergence tables
Invoke-Step @("realism.py")                        # dynamic self-energy / Dynes / disorder
Invoke-Step @("orbital.py")                        # Peierls orbital control
Invoke-Step @("pairing_mix.py")                    # pairing-channel mixing control
Invoke-Step @("morphology.py")                     # miscut / terrace / bunching ensembles
Invoke-Step @("kp6_holes.py")                      # six-band k.p fin model ([100])
Invoke-Step @("kp6_110.py")                        # [110] channel rotation
Invoke-Step @("poisson2d.py")                      # tri-gate Poisson + SCF harness
Invoke-Step @("kp6_sc.py")                         # self-consistent six-band + Poisson
Invoke-Step @("platform110.py")                    # platform points with [110] parameters

# 3. any round-6 add-on scripts present
Get-ChildItem -File -Filter "r6_*.py" | Sort-Object Name | ForEach-Object {
    Invoke-Step @($_.Name)
}

# 4. integrity manifest (fails on any paper-number / ledger mismatch)
Invoke-Step @("tools/manifest.py")

Write-Host ""
Write-Host "ALL REPRODUCED"
