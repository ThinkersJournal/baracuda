# Baracuda release preflight: the project-specific checks that run BEFORE
# `releaseant publish` (ReleaseAnt R5, adopted for baracuda). Publishing
# mechanics (order, retries, rate limits, tagging) are ReleaseAnt's; these two
# rules are baracuda's, so they live here. This script uploads nothing and
# changes nothing; it exits non-zero on the first failure.
#
#   pwsh scripts/preflight.ps1
#
# Moved out of `scripts/publish.ps1` (which this replaces). The release tag
# names its stream through ReleaseAnt's `--tag-version`; this script never
# picks one.

[CmdletBinding()]
param()
$ErrorActionPreference = "Stop"

$meta = cargo metadata --no-deps --format-version 1 | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw "cargo metadata failed (exit $LASTEXITCODE)" }

# Publishable: `publish` is null (any registry) or a non-empty allowlist;
# `publish = false` is an empty array in cargo metadata.
$pub = @($meta.packages | Where-Object {
    $null -eq $_.publish -or @($_.publish).Count -gt 0
})
if ($pub.Count -eq 0) { throw "no publishable crates found in workspace metadata" }

# 1. Licence files: every publishable crate must carry LICENSE-MIT and
# LICENSE-APACHE identical to the root copies, because a published crate
# contains only its own directory. CI runs the same script.
$py = Get-Command python3 -ErrorAction SilentlyContinue
if ($null -eq $py) { $py = Get-Command python -ErrorAction Stop }
& $py.Source (Join-Path $PSScriptRoot 'check-crate-licences.py')
if ($LASTEXITCODE -ne 0) { throw "crate licence guard failed (exit $LASTEXITCODE); do not publish" }

# 2. The crates off the lockstep version: each one's MAJOR.MINOR follows the
# `unpopped` it builds against, and its PATCH is ours (the reason is next to
# each one's line in the root Cargo.toml). A NAMED LIST, not a count or a
# single name: add a new exception-line crate here.
$exceptionCrates = @('baracuda-cuda-emit', 'baracuda-cuda-parse')
$lockstep = @($pub | Where-Object { $_.name -notin $exceptionCrates })

# Every other publishable crate shares one lockstep version.
$distinct = @($lockstep | ForEach-Object { $_.version } | Sort-Object -Unique)
if ($distinct.Count -ne 1) {
    Write-Warning "lockstep crates are NOT on one version: $($distinct -join ', ')"
}

# Each exception-line crate's MAJOR.MINOR must match its `unpopped`
# requirement. This stops the run: a published version can never be taken back.
foreach ($name in $exceptionCrates) {
    $crate = @($pub | Where-Object { $_.name -eq $name })
    if ($crate.Count -eq 1) {
        $unp = @($crate[0].dependencies | Where-Object { $_.name -eq 'unpopped' -and $null -eq $_.kind })
        if ($unp.Count -ne 1) { throw "$name has $($unp.Count) normal 'unpopped' dependencies; expected 1" }
        $unpMM = (($unp[0].req -replace '^[^0-9]*', '') -split '\.')[0..1] -join '.'
        $crateMM = ($crate[0].version -split '\.')[0..1] -join '.'
        if ($crateMM -ne $unpMM) {
            throw "$name is $($crate[0].version) but requires unpopped $($unp[0].req): MAJOR.MINOR must match ($crateMM vs $unpMM)"
        }
        Write-Host "Exception line: $name $($crate[0].version) (unpopped $($unp[0].req))"
    }
}

Write-Host "preflight OK: $($pub.Count) publishable crate(s), lockstep version(s): $($distinct -join ', ')"
