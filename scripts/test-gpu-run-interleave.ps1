#Requires -Version 5.1
<#
.SYNOPSIS
  Test: with the gpu-run lock taken per test binary, two concurrent holders alternate.

.DESCRIPTION
  `xtask test-gpu` used to hold Global\gpu-run across the whole `cargo test`, so a 7-12 h CI job
  starved every other lane's short GPU run (Fuel waited from ~01:43Z, 2026-11-12). It now runs
  `scripts/gpu-run.ps1 -Project baracuda -- <test exe>` once per test binary.

  This test runs the REAL gpu-run.ps1 (a temp copy with only the mutex name replaced by a unique
  test name, so it never touches the real lock or waits for the GPU) in two shapes, each with two
  lanes A and B (B starts 1 s after A):

    per-binary  A and B each make 4 gpu-run calls of 2 s, one after another (what xtask now does).
                Asserts: the holds never overlap (mutual exclusion), AND B's first hold starts before
                A's last hold starts (the lock changed hands between A's binaries).
    whole-run   A makes ONE gpu-run call whose child runs 4 x 2 s (what xtask did). Asserts the
                control: B's first hold starts only after A's last hold ended. This proves the test
                can see starvation, so the per-binary result is not vacuous.

  Run under both pwsh and powershell (5.1):
    pwsh -NoProfile -File scripts/test-gpu-run-interleave.ps1
    powershell -NoProfile -File scripts/test-gpu-run-interleave.ps1
  Exit 0 = pass, 1 = fail. Limit: it tests the protocol with `sleep` children, not a GPU; and that
  xtask makes one gpu-run call per binary is asserted by xtask's unit test
  `one_lock_hold_per_binary_in_order`, not here.
#>
$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
$exe  = (Get-Process -Id $PID).Path
$name = 'Global\gpu-run-interleave-' + [guid]::NewGuid().ToString('N')
$tmp  = Join-Path ([IO.Path]::GetTempPath()) ('gpu-run-interleave-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tmp | Out-Null
$holdMs = 2000
$calls  = 4
try {
    foreach ($f in 'gpu-run.ps1', 'gpu-run-mutex.ps1') {
        $text = [IO.File]::ReadAllText((Join-Path $here $f))
        $text = $text.Replace("'Global\gpu-run'", "'$name'")
        # The real gpu-run keeps its holder metadata in %TEMP%\gpu-run.lock: a test must not overwrite or
        # delete the lockfile of a real run in progress, so point the copy's lockfile and log into $tmp.
        $text = $text.Replace('Join-Path $env:TEMP ''gpu-run.lock''', "Join-Path '$tmp' 'gpu-run.lock'")
        $text = $text.Replace('Join-Path $env:TEMP ''gpu-run.log''', "Join-Path '$tmp' 'gpu-run.log'")
        [IO.File]::WriteAllText((Join-Path $tmp $f), $text, (New-Object System.Text.UTF8Encoding($true)))
    }
    # The "test binary": records its start and end ticks around a hold.
    $child = Join-Path $tmp 'child.ps1'
    Set-Content -LiteralPath $child -Encoding UTF8 -Value @(
        'param([string]$Tag, [string]$Log, [int]$Ms, [int]$N)',
        'for ($i = 0; $i -lt $N; $i++) {',
        '    Add-Content -LiteralPath $Log -Value ("start $Tag " + [DateTime]::UtcNow.Ticks)',
        '    Start-Sleep -Milliseconds $Ms',
        '    Add-Content -LiteralPath $Log -Value ("end $Tag " + [DateTime]::UtcNow.Ticks)',
        '}'
    )
    # A lane: either $Calls gpu-run calls of one hold each, or one call of $Calls holds.
    $lane = Join-Path $tmp 'lane.ps1'
    Set-Content -LiteralPath $lane -Encoding UTF8 -Value @(
        'param([string]$Tag, [string]$Log, [string]$Mode, [string]$Exe, [string]$Tmp, [int]$Ms, [int]$Calls)',
        '$run = Join-Path $Tmp ''gpu-run.ps1''',
        '$child = Join-Path $Tmp ''child.ps1''',
        'if ($Mode -eq ''per-binary'') {',
        '    for ($i = 0; $i -lt $Calls; $i++) {',
        '        & $Exe -NoProfile -File $run -Project $Tag -- $Exe -NoProfile -File $child $Tag $Log $Ms 1',
        '    }',
        '} else {',
        '    & $Exe -NoProfile -File $run -Project $Tag -- $Exe -NoProfile -File $child $Tag $Log $Ms $Calls',
        '}'
    )

    function Invoke-Scenario([string] $mode) {
        $log = Join-Path $tmp ("log-$mode.txt")
        New-Item -ItemType File -Path $log | Out-Null
        # Start-Process joins the array with spaces and does not quote: quote every path (pwsh lives under 'Program Files').
        $argsFor = { param($tag) @('-NoProfile', '-File', "`"$lane`"", $tag, "`"$log`"", $mode, "`"$exe`"", "`"$tmp`"", "$holdMs", "$calls") }
        $ea = Join-Path $tmp "err-A-$mode.txt"; $eb = Join-Path $tmp "err-B-$mode.txt"
        $a = Start-Process -FilePath $exe -ArgumentList (& $argsFor 'A') -PassThru -WindowStyle Hidden -RedirectStandardError $ea -RedirectStandardOutput (Join-Path $tmp "out-A-$mode.txt")
        Start-Sleep -Milliseconds 1000
        $b = Start-Process -FilePath $exe -ArgumentList (& $argsFor 'B') -PassThru -WindowStyle Hidden -RedirectStandardError $eb -RedirectStandardOutput (Join-Path $tmp "out-B-$mode.txt")
        $null = $a.WaitForExit(120000); $null = $b.WaitForExit(120000)
        foreach ($e in $ea, $eb) { if ((Test-Path $e) -and (Get-Item $e).Length -gt 0) { Write-Host "lane stderr ($e):"; Get-Content -LiteralPath $e | Select-Object -First 8 | ForEach-Object { Write-Host "  $_" } } }
        # Parse "start|end TAG ticks" into hold intervals per tag, in order.
        $open = @{}; $holds = @()
        foreach ($l in (Get-Content -LiteralPath $log)) {
            $p = $l -split ' '
            if ($p.Count -ne 3) { continue }
            if ($p[0] -eq 'start') { $open[$p[1]] = [long]$p[2] }
            elseif ($p[0] -eq 'end') { $holds += [pscustomobject]@{ Tag = $p[1]; Start = $open[$p[1]]; End = [long]$p[2] } }
        }
        return ,@($holds | Sort-Object Start)
    }

    $fail = @()

    $per = Invoke-Scenario 'per-binary'
    if ($per.Count -ne 2 * $calls) { $fail += "per-binary: expected $($calls * 2) holds, saw $($per.Count)" }
    for ($i = 1; $i -lt $per.Count; $i++) {
        if ($per[$i].Start -lt $per[$i - 1].End) { $fail += "per-binary: holds $($i - 1) and $i overlap (mutual exclusion broken)" }
    }
    $aLastStart = ($per | Where-Object Tag -eq 'A' | Select-Object -Last 1).Start
    $bFirstStart = ($per | Where-Object Tag -eq 'B' | Select-Object -First 1).Start
    if (-not ($bFirstStart -lt $aLastStart)) { $fail += 'per-binary: B did not get in before A''s last binary (no alternation)' }
    $order = ($per | ForEach-Object { $_.Tag }) -join ''
    Write-Host "per-binary hold order: $order"

    $whole = Invoke-Scenario 'whole-run'
    $aLastEnd = ($whole | Where-Object Tag -eq 'A' | Select-Object -Last 1).End
    $bFirstStart2 = ($whole | Where-Object Tag -eq 'B' | Select-Object -First 1).Start
    if (-not ($bFirstStart2 -ge $aLastEnd)) { $fail += 'whole-run control: B ran before A finished (the test cannot see starvation)' }
    $order2 = ($whole | ForEach-Object { $_.Tag }) -join ''
    Write-Host "whole-run hold order:  $order2 (control: B waits for all of A)"

    if ($fail.Count -gt 0) { $fail | ForEach-Object { Write-Host "FAIL: $_" }; exit 1 }
    Write-Host 'PASS: per-binary holds alternate and never overlap; whole-run control starves B'
    exit 0
}
finally {
    Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
}
