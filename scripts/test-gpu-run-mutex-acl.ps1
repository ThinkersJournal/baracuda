#Requires -Version 5.1
<#
.SYNOPSIS
  Test: Global\gpu-run is created so EVERY authenticated account can open it.

.DESCRIPTION
  gpu-run.ps1 used to create the machine-wide mutex with the default ACL, which
  only the creating account can open. The runners are separate accounts
  (ghr-baracuda, ghr-fuel, interactive lanes), so whichever created it first
  locked the others out: "Access to the path 'Global\gpu-run' is denied"
  (device-tests run 42386434774, 2026-11-07 19:49Z: baracuda-kernels ran 0 tests).

  Runs the REAL gpu-run.ps1 (a temp copy with only the mutex name replaced by a unique
  test name, so it never touches the real lock or waits for the GPU) around a child
  that reads the mutex's ACL back, and asserts an Allow ACE for Authenticated Users
  carrying Synchronize and Modify. Run under both pwsh and powershell (5.1):
    pwsh -NoProfile -File scripts/test-gpu-run-mutex-acl.ps1
    powershell -NoProfile -File scripts/test-gpu-run-mutex-acl.ps1
  Exit 0 = pass, 1 = fail. Limit: this reads the ACL back; it does not log in as a
  second account.
#>
$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
$name = 'Global\gpu-run-acltest-' + [guid]::NewGuid().ToString('N')
$tmp  = Join-Path ([IO.Path]::GetTempPath()) ('gpu-run-acltest-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tmp | Out-Null
try {
    # Copy gpu-run.ps1 and its helper (if present) next to each other, swap ONLY the name.
    foreach ($f in 'gpu-run.ps1', 'gpu-run-mutex.ps1') {
        $src = Join-Path $here $f
        if (Test-Path -LiteralPath $src) {
            $text = [IO.File]::ReadAllText($src)
            $text = $text.Replace("'Global\gpu-run'", "'$name'")
            [IO.File]::WriteAllText((Join-Path $tmp $f), $text, (New-Object System.Text.UTF8Encoding($true)))
        }
    }
    # Child: open the mutex gpu-run is holding and print its Allow ACEs as "sid rights".
    $child = Join-Path $tmp 'child.ps1'
    $childLines = @(
        'param([string]$N)',
        '$rights = [System.Security.AccessControl.MutexRights]::ReadPermissions -bor [System.Security.AccessControl.MutexRights]::Synchronize',
        'if ($PSVersionTable.PSEdition -eq ''Core'') {',
        '    $m = [System.Threading.MutexAcl]::OpenExisting($N, $rights)',
        '    $acl = [System.Threading.ThreadingAclExtensions]::GetAccessControl($m)',
        '} else {',
        '    $m = [System.Threading.Mutex]::OpenExisting($N, $rights)',
        '    $acl = $m.GetAccessControl()',
        '}',
        'foreach ($r in $acl.GetAccessRules($true, $false, [System.Security.Principal.SecurityIdentifier])) {',
        '    ''ACE {0} {1} {2}'' -f $r.AccessControlType, $r.IdentityReference.Value, [int]$r.MutexRights',
        '}'
    )
    Set-Content -LiteralPath $child -Value $childLines -Encoding UTF8
    $exe = (Get-Process -Id $PID).Path
    $out = & $exe -NoProfile -File (Join-Path $tmp 'gpu-run.ps1') -Project acltest -- $exe -NoProfile -File $child $name 2>&1 | Out-String
    $need = 0x100000 -bor 0x1   # Synchronize | Modify
    $ok = $false
    foreach ($line in ($out -split "`r?`n")) {
        if ($line -match '^ACE Allow S-1-5-11 (\d+)$') {
            if (([int]$Matches[1] -band $need) -eq $need) { $ok = $true }
        }
    }
    $ed = "$($PSVersionTable.PSEdition) $($PSVersionTable.PSVersion)"
    if ($ok) { "PASS ($ed): Authenticated Users has Synchronize|Modify on the gpu-run mutex"; exit 0 }
    "FAIL ($ed): no Allow ACE for S-1-5-11 with Synchronize|Modify. Child output:"
    $out
    exit 1
}
finally { Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue }
