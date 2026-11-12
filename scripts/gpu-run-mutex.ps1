# Dot-sourced by gpu-run.ps1. Defines New-GpuRunMutex.
#
# WHY this exists: `New-Object System.Threading.Mutex($false, 'Global\gpu-run')` creates the
# mutex with the DEFAULT ACL (SYSTEM, Administrators, and the creating account only). The
# runners are separate accounts (ghr-baracuda, ghr-fuel, interactive lanes), so whichever
# account created it first locked every other one out with "Access to the path
# 'Global\gpu-run' is denied" (device-tests run 42386434774, 2026-11-07 19:49Z: the
# baracuda-kernels step built for 2h35m, then ran 0 tests). The name stays the protocol
# constant; only the ACL changes: Authenticated Users get Synchronize|Modify, which is all
# WaitOne/ReleaseMutex need.
#
# The first creator's ACL wins: a mutex created by an OLDER gpu-run (version < 4) still has
# the default ACL until its holder exits, and every copy of gpu-run (baracuda, fuel,
# vulkane) must carry this change.
function New-GpuRunMutex([string] $Name) {
    $authenticated = New-Object System.Security.Principal.SecurityIdentifier(
        [System.Security.Principal.WellKnownSidType]::AuthenticatedUserSid, $null)
    $rights = [System.Security.AccessControl.MutexRights]::Synchronize -bor `
              [System.Security.AccessControl.MutexRights]::Modify
    $security = New-Object System.Security.AccessControl.MutexSecurity
    $security.AddAccessRule((New-Object System.Security.AccessControl.MutexAccessRule(
        $authenticated, $rights, [System.Security.AccessControl.AccessControlType]::Allow)))
    $core = ($PSVersionTable.PSEdition -eq 'Core')
    # OPEN FIRST, with only the rights WaitOne/ReleaseMutex need. Creating with a security descriptor
    # asks for MUTEX_ALL_ACCESS even when the mutex already exists, and the ACL above grants only
    # Synchronize|Modify, so every SECOND process (same account or not) got "access denied" instead of
    # waiting its turn: two concurrent gpu-run calls could not both run. Found 2026-11-12 by
    # scripts/test-gpu-run-interleave.ps1; it is the likely cause of Fuel being unable to take
    # Global\gpu-run while a baracuda device-tests run held it. Create only when it does not exist.
    for ($attempt = 0; $attempt -lt 8; $attempt++) {
        try {
            if ($core) { return [System.Threading.MutexAcl]::OpenExisting($Name, $rights) }
            return [System.Threading.Mutex]::OpenExisting($Name, $rights)
        }
        catch [System.Threading.WaitHandleCannotBeOpenedException] { }   # not there yet: create it
        catch [System.UnauthorizedAccessException] {
            throw ("gpu-run: cannot open '$Name' (access denied). It already exists with an ACL that " +
                   "excludes this account, i.e. another account's older gpu-run (version < 4) created it " +
                   "with the default ACL. Let that holder finish, or stop it, so the mutex is destroyed; " +
                   "the next gpu-run (this version) then recreates it open to Authenticated Users.")
        }
        $createdNew = $false
        try {
            if ($core) {
                # .NET (pwsh 7): the security-taking constructor moved to MutexAcl.Create.
                $m = [System.Threading.MutexAcl]::Create($false, $Name, [ref] $createdNew, $security)
            } else {
                # .NET Framework (Windows PowerShell 5.1).
                $m = [System.Threading.Mutex]::new($false, $Name, [ref] $createdNew, $security)
            }
            if ($createdNew) { return $m }
            $m.Dispose()   # someone else created it between our open and create: open theirs
        }
        catch [System.UnauthorizedAccessException] { }   # lost the create race to a mutex with our ACL: open it
    }
    throw "gpu-run: could not open or create '$Name' after 8 attempts"
}
