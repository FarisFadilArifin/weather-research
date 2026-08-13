param(
    [string]$StartDate = "2024-01-01",
    [string]$EndDate = "2026-07-14",
    [int]$ShardMonths = 3,
    [int]$MaxParallel = 4,
    [string[]]$Stations = @("KATL", "KDAL"),
    [string]$LogLabel = "pilot_20240715",
    [string]$NbmSeedRoot = "data\calibration\nbm_rap_features_shards_priority_20260702_full",
    [switch]$KeepRaw
)

$ErrorActionPreference = "Stop"
$Stations = @(
    $Stations |
        ForEach-Object { $_ -split "," } |
        Where-Object { $_ } |
        ForEach-Object { $_.Trim().ToUpperInvariant() }
)
$unsupported = @($Stations | Where-Object { $_ -notin @("KATL", "KDAL") })
if ($unsupported.Count -gt 0) {
    throw "peak_timing_v1 supports KATL and KDAL only; got: $($unsupported -join ',')"
}

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptRoot "..")
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$CacheRoot = Join-Path $ProjectRoot "data\calibration\peak_timing_features_shards_$LogLabel"
$RawRoot = Join-Path $ProjectRoot "data\raw\peak_timing_features_shards_$LogLabel"
$LogRoot = Join-Path $ProjectRoot "logs\peak_timing_features_$LogLabel"
$ManifestPath = Join-Path $LogRoot "manifest.csv"
$StatusPath = Join-Path $LogRoot "status.csv"
$EventsPath = Join-Path $LogRoot "events.log"
$SeedPath = if ([System.IO.Path]::IsPathRooted($NbmSeedRoot)) {
    $NbmSeedRoot
} else {
    Join-Path $ProjectRoot $NbmSeedRoot
}

foreach ($path in @($CacheRoot, $RawRoot, $LogRoot)) {
    New-Item -ItemType Directory -Force -Path $path | Out-Null
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Missing project Python interpreter: $Python"
}
if ($ShardMonths -lt 1 -or $MaxParallel -lt 1) {
    throw "ShardMonths and MaxParallel must be positive."
}

function Write-EventLine {
    param([string]$Message)
    Add-Content -LiteralPath $EventsPath -Value "$((Get-Date).ToString('o')) $Message"
}

function New-ShardPeriods {
    param([datetime]$Start, [datetime]$End, [int]$Months)
    $cursor = $Start
    while ($cursor -le $End) {
        $periodEnd = $cursor.AddMonths($Months).AddDays(-1)
        if ($periodEnd -gt $End) { $periodEnd = $End }
        [pscustomobject]@{
            start_date = $cursor.ToString("yyyy-MM-dd")
            end_date = $periodEnd.ToString("yyyy-MM-dd")
        }
        $cursor = $periodEnd.AddDays(1)
    }
}

function Get-ExpectedRows {
    param([object]$Shard)
    $days = (([datetime]::Parse($Shard.end_date)) - ([datetime]::Parse($Shard.start_date))).Days + 1
    return $days * $Stations.Count
}

function Get-ShardStats {
    param([object]$Shard)
    $cachePath = Join-Path $Shard.cache_dir "peak_timing_features.csv"
    $expected = Get-ExpectedRows -Shard $Shard
    if (-not (Test-Path -LiteralPath $cachePath -PathType Leaf)) {
        return [pscustomobject]@{ rows = 0; ok_rows = 0; expected_rows = $expected; complete = $false }
    }
    try {
        $rows = @(Import-Csv -LiteralPath $cachePath)
    }
    catch {
        return [pscustomobject]@{ rows = 0; ok_rows = 0; expected_rows = $expected; complete = $false }
    }
    $okRows = @($rows | Where-Object {
        $_.schema_version -eq "1" -and
        $_.nbm_core_fetch_status -eq "ok" -and
        $_.hrrr_fetch_status -eq "ok"
    }).Count
    return [pscustomobject]@{
        rows = $rows.Count
        ok_rows = $okRows
        expected_rows = $expected
        complete = ($rows.Count -ge $expected -and $okRows -ge $expected)
    }
}

function Save-Status {
    param([object[]]$Shards)
    $Shards |
        Select-Object shard_id, start_date, end_date, state, pid, exit_code, rows, ok_rows, expected_rows, started_at, finished_at, cache_dir, raw_dir, stdout_log, stderr_log |
        Export-Csv -LiteralPath $StatusPath -NoTypeInformation
}

$start = [datetime]::Parse($StartDate)
$end = [datetime]::Parse($EndDate)
if ($end -lt $start) { throw "EndDate must be on or after StartDate." }

$shards = @()
foreach ($period in @(New-ShardPeriods -Start $start -End $end -Months $ShardMonths)) {
    $id = "peak_{0}_{1}" -f ($period.start_date -replace "-", ""), ($period.end_date -replace "-", "")
    $cacheDir = Join-Path $CacheRoot $id
    $rawDir = Join-Path $RawRoot $id
    $shards += [pscustomobject]@{
        shard_id = $id
        start_date = $period.start_date
        end_date = $period.end_date
        state = "pending"
        pid = ""
        exit_code = ""
        rows = 0
        ok_rows = 0
        expected_rows = 0
        started_at = ""
        finished_at = ""
        cache_dir = $cacheDir
        raw_dir = $rawDir
        stdout_log = Join-Path $LogRoot "$id.out.log"
        stderr_log = Join-Path $LogRoot "$id.err.log"
        process = $null
    }
}
foreach ($shard in $shards) { $shard.expected_rows = Get-ExpectedRows -Shard $shard }

$shards |
    Select-Object shard_id, start_date, end_date, expected_rows, cache_dir, raw_dir, stdout_log, stderr_log |
    Export-Csv -LiteralPath $ManifestPath -NoTypeInformation
Write-EventLine "created manifest shards=$($shards.Count) stations=$($Stations -join ',') max_parallel=$MaxParallel shard_months=$ShardMonths"
Save-Status -Shards $shards

while (@($shards | Where-Object { $_.state -in @("pending", "running") }).Count -gt 0) {
    foreach ($shard in @($shards | Where-Object { $_.state -eq "running" })) {
        $shard.process.Refresh()
        $stats = Get-ShardStats -Shard $shard
        $shard.rows = $stats.rows
        $shard.ok_rows = $stats.ok_rows
        if ($shard.process.HasExited) {
            $shard.process.WaitForExit()
            $shard.exit_code = $shard.process.ExitCode
            $shard.finished_at = (Get-Date).ToString("o")
            if ($stats.complete) {
                $shard.state = "done"
                Write-EventLine "done $($shard.shard_id) rows=$($stats.rows)/$($stats.expected_rows)"
            } else {
                $shard.state = "failed"
                Write-EventLine "failed $($shard.shard_id) exit=$($shard.exit_code) ok=$($stats.ok_rows)/$($stats.expected_rows)"
            }
            $shard.process = $null
        }
    }

    $slots = $MaxParallel - @($shards | Where-Object { $_.state -eq "running" }).Count
    foreach ($shard in @($shards | Where-Object { $_.state -eq "pending" } | Select-Object -First $slots)) {
        foreach ($path in @($shard.cache_dir, $shard.raw_dir)) {
            New-Item -ItemType Directory -Force -Path $path | Out-Null
        }
        $stats = Get-ShardStats -Shard $shard
        $shard.rows = $stats.rows
        $shard.ok_rows = $stats.ok_rows
        if ($stats.complete) {
            $shard.state = "done"
            $shard.exit_code = "cache_complete"
            $shard.finished_at = (Get-Date).ToString("o")
            Write-EventLine "skipped $($shard.shard_id) cache_complete"
            continue
        }

        $arguments = @(
            "-m", "src.backfill_peak_timing_features",
            "--cache-dir", $shard.cache_dir,
            "--raw-dir", $shard.raw_dir,
            "--nbm-seed-root", $SeedPath,
            "--stations"
        )
        $arguments += $Stations
        $arguments += @(
            "--start-date", $shard.start_date,
            "--end-date", $shard.end_date,
            "--log-level", "INFO"
        )
        if (-not $KeepRaw) { $arguments += "--discard-raw" }
        $process = Start-Process -FilePath $Python `
            -ArgumentList $arguments `
            -WorkingDirectory $ProjectRoot `
            -RedirectStandardOutput $shard.stdout_log `
            -RedirectStandardError $shard.stderr_log `
            -WindowStyle Hidden `
            -PassThru
        $shard.process = $process
        $shard.pid = $process.Id
        $shard.started_at = (Get-Date).ToString("o")
        $shard.state = "running"
        Write-EventLine "started $($shard.shard_id) pid=$($process.Id)"
    }
    Save-Status -Shards $shards
    if (@($shards | Where-Object { $_.state -eq "running" }).Count -gt 0) {
        Start-Sleep -Seconds 10
    }
}

Save-Status -Shards $shards
$failed = @($shards | Where-Object { $_.state -eq "failed" })
if ($failed.Count -gt 0) {
    Write-EventLine "finished failures=$($failed.Count)"
    exit 1
}
Write-EventLine "finished successfully"
