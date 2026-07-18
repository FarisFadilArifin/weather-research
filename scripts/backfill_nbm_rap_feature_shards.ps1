param(
    [string]$StartDate = "2021-01-01",
    [string]$EndDate = "2026-07-01",
    [int]$ShardMonths = 6,
    [int]$MaxParallel = 3,
    [string[]]$Stations = @("KATL", "KDAL", "KMIA"),
    [string[]]$Blocks = @("nbm", "rap"),
    [ValidateSet("hrrr", "rap")]
    [string]$PhysicsModel = "hrrr",
    [switch]$KeepRaw,
    [string]$LogLabel = "priority"
)

$ErrorActionPreference = "Stop"

$Stations = @($Stations | ForEach-Object { $_ -split "," } | Where-Object { $_ } | ForEach-Object { $_.Trim().ToUpperInvariant() })
$Blocks = @($Blocks | ForEach-Object { $_ -split "," } | Where-Object { $_ } | ForEach-Object { $_.Trim().ToLowerInvariant() })

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptRoot "..")
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$LogRoot = Join-Path $ProjectRoot "logs\nbm_rap_features_$LogLabel"
$CacheRoot = Join-Path $ProjectRoot "data\calibration\nbm_rap_features_shards_$LogLabel"
$RawRoot = Join-Path $ProjectRoot "data\raw\nbm_rap_features_shards_$LogLabel"
$ManifestPath = Join-Path $LogRoot "manifest.csv"
$StatusPath = Join-Path $LogRoot "status.csv"
$EventsPath = Join-Path $LogRoot "events.log"

New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
New-Item -ItemType Directory -Force -Path $CacheRoot | Out-Null
New-Item -ItemType Directory -Force -Path $RawRoot | Out-Null

if (-not (Test-Path $Python)) {
    throw "Missing project Python interpreter: $Python"
}

function Write-EventLine {
    param([string]$Message)
    $timestamp = (Get-Date).ToString("o")
    Add-Content -Path $EventsPath -Value "$timestamp $Message"
}

function New-ShardPeriods {
    param([datetime]$Start, [datetime]$End, [int]$Months)
    $cursor = $Start
    while ($cursor -le $End) {
        $periodEnd = $cursor.AddMonths($Months).AddDays(-1)
        if ($periodEnd -gt $End) {
            $periodEnd = $End
        }
        [pscustomobject]@{
            start_date = $cursor.ToString("yyyy-MM-dd")
            end_date = $periodEnd.ToString("yyyy-MM-dd")
        }
        $cursor = $periodEnd.AddDays(1)
    }
}

function Save-Status {
    param([object[]]$Shards)
    $Shards |
        Select-Object shard_id, start_date, end_date, state, pid, exit_code, rows, ok_rows, expected_rows, started_at, finished_at, cache_dir, raw_dir, stdout_log, stderr_log |
        Export-Csv -Path $StatusPath -NoTypeInformation
}

function Get-ShardExpectedRows {
    param([object]$Shard)
    $days = (([datetime]::Parse($Shard.end_date)) - ([datetime]::Parse($Shard.start_date))).Days + 1
    return $days * $Stations.Count
}

function Get-ShardCacheStats {
    param([object]$Shard)
    $cachePath = Join-Path $Shard.cache_dir "nbm_rap_features.csv"
    $expectedRows = Get-ShardExpectedRows -Shard $Shard
    if (-not (Test-Path $cachePath)) {
        return [pscustomobject]@{ rows = 0; ok_rows = 0; expected_rows = $expectedRows; complete = $false }
    }
    try {
        $rows = @(Import-Csv -Path $cachePath)
    }
    catch {
        return [pscustomobject]@{ rows = 0; ok_rows = 0; expected_rows = $expectedRows; complete = $false }
    }
    $okRows = @($rows | Where-Object {
        ((-not ($Blocks -contains "nbm")) -or $_.nbm_core_fetch_status -eq "ok") -and
        ((-not ($Blocks -contains "rap")) -or $_.rap_fetch_status -eq "ok")
    }).Count
    return [pscustomobject]@{
        rows = $rows.Count
        ok_rows = $okRows
        expected_rows = $expectedRows
        complete = ($okRows -ge $expectedRows)
    }
}

$start = [datetime]::Parse($StartDate)
$end = [datetime]::Parse($EndDate)
if ($end -lt $start) { throw "EndDate must be >= StartDate" }
if ($ShardMonths -lt 1) { throw "ShardMonths must be >= 1" }
if ($MaxParallel -lt 1) { throw "MaxParallel must be >= 1" }

$periods = @(New-ShardPeriods -Start $start -End $end -Months $ShardMonths)
$shards = @()
foreach ($period in $periods) {
    $shardId = "nbm_{0}_{1}" -f ($period.start_date -replace "-", ""), ($period.end_date -replace "-", "")
    $cacheDir = Join-Path $CacheRoot $shardId
    $rawDir = Join-Path $RawRoot $shardId
    $stdoutLog = Join-Path $LogRoot "$shardId.out.log"
    $stderrLog = Join-Path $LogRoot "$shardId.err.log"
    $expectedRows = (([datetime]::Parse($period.end_date)) - ([datetime]::Parse($period.start_date))).Days + 1
    $expectedRows *= $Stations.Count
    $shards += [pscustomobject]@{
        shard_id = $shardId
        start_date = $period.start_date
        end_date = $period.end_date
        state = "pending"
        pid = ""
        exit_code = ""
        rows = 0
        ok_rows = 0
        expected_rows = $expectedRows
        started_at = ""
        finished_at = ""
        cache_dir = $cacheDir
        raw_dir = $rawDir
        stdout_log = $stdoutLog
        stderr_log = $stderrLog
        process = $null
    }
}

$shards |
    Select-Object shard_id, start_date, end_date, expected_rows, cache_dir, raw_dir, stdout_log, stderr_log |
    Export-Csv -Path $ManifestPath -NoTypeInformation

Write-EventLine "created manifest with $($shards.Count) shards; max_parallel=$MaxParallel shard_months=$ShardMonths stations=$($Stations -join ',') blocks=$($Blocks -join ',') physics_model=$PhysicsModel keep_raw=$KeepRaw"
Save-Status -Shards $shards

while (@($shards | Where-Object { $_.state -in @("pending", "running") }).Count -gt 0) {
    foreach ($shard in @($shards | Where-Object { $_.state -eq "running" })) {
        $proc = $shard.process
        if ($null -eq $proc) {
            continue
        }
        $proc.Refresh()
        $stats = Get-ShardCacheStats -Shard $shard
        $shard.rows = $stats.rows
        $shard.ok_rows = $stats.ok_rows
        if ($proc.HasExited) {
            try {
                $proc.WaitForExit()
                $exitCode = $proc.ExitCode
            }
            catch {
                $exitCode = $null
            }
            $shard.exit_code = if ($null -eq $exitCode) { "" } else { $exitCode }
            $shard.finished_at = (Get-Date).ToString("o")
            if (($exitCode -eq 0) -or $stats.complete) {
                $shard.state = "done"
                if ($stats.complete -and ($exitCode -ne 0)) {
                    $shard.exit_code = "cache_complete"
                }
                Write-EventLine "done $($shard.shard_id) exit=$($shard.exit_code) rows=$($stats.rows)/$($stats.expected_rows) ok=$($stats.ok_rows)"
            }
            else {
                $shard.state = "failed"
                Write-EventLine "failed $($shard.shard_id) exit=$exitCode rows=$($stats.rows)/$($stats.expected_rows) ok=$($stats.ok_rows)"
            }
            $shard.process = $null
        }
    }

    $runningCount = @($shards | Where-Object { $_.state -eq "running" }).Count
    $slots = $MaxParallel - $runningCount
    if ($slots -gt 0) {
        foreach ($shard in @($shards | Where-Object { $_.state -eq "pending" } | Select-Object -First $slots)) {
            New-Item -ItemType Directory -Force -Path $shard.cache_dir | Out-Null
            New-Item -ItemType Directory -Force -Path $shard.raw_dir | Out-Null
            $stats = Get-ShardCacheStats -Shard $shard
            $shard.rows = $stats.rows
            $shard.ok_rows = $stats.ok_rows
            if ($stats.complete) {
                $shard.state = "done"
                $shard.exit_code = "cache_complete"
                $shard.finished_at = (Get-Date).ToString("o")
                Write-EventLine "skipped $($shard.shard_id) cache_complete rows=$($stats.rows)/$($stats.expected_rows) ok=$($stats.ok_rows)"
                continue
            }
            $args = @(
                "-m", "src.backfill_nbm_rap_features",
                "--cache-dir", $shard.cache_dir,
                "--raw-dir", $shard.raw_dir,
                "--stations"
            )
            $args += $Stations
            $args += @(
                "--start-date", $shard.start_date,
                "--end-date", $shard.end_date,
                "--blocks"
            )
            $args += $Blocks
            $args += @(
                "--physics-model", $PhysicsModel,
                "--log-level", "INFO"
            )
            if (-not $KeepRaw) {
                $args += "--discard-raw"
            }
            $proc = Start-Process -FilePath $Python `
                -ArgumentList $args `
                -WorkingDirectory $ProjectRoot `
                -RedirectStandardOutput $shard.stdout_log `
                -RedirectStandardError $shard.stderr_log `
                -WindowStyle Hidden `
                -PassThru
            $shard.process = $proc
            $shard.pid = $proc.Id
            $shard.started_at = (Get-Date).ToString("o")
            $shard.state = "running"
            Write-EventLine "started $($shard.shard_id) pid=$($proc.Id)"
        }
    }

    Save-Status -Shards $shards
    Start-Sleep -Seconds 30
}

Save-Status -Shards $shards
$failed = @($shards | Where-Object { $_.state -eq "failed" })
if ($failed.Count -gt 0) {
    Write-EventLine "finished with failures count=$($failed.Count)"
    exit 1
}

Write-EventLine "finished all shards successfully"
exit 0
