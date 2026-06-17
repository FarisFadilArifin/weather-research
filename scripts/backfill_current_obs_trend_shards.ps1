param(
    [string]$StartDate = "2021-01-01",
    [string]$EndDate = "2026-06-10",
    [int]$ShardMonths = 12,
    [int]$MaxParallel = 18,
    [int]$ChunkDays = 365,
    [string]$Source = "iem",
    [string[]]$Stations = @("KATL", "KAUS", "KORD", "KDAL", "KHOU", "KLAX", "KMIA", "KLGA", "KSEA")
)

$ErrorActionPreference = "Stop"

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptRoot "..")
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$LogRoot = Join-Path $ProjectRoot "logs\current_obs_trend_shards"
$DataRoot = Join-Path $ProjectRoot "data\calibration"
$ManifestPath = Join-Path $LogRoot "manifest.csv"
$StatusPath = Join-Path $LogRoot "status.csv"
$EventsPath = Join-Path $LogRoot "events.log"

New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

if (-not (Test-Path $Python)) {
    throw "Missing project Python interpreter: $Python"
}

function Write-EventLine {
    param([string]$Message)
    $timestamp = (Get-Date).ToString("o")
    Add-Content -Path $EventsPath -Value "$timestamp $Message"
}

function New-ShardPeriods {
    param(
        [datetime]$Start,
        [datetime]$End,
        [int]$Months
    )
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
        Select-Object shard_id, station_id, start_date, end_date, state, pid, exit_code, started_at, finished_at, cache_dir, stdout_log, stderr_log |
        Export-Csv -Path $StatusPath -NoTypeInformation
}

function Get-ShardExpectedRows {
    param([object]$Shard)
    $days = (([datetime]::Parse($Shard.end_date)) - ([datetime]::Parse($Shard.start_date))).Days + 1
    return $days
}

function Get-ShardCacheStats {
    param([object]$Shard)
    $cachePath = Join-Path $Shard.cache_dir "sdk_current_observations_11am.csv"
    $expectedRows = Get-ShardExpectedRows -Shard $Shard
    if (-not (Test-Path $cachePath)) {
        return [pscustomobject]@{ rows = 0; ok_rows = 0; non_ok_rows = 0; trend_rows = 0; expected_rows = $expectedRows; complete = $false }
    }
    try {
        $rows = @(Import-Csv -Path $cachePath)
    }
    catch {
        return [pscustomobject]@{ rows = 0; ok_rows = 0; non_ok_rows = 0; trend_rows = 0; expected_rows = $expectedRows; complete = $false }
    }
    $okRows = @($rows | Where-Object { $_.observed_fetch_status -eq "ok" }).Count
    $trendRows = @(
        $rows | Where-Object {
            $_.observed_temp_change_last_1h_f -ne "" -and
            $_.observed_temp_change_last_1h_f -ne $null -and
            $_.observed_morning_warmup_rate_f_per_hour -ne "" -and
            $_.observed_morning_warmup_rate_f_per_hour -ne $null
        }
    ).Count
    return [pscustomobject]@{
        rows = $rows.Count
        ok_rows = $okRows
        non_ok_rows = $rows.Count - $okRows
        trend_rows = $trendRows
        expected_rows = $expectedRows
        complete = ($rows.Count -ge $expectedRows)
    }
}

$start = [datetime]::Parse($StartDate)
$end = [datetime]::Parse($EndDate)
if ($end -lt $start) {
    throw "EndDate must be >= StartDate"
}
if ($ShardMonths -lt 1) {
    throw "ShardMonths must be >= 1"
}
if ($MaxParallel -lt 1) {
    throw "MaxParallel must be >= 1"
}
if ($ChunkDays -lt 1) {
    throw "ChunkDays must be >= 1"
}

$periods = @(New-ShardPeriods -Start $start -End $end -Months $ShardMonths)
$shards = @()
foreach ($station in $Stations) {
    foreach ($period in $periods) {
        $stationKey = $station.ToUpper()
        $startKey = $period.start_date -replace "-", ""
        $endKey = $period.end_date -replace "-", ""
        $shardId = "obs_trends_{0}_{1}_{2}" -f $stationKey, $startKey, $endKey
        $cacheDir = Join-Path $DataRoot "sdk_current_obs_trends_${stationKey}_${startKey}_${endKey}"
        $stdoutLog = Join-Path $LogRoot "$shardId.out.log"
        $stderrLog = Join-Path $LogRoot "$shardId.err.log"
        $shards += [pscustomobject]@{
            shard_id = $shardId
            station_id = $station.ToUpper()
            start_date = $period.start_date
            end_date = $period.end_date
            state = "pending"
            pid = ""
            exit_code = ""
            started_at = ""
            finished_at = ""
            cache_dir = $cacheDir
            stdout_log = $stdoutLog
            stderr_log = $stderrLog
            process = $null
        }
    }
}

$shards |
    Select-Object shard_id, station_id, start_date, end_date, cache_dir, stdout_log, stderr_log |
    Export-Csv -Path $ManifestPath -NoTypeInformation

Write-EventLine "created manifest with $($shards.Count) shards; max_parallel=$MaxParallel shard_months=$ShardMonths chunk_days=$ChunkDays source=$Source"
Save-Status -Shards $shards

while (@($shards | Where-Object { $_.state -in @("pending", "running") }).Count -gt 0) {
    foreach ($shard in @($shards | Where-Object { $_.state -eq "running" })) {
        $proc = $shard.process
        if ($null -eq $proc) {
            continue
        }
        $proc.Refresh()
        if ($proc.HasExited) {
            try {
                $proc.WaitForExit()
                $exitCode = $proc.ExitCode
            }
            catch {
                $exitCode = $null
            }
            $stats = Get-ShardCacheStats -Shard $shard
            $shard.exit_code = if ($null -eq $exitCode) { "" } else { $exitCode }
            $shard.finished_at = (Get-Date).ToString("o")
            if ($stats.complete) {
                $shard.state = "done"
                if ($exitCode -ne 0) {
                    $shard.exit_code = "cache_complete"
                }
                Write-EventLine "done $($shard.shard_id) rows=$($stats.rows) ok=$($stats.ok_rows) non_ok=$($stats.non_ok_rows) trend=$($stats.trend_rows)"
            }
            else {
                $shard.state = "failed"
                Write-EventLine "failed $($shard.shard_id) exit=$exitCode rows=$($stats.rows) ok=$($stats.ok_rows) non_ok=$($stats.non_ok_rows) trend=$($stats.trend_rows)"
            }
            $shard.process = $null
        }
    }

    $runningCount = @($shards | Where-Object { $_.state -eq "running" }).Count
    $slots = $MaxParallel - $runningCount
    if ($slots -gt 0) {
        foreach ($shard in @($shards | Where-Object { $_.state -eq "pending" } | Select-Object -First $slots)) {
            New-Item -ItemType Directory -Force -Path $shard.cache_dir | Out-Null
            $stats = Get-ShardCacheStats -Shard $shard
            if ($stats.complete) {
                $shard.state = "done"
                $shard.exit_code = "cache_complete"
                $shard.finished_at = (Get-Date).ToString("o")
                Write-EventLine "skipped $($shard.shard_id) cache_complete rows=$($stats.rows) ok=$($stats.ok_rows) non_ok=$($stats.non_ok_rows) trend=$($stats.trend_rows)"
                continue
            }
            $args = @(
                "-m", "src.backfill_mostlyright_current_observations",
                "--sdk-cache-dir", $shard.cache_dir,
                "--stations", $shard.station_id,
                "--start-date", $shard.start_date,
                "--end-date", $shard.end_date,
                "--chunk-days", [string]$ChunkDays,
                "--source", $Source,
                "--request-retries", "1",
                "--retry-sleep-seconds", "5",
                "--log-level", "INFO"
            )
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
    Start-Sleep -Seconds 15
}

Save-Status -Shards $shards
$failed = @($shards | Where-Object { $_.state -eq "failed" })
if ($failed.Count -gt 0) {
    Write-EventLine "finished with failures count=$($failed.Count)"
    exit 1
}

Write-EventLine "finished all shards successfully"
exit 0
