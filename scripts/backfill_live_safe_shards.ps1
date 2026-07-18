param(
    [string]$StartDate = "2021-01-01",
    [string]$EndDate = "2026-06-10",
    [int]$ShardMonths = 6,
    [int]$MaxParallel = 4,
    [int]$FxxWorkers = 1,
    [string[]]$Models = @("hrrr", "gfs"),
    [string[]]$Stations = @("KATL", "KAUS", "KORD", "KDAL", "KHOU", "KLAX", "KMIA", "KLGA", "KSEA"),
    [string]$TimingMode = "same_day_11am_live_safe",
    [switch]$IncludeWeatherFeatures,
    [string]$LogLabel = ""
)

$ErrorActionPreference = "Stop"

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptRoot "..")
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$LogRootName = if ($IncludeWeatherFeatures) { "live_safe_shards_weather" } else { "live_safe_shards" }
if ($LogLabel) {
    $LogRootName = "${LogRootName}_$LogLabel"
}
$LogRoot = Join-Path $ProjectRoot "logs\$LogRootName"
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
        Select-Object shard_id, model, start_date, end_date, state, pid, exit_code, started_at, finished_at, cache_dir, stdout_log, stderr_log |
        Export-Csv -Path $StatusPath -NoTypeInformation
}

function Get-ShardExpectedRows {
    param([object]$Shard)
    $days = (([datetime]::Parse($Shard.end_date)) - ([datetime]::Parse($Shard.start_date))).Days + 1
    return $days * $Stations.Count
}

function Get-ShardCacheStats {
    param([object]$Shard)
    $cachePath = Join-Path $Shard.cache_dir "sdk_nwp_0h_cache.csv"
    $expectedRows = Get-ShardExpectedRows -Shard $Shard
    if (-not (Test-Path $cachePath)) {
        return [pscustomobject]@{
            rows = 0
            ok_rows = 0
            non_ok_rows = 0
            expected_rows = $expectedRows
            complete = $false
        }
    }
    try {
        $rows = @(Import-Csv -Path $cachePath)
    }
    catch {
        return [pscustomobject]@{
            rows = 0
            ok_rows = 0
            non_ok_rows = 0
            expected_rows = $expectedRows
            complete = $false
        }
    }
    $okRows = @($rows | Where-Object { $_.fetch_status -eq "ok" }).Count
    return [pscustomobject]@{
        rows = $rows.Count
        ok_rows = $okRows
        non_ok_rows = $rows.Count - $okRows
        expected_rows = $expectedRows
        weather_ok_rows = @($rows | Where-Object { $_.fetch_status -eq "ok" -and $_.weather_features_included -eq "True" }).Count
        complete = (
            ($rows.Count -ge $expectedRows) -and
            (-not $IncludeWeatherFeatures -or $okRows -eq 0 -or @($rows | Where-Object { $_.fetch_status -eq "ok" -and $_.weather_features_included -ne "True" }).Count -eq 0)
        )
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
if ($FxxWorkers -lt 1) {
    throw "FxxWorkers must be >= 1"
}

$periods = @(New-ShardPeriods -Start $start -End $end -Months $ShardMonths)
$shards = @()
foreach ($period in $periods) {
    foreach ($model in $Models) {
        $shardId = "{0}_{1}_{2}" -f $model, ($period.start_date -replace "-", ""), ($period.end_date -replace "-", "")
        $cacheDir = Join-Path $DataRoot "sdk_11am_live_safe_$shardId"
        $stdoutLog = Join-Path $LogRoot "$shardId.out.log"
        $stderrLog = Join-Path $LogRoot "$shardId.err.log"
        $shards += [pscustomobject]@{
            shard_id = $shardId
            model = $model
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
    Select-Object shard_id, model, start_date, end_date, cache_dir, stdout_log, stderr_log |
    Export-Csv -Path $ManifestPath -NoTypeInformation

Write-EventLine "created manifest with $($shards.Count) shards; max_parallel=$MaxParallel shard_months=$ShardMonths fxx_workers=$FxxWorkers include_weather_features=$IncludeWeatherFeatures"
Save-Status -Shards $shards

while (($shards | Where-Object { $_.state -in @("pending", "running") }).Count -gt 0) {
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
            if (($exitCode -eq 0) -or $stats.complete) {
                $shard.state = "done"
                if ($stats.complete -and ($exitCode -ne 0)) {
                    $shard.exit_code = "cache_complete"
                    Write-EventLine "done $($shard.shard_id) cache_complete rows=$($stats.rows) ok=$($stats.ok_rows) non_ok=$($stats.non_ok_rows)"
                }
                else {
                    Write-EventLine "done $($shard.shard_id) exit=0 rows=$($stats.rows) ok=$($stats.ok_rows) non_ok=$($stats.non_ok_rows)"
                }
            }
            else {
                $shard.state = "failed"
                Write-EventLine "failed $($shard.shard_id) exit=$exitCode rows=$($stats.rows) ok=$($stats.ok_rows) non_ok=$($stats.non_ok_rows)"
            }
            $shard.process = $null
        }
    }

    $runningCount = ($shards | Where-Object { $_.state -eq "running" }).Count
    $slots = $MaxParallel - $runningCount
    if ($slots -gt 0) {
        foreach ($shard in @($shards | Where-Object { $_.state -eq "pending" } | Select-Object -First $slots)) {
            New-Item -ItemType Directory -Force -Path $shard.cache_dir | Out-Null
            $stats = Get-ShardCacheStats -Shard $shard
            if ($stats.complete) {
                $shard.state = "done"
                $shard.exit_code = "cache_complete"
                $shard.finished_at = (Get-Date).ToString("o")
                Write-EventLine "skipped $($shard.shard_id) cache_complete rows=$($stats.rows) ok=$($stats.ok_rows) non_ok=$($stats.non_ok_rows)"
                continue
            }
            $args = @(
                "-m", "src.backfill_mostlyright_sdk_nwp",
                "--sdk-cache-dir", $shard.cache_dir,
                "--stations"
            )
            $args += $Stations
            $args += @(
                "--models", $shard.model,
                "--timing-mode", $TimingMode,
                "--start-date", $shard.start_date,
                "--end-date", $shard.end_date,
                "--fxx-workers", "$FxxWorkers",
                "--log-level", "INFO"
            )
            if ($IncludeWeatherFeatures) {
                $args += "--include-weather-features"
            }
            else {
                $args += "--temperature-only"
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
