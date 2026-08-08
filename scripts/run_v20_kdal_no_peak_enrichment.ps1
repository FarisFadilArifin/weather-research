param(
    [string]$StartDate = "2026-07-21",
    [string]$EndDate = "2026-07-29",
    [int]$ShardDays = 1,
    [int]$MaxParallel = 12,
    [int]$ForecastFxxWorkers = 3,
    [int]$NbmPrefetchWorkers = 3,
    [string]$Python = ".venv\Scripts\python.exe",
    [string]$LogRoot = "logs\v20_kdal_no_peak_enrichment"
)

$ErrorActionPreference = "Stop"
$Station = "KDAL"
$TimingMode = "same_day_11am_live_safe"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not [IO.Path]::IsPathRooted($Python)) { $Python = Join-Path $ProjectRoot $Python }
if (-not [IO.Path]::IsPathRooted($LogRoot)) { $LogRoot = Join-Path $ProjectRoot $LogRoot }
if (-not (Test-Path -LiteralPath $Python)) { throw "Missing Python interpreter: $Python" }
if ($ShardDays -lt 1) { throw "ShardDays must be >= 1" }
if ($MaxParallel -lt 1) { throw "MaxParallel must be >= 1" }
if ($ForecastFxxWorkers -lt 1) { throw "ForecastFxxWorkers must be >= 1" }
if ($NbmPrefetchWorkers -lt 1) { throw "NbmPrefetchWorkers must be >= 1" }
$Start = [datetime]::ParseExact($StartDate, "yyyy-MM-dd", $null)
$End = [datetime]::ParseExact($EndDate, "yyyy-MM-dd", $null)
if ($End -lt $Start) { throw "EndDate must be >= StartDate" }

$DataRoot = Join-Path $ProjectRoot "data\calibration"
$ManifestPath = Join-Path $LogRoot "manifest.csv"
$StatusPath = Join-Path $LogRoot "status.csv"
$EventsPath = Join-Path $LogRoot "events.log"
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
$RunStarted = Get-Date
$env:WEATHER_RESEARCH_NBM_PREFETCH_WORKERS = [string]$NbmPrefetchWorkers
$env:WEATHER_RESEARCH_INCLUDE_DIRECT_NBM = "1"

function Write-Event([string]$Message) {
    "$(Get-Date -Format o) $Message" | Tee-Object -FilePath $EventsPath -Append
}

function New-DayPeriods {
    $cursor = $Start
    while ($cursor -le $End) {
        $periodEnd = $cursor.AddDays($ShardDays - 1)
        if ($periodEnd -gt $End) { $periodEnd = $End }
        [pscustomobject]@{
            start_date = $cursor.ToString("yyyy-MM-dd")
            end_date = $periodEnd.ToString("yyyy-MM-dd")
            expected_rows = ($periodEnd - $cursor).Days + 1
        }
        $cursor = $periodEnd.AddDays(1)
    }
}

function New-Tasks {
    $tasks = @()
    foreach ($period in @(New-DayPeriods)) {
        $key = "$($period.start_date.Replace('-', ''))_$($period.end_date.Replace('-', ''))"
        foreach ($model in @("gfs", "hrrr")) {
            $id = "sdk_${model}_$key"
            $tasks += [pscustomobject]@{
                kind = "sdk"; model = $model; shard_id = $id
                start_date = $period.start_date; end_date = $period.end_date
                expected_rows = $period.expected_rows
                cache_dir = Join-Path $DataRoot "sdk_11am_live_safe_v20_kdal_enrich_${model}_$key"
                stdout_log = Join-Path $LogRoot "$id.out.log"
                stderr_log = Join-Path $LogRoot "$id.err.log"
                state = "pending"; attempt = 0; pid = ""; exit_code = ""; rows = 0; ok_rows = 0
                started_at = ""; finished_at = ""; process = $null
            }
        }
        $nbmId = "nbm_$key"
        $tasks += [pscustomobject]@{
            kind = "nbm"; model = "nbm"; shard_id = $nbmId
            start_date = $period.start_date; end_date = $period.end_date
            expected_rows = $period.expected_rows
            cache_dir = Join-Path $DataRoot "direct_nbm_v20_kdal_enrich_$key"
            stdout_log = Join-Path $LogRoot "$nbmId.out.log"
            stderr_log = Join-Path $LogRoot "$nbmId.err.log"
            state = "pending"; attempt = 0; pid = ""; exit_code = ""; rows = 0; ok_rows = 0
            started_at = ""; finished_at = ""; process = $null
        }
        $obsId = "observations_$key"
        $tasks += [pscustomobject]@{
            kind = "observations"; model = "iem"; shard_id = $obsId
            start_date = $period.start_date; end_date = $period.end_date
            expected_rows = $period.expected_rows
            cache_dir = Join-Path $DataRoot "sdk_current_obs_v20_kdal_enrich_$key"
            stdout_log = Join-Path $LogRoot "$obsId.out.log"
            stderr_log = Join-Path $LogRoot "$obsId.err.log"
            state = "pending"; attempt = 0; pid = ""; exit_code = ""; rows = 0; ok_rows = 0
            started_at = ""; finished_at = ""; process = $null
        }
    }
    return $tasks
}

function Get-TaskStats($Task) {
    $fileName = switch ($Task.kind) {
        "sdk" { "sdk_nwp_0h_cache.csv" }
        "nbm" { "direct_nbm_0h_cache.csv" }
        default { "sdk_current_observations_11am.csv" }
    }
    $path = Join-Path $Task.cache_dir $fileName
    if (-not (Test-Path -LiteralPath $path)) {
        return @{ rows = 0; ok = 0; complete = $false }
    }
    try { $rows = @(Import-Csv -LiteralPath $path) }
    catch { return @{ rows = 0; ok = 0; complete = $false } }
    $matching = @($rows | Where-Object {
        $_.station_id -eq $Station -and
        $_.contract_date -ge $Task.start_date -and
        $_.contract_date -le $Task.end_date
    })
    if ($Task.kind -eq "observations") {
        $okRows = @($matching | Where-Object { $_.observed_fetch_status -eq "ok" }).Count
        $complete = $matching.Count -ge $Task.expected_rows -and $okRows -ge $Task.expected_rows
    }
    else {
        $matching = @($matching | Where-Object {
            $_.provider -eq $Task.model -and $_.timing_mode -eq $TimingMode
        })
        $okRows = @($matching | Where-Object {
            $_.fetch_status -eq "ok" -and $_.weather_features_included -eq "True"
        }).Count
        $complete = $matching.Count -ge $Task.expected_rows -and $okRows -ge $Task.expected_rows
    }
    return @{ rows = $matching.Count; ok = $okRows; complete = $complete }
}

function Save-Status($Tasks) {
    $Tasks |
        Select-Object kind,model,shard_id,start_date,end_date,expected_rows,state,attempt,pid,exit_code,rows,ok_rows,started_at,finished_at,cache_dir,stdout_log,stderr_log |
        Export-Csv -Path $StatusPath -NoTypeInformation
}

function Start-Task($Task) {
    New-Item -ItemType Directory -Force -Path $Task.cache_dir | Out-Null
    $Task.attempt = [int]$Task.attempt + 1
    $args = switch ($Task.kind) {
        "sdk" {
            @(
                "-m", "src.backfill_mostlyright_sdk_nwp",
                "--sdk-cache-dir", $Task.cache_dir,
                "--stations", $Station,
                "--models", $Task.model,
                "--timing-mode", $TimingMode,
                "--start-date", $Task.start_date,
                "--end-date", $Task.end_date,
                "--fxx-workers", [string]$ForecastFxxWorkers,
                "--include-weather-features",
                "--log-level", "INFO"
            )
        }
        "nbm" {
            @(
                "-m", "src.backfill_direct_nbm",
                "--sdk-cache-dir", $Task.cache_dir,
                "--stations", $Station,
                "--timing-mode", $TimingMode,
                "--start-date", $Task.start_date,
                "--end-date", $Task.end_date,
                "--include-weather-features",
                "--log-level", "INFO"
            )
        }
        default {
            @(
                "-m", "src.backfill_mostlyright_current_observations",
                "--sdk-cache-dir", $Task.cache_dir,
                "--stations", $Station,
                "--timing-mode", "same_day_11am",
                "--as-of-hour-local", "11",
                "--start-date", $Task.start_date,
                "--end-date", $Task.end_date,
                "--chunk-days", [string]$ShardDays,
                "--source", "iem",
                "--retry-unavailable",
                "--request-retries", "3",
                "--retry-sleep-seconds", "5",
                "--log-level", "INFO"
            )
        }
    }
    $proc = Start-Process -FilePath $Python -ArgumentList $args -WorkingDirectory $ProjectRoot `
        -RedirectStandardOutput $Task.stdout_log -RedirectStandardError $Task.stderr_log `
        -WindowStyle Hidden -PassThru
    $Task.process = $proc
    $Task.pid = $proc.Id
    $Task.started_at = (Get-Date).ToString("o")
    $Task.state = "running"
    Write-Event "started shard=$($Task.shard_id) kind=$($Task.kind) pid=$($proc.Id) attempt=$($Task.attempt)"
}

function Invoke-Tasks($Tasks) {
    foreach ($task in $Tasks) {
        $stats = Get-TaskStats $task
        if ($stats.complete) {
            $task.state = "done"; $task.exit_code = "cache_complete"
            $task.rows = $stats.rows; $task.ok_rows = $stats.ok
        }
    }
    for ($round = 1; $round -le 2; $round++) {
        while (@($Tasks | Where-Object { $_.state -in @("pending", "running") }).Count -gt 0) {
            foreach ($task in @($Tasks | Where-Object { $_.state -eq "running" })) {
                $task.process.Refresh()
                if ($task.process.HasExited) {
                    $task.process.WaitForExit()
                    $task.exit_code = $task.process.ExitCode
                    $task.finished_at = (Get-Date).ToString("o")
                    $stats = Get-TaskStats $task
                    $task.rows = $stats.rows; $task.ok_rows = $stats.ok
                    $task.state = if ($stats.complete) { "done" } else { "failed" }
                    Write-Event "finished shard=$($task.shard_id) state=$($task.state) exit=$($task.exit_code) rows=$($task.rows) ok=$($task.ok_rows)"
                    $task.process = $null
                }
            }
            $slots = $MaxParallel - @($Tasks | Where-Object { $_.state -eq "running" }).Count
            foreach ($task in @($Tasks | Where-Object { $_.state -eq "pending" } | Select-Object -First $slots)) {
                Start-Task $task
            }
            Save-Status $Tasks
            Start-Sleep -Seconds 2
        }
        $failed = @($Tasks | Where-Object { $_.state -eq "failed" })
        if ($failed.Count -eq 0) { break }
        if ($round -eq 1) {
            Write-Event "retrying failed_shards=$($failed.Count)"
            foreach ($task in $failed) { $task.state = "pending" }
        }
    }
    Save-Status $Tasks
    $failed = @($Tasks | Where-Object { $_.state -eq "failed" })
    if ($failed.Count -gt 0) {
        throw "Enrichment failed after retry: $($failed.shard_id -join ', ')"
    }
}

$tasks = @(New-Tasks)
$tasks |
    Select-Object kind,model,shard_id,start_date,end_date,expected_rows,cache_dir,stdout_log,stderr_log |
    Export-Csv -Path $ManifestPath -NoTypeInformation
Write-Event "run_start start=$StartDate end=$EndDate shards=$($tasks.Count) shard_days=$ShardDays max_parallel=$MaxParallel fxx_workers=$ForecastFxxWorkers nbm_prefetch_workers=$NbmPrefetchWorkers"
Invoke-Tasks $tasks

$settlementOut = Join-Path $LogRoot "settlements.out.log"
$settlementErr = Join-Path $LogRoot "settlements.err.log"
$settlementArgs = @(
    "-m", "src.backfill_settlement_actuals",
    "--wunderground-history",
    "--stations", $Station,
    "--start-date", $StartDate,
    "--end-date", $EndDate,
    "--log-level", "INFO"
)
$settlement = Start-Process -FilePath $Python -ArgumentList $settlementArgs -WorkingDirectory $ProjectRoot `
    -RedirectStandardOutput $settlementOut -RedirectStandardError $settlementErr `
    -WindowStyle Hidden -PassThru -Wait
if ($settlement.ExitCode -ne 0) { throw "Wunderground settlement enrichment failed; inspect $settlementErr" }
Write-Event "settlements_complete"

& $Python scripts\audit_v20_kdal_no_peak_enrichment.py `
    --project-root $ProjectRoot `
    --start-date $StartDate `
    --end-date $EndDate
if ($LASTEXITCODE -ne 0) {
    throw "V20 KDAL no-peak feature audit failed; inspect data\calibration\station_stacking_v20_kdal_no_peak\audit"
}
Write-Event "run_complete elapsed=$((Get-Date) - $RunStarted)"
