param(
    [string]$StartDate = "2021-01-01",
    [string]$EndDate = "latest-complete",
    [int]$MaxParallel = 12,
    [int]$WeatherMaxParallel = 12,
    [int]$WeatherFxxWorkers = 3,
    [int]$FxxWorkers = 1,
    [string]$Python = ".venv\Scripts\python.exe",
    [string]$CanonicalCacheDir = "data/calibration/sdk_1pm_live_safe_2021_latest",
    [string]$ObservationCacheDir = "data/calibration/sdk_current_obs_1pm_live_safe_2021_latest",
    [string]$ShardsRoot = "data/calibration/sdk_1pm_live_safe_shards",
    [string]$LogRoot = "logs/kdal_1pm_pull",
    [switch]$SkipWeatherEnrichment,
    [switch]$SkipObservations,
    [switch]$SkipAudit
)

$ErrorActionPreference = "Stop"
$TimingMode = "same_day_1pm_live_safe"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not [IO.Path]::IsPathRooted($Python)) { $Python = Join-Path $ProjectRoot $Python }
if (-not (Test-Path -LiteralPath $Python)) { throw "Missing Python interpreter: $Python" }
foreach ($name in @("CanonicalCacheDir", "ObservationCacheDir", "ShardsRoot", "LogRoot")) {
    $value = Get-Variable -Name $name -ValueOnly
    if (-not [IO.Path]::IsPathRooted($value)) { Set-Variable -Name $name -Value (Join-Path $ProjectRoot $value) }
}
if ($EndDate -eq "latest-complete") { $EndDate = [datetime]::UtcNow.Date.AddDays(-1).ToString("yyyy-MM-dd") }
if ($MaxParallel -lt 1) { throw "MaxParallel must be >= 1" }
if ($WeatherMaxParallel -lt 1 -or $WeatherMaxParallel -gt $MaxParallel) { throw "WeatherMaxParallel must be between 1 and MaxParallel" }
if ($WeatherFxxWorkers -lt 1) { throw "WeatherFxxWorkers must be >= 1" }
if ($FxxWorkers -ne 1) { throw "The 12-worker safety contract requires FxxWorkers=1" }
New-Item -ItemType Directory -Force -Path $ShardsRoot, $LogRoot, $CanonicalCacheDir, $ObservationCacheDir | Out-Null

$ManifestPath = Join-Path $LogRoot "manifest.csv"
$StatusPath = Join-Path $LogRoot "status.csv"
$EventsPath = Join-Path $LogRoot "events.log"
$RunStarted = Get-Date

function Write-Event([string]$Message) {
    "$(Get-Date -Format o) $Message" | Tee-Object -FilePath $EventsPath -Append
}

function New-QuarterPeriods([datetime]$Start, [datetime]$End) {
    $cursor = $Start
    while ($cursor -le $End) {
        $quarter = [math]::Floor(($cursor.Month - 1) / 3)
        $quarterEndMonth = ([int]$quarter + 1) * 3
        $periodEnd = (Get-Date -Year $cursor.Year -Month $quarterEndMonth -Day 1).AddMonths(1).AddDays(-1)
        if ($periodEnd -gt $End) { $periodEnd = $End }
        [pscustomobject]@{ Start = $cursor.ToString("yyyy-MM-dd"); End = $periodEnd.ToString("yyyy-MM-dd") }
        $cursor = $periodEnd.AddDays(1)
    }
}

function New-Tasks([string]$Phase) {
    $models = if ($Phase -eq "core") { @("gfs", "hrrr", "nbm") } else { @("gfs", "hrrr") }
    $tasks = @()
    foreach ($period in @(New-QuarterPeriods ([datetime]$StartDate) ([datetime]$EndDate))) {
        foreach ($model in $models) {
            $id = "${Phase}_${model}_$($period.Start.Replace('-', ''))_$($period.End.Replace('-', ''))"
            $cacheDir = Join-Path $ShardsRoot "$($model)_$($period.Start.Replace('-', ''))_$($period.End.Replace('-', ''))"
            $days = (([datetime]$period.End) - ([datetime]$period.Start)).Days + 1
            $tasks += [pscustomobject]@{
                phase=$Phase; shard_id=$id; model=$model; start_date=$period.Start; end_date=$period.End
                expected_rows=$days; state="pending"; attempt=0; pid=""; exit_code=""; rows=0; ok_rows=0
                started_at=""; finished_at=""; eta=""; cache_dir=$cacheDir
                stdout_log=(Join-Path $LogRoot "$id.out.log"); stderr_log=(Join-Path $LogRoot "$id.err.log"); process=$null
            }
        }
    }
    return $tasks
}

function Get-TaskStats($Task) {
    $file = if ($Task.model -eq "nbm") { "direct_nbm_0h_cache.csv" } else { "sdk_nwp_0h_cache.csv" }
    $path = Join-Path $Task.cache_dir $file
    if (-not (Test-Path -LiteralPath $path)) { return @{ rows=0; ok=0; complete=$false } }
    try { $rows = @(Import-Csv -LiteralPath $path) } catch { return @{ rows=0; ok=0; complete=$false } }
    $matching = @($rows | Where-Object {
        $_.station_id -eq "KDAL" -and $_.timing_mode -eq $TimingMode -and $_.provider -eq $Task.model -and
        $_.contract_date -ge $Task.start_date -and $_.contract_date -le $Task.end_date
    })
    $ok = @($matching | Where-Object { $_.fetch_status -eq "ok" }).Count
    $weatherComplete = $true
    if ($Task.phase -eq "weather") {
        $weatherComplete = @($matching | Where-Object { $_.fetch_status -eq "ok" -and $_.weather_features_included -ne "True" }).Count -eq 0
    }
    return @{ rows=$matching.Count; ok=$ok; complete=($matching.Count -ge $Task.expected_rows -and $weatherComplete) }
}

function Save-Status($Tasks) {
    $Tasks | Select-Object phase,shard_id,model,start_date,end_date,expected_rows,state,attempt,pid,exit_code,rows,ok_rows,started_at,finished_at,eta,cache_dir,stdout_log,stderr_log |
        Export-Csv -Path $StatusPath -NoTypeInformation
}

function Update-Eta($Tasks) {
    $done = @($Tasks | Where-Object { $_.state -eq "done" })
    if ($done.Count -lt 3) { return "pending_first_3_shards" }
    $completedRows = ($done | Measure-Object expected_rows -Sum).Sum
    $elapsedMinutes = [math]::Max(((Get-Date) - $RunStarted).TotalMinutes, 0.1)
    $rowsPerMinute = $completedRows / $elapsedMinutes
    $remainingRows = (($Tasks | Where-Object { $_.state -in @("pending", "running") }) | Measure-Object expected_rows -Sum).Sum
    if ($rowsPerMinute -le 0) { return "unknown" }
    $minutes = [math]::Ceiling($remainingRows / $rowsPerMinute)
    return "$((Get-Date).AddMinutes($minutes).ToString('o')) rows_per_minute=$([math]::Round($rowsPerMinute,2))"
}

function Start-Task($Task) {
    New-Item -ItemType Directory -Force -Path $Task.cache_dir | Out-Null
    $Task.attempt = [int]$Task.attempt + 1
    $args = if ($Task.model -eq "nbm") {
        @("-m","src.backfill_direct_nbm","--sdk-cache-dir",$Task.cache_dir,"--stations","KDAL","--timing-mode",$TimingMode,"--start-date",$Task.start_date,"--end-date",$Task.end_date,"--log-level","INFO")
    } else {
        $taskFxxWorkers = if ($Task.phase -eq "weather") { $WeatherFxxWorkers } else { $FxxWorkers }
        @("-m","src.backfill_mostlyright_sdk_nwp","--sdk-cache-dir",$Task.cache_dir,"--stations","KDAL","--models",$Task.model,"--timing-mode",$TimingMode,"--start-date",$Task.start_date,"--end-date",$Task.end_date,"--fxx-workers",$taskFxxWorkers,"--log-level","INFO")
    }
    if ($Task.phase -eq "core" -and $Task.model -ne "nbm") { $args += "--temperature-only" }
    if ($Task.phase -eq "weather") { $args += "--include-weather-features" }
    $proc = Start-Process -FilePath $Python -ArgumentList $args -WorkingDirectory $ProjectRoot -RedirectStandardOutput $Task.stdout_log -RedirectStandardError $Task.stderr_log -WindowStyle Hidden -PassThru
    $Task.process=$proc; $Task.pid=$proc.Id; $Task.started_at=(Get-Date).ToString("o"); $Task.state="running"
    Write-Event "started phase=$($Task.phase) shard=$($Task.shard_id) model=$($Task.model) pid=$($proc.Id) attempt=$($Task.attempt)"
}

function Invoke-Phase([string]$Phase) {
    $tasks = @(New-Tasks $Phase)
    $phaseMaxParallel = if ($Phase -eq "weather") { $WeatherMaxParallel } else { $MaxParallel }
    $tasks | Select-Object phase,shard_id,model,start_date,end_date,expected_rows,cache_dir,stdout_log,stderr_log | Export-Csv -Path $ManifestPath -NoTypeInformation -Append
    foreach ($task in $tasks) {
        $stats = Get-TaskStats $task
        if ($stats.complete) { $task.state="done"; $task.exit_code="cache_complete"; $task.rows=$stats.rows; $task.ok_rows=$stats.ok }
    }
    for ($round=1; $round -le 2; $round++) {
        while (@($tasks | Where-Object { $_.state -in @("pending","running") }).Count -gt 0) {
            foreach ($task in @($tasks | Where-Object { $_.state -eq "running" })) {
                $task.process.Refresh()
                if ($task.process.HasExited) {
                    $task.process.WaitForExit(); $task.exit_code=$task.process.ExitCode; $task.finished_at=(Get-Date).ToString("o")
                    $stats=Get-TaskStats $task; $task.rows=$stats.rows; $task.ok_rows=$stats.ok
                    $task.state = if ($stats.complete) { "done" } else { "failed" }
                    Write-Event "finished phase=$Phase shard=$($task.shard_id) state=$($task.state) exit=$($task.exit_code) rows=$($task.rows) ok=$($task.ok_rows)"
                    $task.process=$null
                }
            }
            $slots=$phaseMaxParallel-@($tasks | Where-Object {$_.state -eq "running"}).Count
            foreach ($task in @($tasks | Where-Object {$_.state -eq "pending"} | Select-Object -First $slots)) { Start-Task $task }
            $eta=Update-Eta $tasks
            foreach ($task in $tasks) { $task.eta=$eta }
            Save-Status $tasks
            Start-Sleep -Seconds 15
        }
        $failed=@($tasks | Where-Object {$_.state -eq "failed"})
        if ($failed.Count -eq 0) { break }
        if ($round -eq 1) {
            Write-Event "retrying phase=$Phase failed_shards=$($failed.Count)"
            foreach ($task in $failed) { $task.state="pending" }
        }
    }
    Save-Status $tasks
    $failed=@($tasks | Where-Object {$_.state -eq "failed"})
    if ($failed.Count -gt 0) { throw "$Phase failed after retry: $($failed.shard_id -join ', ')" }
    Write-Event "phase_complete phase=$Phase eta=$(Update-Eta $tasks)"
    return $tasks
}

Set-Content -Path $ManifestPath -Value 'phase,shard_id,model,start_date,end_date,expected_rows,cache_dir,stdout_log,stderr_log'
Write-Event "run_start start=$StartDate end=$EndDate max_parallel=$MaxParallel weather_max_parallel=$WeatherMaxParallel fxx_workers=$FxxWorkers weather_fxx_workers=$WeatherFxxWorkers"
$core = @(Invoke-Phase "core")
if (-not $SkipWeatherEnrichment) { $weather = @(Invoke-Phase "weather") }

$sdkDirs = @($core | Where-Object {$_.model -in @("gfs","hrrr")} | Select-Object -ExpandProperty cache_dir -Unique)
$nbmDirs = @($core | Where-Object {$_.model -eq "nbm"} | Select-Object -ExpandProperty cache_dir -Unique)
& $Python -m src.merge_sdk_nwp_shards --output-sdk-cache-dir $CanonicalCacheDir --shard-dirs $sdkDirs --log-level INFO
if ($LASTEXITCODE -ne 0) { throw "SDK shard merge failed" }
& $Python -m src.merge_direct_nbm_shards --output-cache-dir $CanonicalCacheDir --shard-dirs $nbmDirs --log-level INFO
if ($LASTEXITCODE -ne 0) { throw "Direct NBM shard merge failed" }

if (-not $SkipObservations) {
    & $Python -m src.backfill_mostlyright_current_observations --sdk-cache-dir $ObservationCacheDir --stations KDAL --timing-mode $TimingMode --as-of-hour-local 13 --start-date $StartDate --end-date $EndDate --chunk-days 31 --source iem --retry-unavailable --request-retries 3 --retry-sleep-seconds 20 --log-level INFO
    if ($LASTEXITCODE -ne 0) { throw "Observation pull failed" }
}
if (-not $SkipAudit) {
    & $Python scripts/audit_kdal_1pm_pull.py --project-root $ProjectRoot --forecast-cache-dir $CanonicalCacheDir --observation-cache-dir $ObservationCacheDir --start-date $StartDate --end-date $EndDate --materialize-features
    if ($LASTEXITCODE -ne 0) { throw "KDAL 1 PM audit failed; inspect unresolved_rows.csv" }
}
Write-Event "run_complete elapsed=$((Get-Date)-$RunStarted)"
