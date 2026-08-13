param(
    [int]$MaxConcurrent = 3,
    [int]$FxxWorkers = 1,
    [string]$Python = ".venv\Scripts\python.exe",
    [string]$CanonicalCacheDir = "data/calibration/sdk_9am_live_safe_2021_latest",
    [string]$ShardsRoot = "data/calibration/sdk_9am_live_safe_shards_safe",
    [string]$LogRoot = "logs/9am_sdk_pull/shards_safe"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $Python)) {
    $Python = "python"
} else {
    $Python = (Resolve-Path $Python).Path
}

New-Item -ItemType Directory -Force -Path $ShardsRoot | Out-Null
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

$tasks = @(
    @{Name = "gfs_20210722_20221231"; Model = "gfs"; Start = "2021-07-22"; End = "2022-12-31"},
    @{Name = "gfs_20230101_20241231"; Model = "gfs"; Start = "2023-01-01"; End = "2024-12-31"},
    @{Name = "gfs_20250101_20260630"; Model = "gfs"; Start = "2025-01-01"; End = "2026-06-30"},
    @{Name = "hrrr_20210101_20221231"; Model = "hrrr"; Start = "2021-01-01"; End = "2022-12-31"},
    @{Name = "hrrr_20230101_20241231"; Model = "hrrr"; Start = "2023-01-01"; End = "2024-12-31"},
    @{Name = "hrrr_20250101_20260630"; Model = "hrrr"; Start = "2025-01-01"; End = "2026-06-30"},
    @{Name = "nbm_20210101_20221231"; Model = "nbm"; Start = "2021-01-01"; End = "2022-12-31"},
    @{Name = "nbm_20230101_20241231"; Model = "nbm"; Start = "2023-01-01"; End = "2024-12-31"},
    @{Name = "nbm_20250101_20260630"; Model = "nbm"; Start = "2025-01-01"; End = "2026-06-30"}
)

$pending = [System.Collections.Queue]::new()
foreach ($task in $tasks) {
    $pending.Enqueue($task)
}

$running = @{}

function Write-SchedulerStatus {
    param([string]$Message)
    $stamp = Get-Date -Format o
    "$stamp $Message" | Tee-Object -FilePath (Join-Path $LogRoot "scheduler.log") -Append
}

function Start-Shard {
    param($Task)

    $name = $Task.Name
    $cacheDir = Join-Path $ShardsRoot $name
    $outLog = Join-Path $LogRoot "$name.out.log"
    $errLog = Join-Path $LogRoot "$name.err.log"
    $pidFile = Join-Path $LogRoot "$name.pid"
    $exitFile = Join-Path $LogRoot "$name.exitcode"
    if (Test-Path $exitFile) {
        Remove-Item -LiteralPath $exitFile -Force
    }
    New-Item -ItemType Directory -Force -Path $cacheDir | Out-Null

    $command = "& '$Python' -m src.backfill_mostlyright_sdk_nwp --sdk-cache-dir '$cacheDir' --stations KATL KDAL KMIA KSEA --models '$($Task.Model)' --timing-mode same_day_9am_live_safe --start-date '$($Task.Start)' --end-date '$($Task.End)' --temperature-only --fxx-workers $FxxWorkers --log-level INFO; `$LASTEXITCODE | Out-File -FilePath '$exitFile' -Encoding ascii"
    $process = Start-Process -FilePath "powershell.exe" `
        -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $command) `
        -WorkingDirectory (Get-Location) `
        -RedirectStandardOutput $outLog `
        -RedirectStandardError $errLog `
        -PassThru `
        -WindowStyle Hidden
    $process.Id | Out-File -FilePath $pidFile -Encoding ascii
    $running[$name] = $process
    Write-SchedulerStatus "started name=$name pid=$($process.Id) model=$($Task.Model) start=$($Task.Start) end=$($Task.End)"
}

Write-SchedulerStatus "scheduler_start max_concurrent=$MaxConcurrent fxx_workers=$FxxWorkers"

while ($pending.Count -gt 0 -or $running.Count -gt 0) {
    while ($pending.Count -gt 0 -and $running.Count -lt $MaxConcurrent) {
        Start-Shard -Task $pending.Dequeue()
    }

    Start-Sleep -Seconds 30

    foreach ($name in @($running.Keys)) {
        $process = $running[$name]
        $live = Get-Process -Id $process.Id -ErrorAction SilentlyContinue
        if (-not $live) {
            $exitPath = Join-Path $LogRoot "$name.exitcode"
            $exitCode = if (Test-Path $exitPath) { (Get-Content $exitPath | Select-Object -First 1) } else { "missing" }
            Write-SchedulerStatus "finished name=$name pid=$($process.Id) exit=$exitCode"
            $running.Remove($name)
        }
    }
}

Write-SchedulerStatus "all_shards_finished; merging"
& $Python scripts/merge_9am_sdk_nwp_shards.py --canonical-cache-dir $CanonicalCacheDir --shards-root $ShardsRoot `
    > (Join-Path $LogRoot "merge.out.log") `
    2> (Join-Path $LogRoot "merge.err.log")
$mergeExit = $LASTEXITCODE
Write-SchedulerStatus "merge_exit=$mergeExit"

& $Python -m src.verify_sdk_coverage `
    --sdk-cache-dir $CanonicalCacheDir `
    --stations KATL KDAL KMIA KSEA `
    --models hrrr gfs nbm `
    --timing-mode same_day_9am_live_safe `
    --start-date 2021-01-01 `
    --end-date latest-complete `
    --log-level INFO `
    > (Join-Path $LogRoot "coverage.out.log") `
    2> (Join-Path $LogRoot "coverage.err.log")
$coverageExit = $LASTEXITCODE
Write-SchedulerStatus "coverage_exit=$coverageExit"

"merge_exit=$mergeExit coverage_exit=$coverageExit completed_at=$(Get-Date -Format o)" |
    Out-File -FilePath (Join-Path $LogRoot "scheduler.done") -Encoding ascii
