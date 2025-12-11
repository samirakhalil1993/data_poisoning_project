# Data Poisoning Experiments - Kör alla experiment systematiskt
# PowerShell script

$PYTHON = "C:\Users\LabPC\AppData\Local\Microsoft\WindowsApps\python3.13.exe"
$WorkDir = "C:\Users\LabPC\data_poisoning_project\str"

Set-Location $WorkDir

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "DATA POISONING EXPERIMENTS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$experiments = @(
    @{Name="Baseline"; Rate=0.0; Script="run_baseline.py"},
    @{Name="Label Flip 1%"; Rate=0.01; Script="run_label_flip.py"},
    @{Name="Label Flip 5%"; Rate=0.05; Script="run_label_flip.py"},
    @{Name="Label Flip 10%"; Rate=0.10; Script="run_label_flip.py"},
    @{Name="Label Flip 30%"; Rate=0.30; Script="run_label_flip.py"},
    @{Name="Label Flip 50%"; Rate=0.50; Script="run_label_flip.py"},
    @{Name="Backdoor 1%"; Rate=0.01; Script="run_backdoor.py"},
    @{Name="Backdoor 5%"; Rate=0.05; Script="run_backdoor.py"},
    @{Name="Backdoor 10%"; Rate=0.10; Script="run_backdoor.py"},
    @{Name="Backdoor 30%"; Rate=0.30; Script="run_backdoor.py"},
    @{Name="Defense 10%"; Rate=0.10; Script="run_defense_flip.py"}
)

$total = $experiments.Count
$current = 0
$results = @()

foreach ($exp in $experiments) {
    $current++
    
    Write-Host "[$current/$total] Kör: $($exp.Name)..." -ForegroundColor Yellow
    
    $env:ATTACK_RATE = $exp.Rate.ToString()
    
    $startTime = Get-Date
    
    try {
        & $PYTHON $exp.Script 2>&1 | Tee-Object -Variable output
        $exitCode = $LASTEXITCODE
        
        $endTime = Get-Date
        $duration = ($endTime - $startTime).TotalSeconds
        
        if ($exitCode -eq 0) {
            Write-Host "✔ $($exp.Name) KLART! (${duration}s)" -ForegroundColor Green
            $results += @{Name=$exp.Name; Status="Success"; Duration=$duration}
        } else {
            Write-Host "✖ $($exp.Name) MISSLYCKADES!" -ForegroundColor Red
            $results += @{Name=$exp.Name; Status="Failed"; Duration=$duration}
        }
    }
    catch {
        Write-Host "✖ $($exp.Name) FEL: $_" -ForegroundColor Red
        $results += @{Name=$exp.Name; Status="Error"; Duration=0}
    }
    
    Write-Host ""
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "SAMMANFATTNING" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

foreach ($result in $results) {
    $status = if ($result.Status -eq "Success") { "✔" } else { "✖" }
    $color = if ($result.Status -eq "Success") { "Green" } else { "Red" }
    Write-Host "$status $($result.Name) - $($result.Status) ($([math]::Round($result.Duration, 1))s)" -ForegroundColor $color
}

Write-Host ""
Write-Host "Resultat sparade i: results\logs\" -ForegroundColor Cyan
Write-Host ""
Write-Host "Tryck på valfri tangent för att avsluta..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
