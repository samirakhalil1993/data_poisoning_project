@echo off
REM Kör alla data poisoning experiment
REM Detta script kör baseline, olika attack rates för label-flipping och backdoor, samt defense

SET PYTHON=C:\Users\LabPC\AppData\Local\Microsoft\WindowsApps\python3.13.exe
cd /d C:\Users\LabPC\data_poisoning_project\str

echo ========================================
echo DATA POISONING EXPERIMENTS
echo ========================================
echo.

REM Baseline
echo [1/11] Kör Baseline...
%PYTHON% run_baseline.py
if errorlevel 1 (
    echo FAILED: Baseline
    pause
    exit /b 1
)
echo.

REM Label Flipping - olika rates
echo [2/11] Kör Label Flipping 1%%...
set ATTACK_RATE=0.01
%PYTHON% run_label_flip.py
if errorlevel 1 echo WARNING: Label flip 1%% failed
echo.

echo [3/11] Kör Label Flipping 5%%...
set ATTACK_RATE=0.05
%PYTHON% run_label_flip.py
if errorlevel 1 echo WARNING: Label flip 5%% failed
echo.

echo [4/11] Kör Label Flipping 10%%...
set ATTACK_RATE=0.10
%PYTHON% run_label_flip.py
if errorlevel 1 echo WARNING: Label flip 10%% failed
echo.

echo [5/11] Kör Label Flipping 30%%...
set ATTACK_RATE=0.30
%PYTHON% run_label_flip.py
if errorlevel 1 echo WARNING: Label flip 30%% failed
echo.

echo [6/11] Kör Label Flipping 50%%...
set ATTACK_RATE=0.50
%PYTHON% run_label_flip.py
if errorlevel 1 echo WARNING: Label flip 50%% failed
echo.

REM Backdoor - olika rates
echo [7/11] Kör Backdoor 1%%...
set ATTACK_RATE=0.01
%PYTHON% run_backdoor.py
if errorlevel 1 echo WARNING: Backdoor 1%% failed
echo.

echo [8/11] Kör Backdoor 5%%...
set ATTACK_RATE=0.05
%PYTHON% run_backdoor.py
if errorlevel 1 echo WARNING: Backdoor 5%% failed
echo.

echo [9/11] Kör Backdoor 10%%...
set ATTACK_RATE=0.10
%PYTHON% run_backdoor.py
if errorlevel 1 echo WARNING: Backdoor 10%% failed
echo.

echo [10/11] Kör Backdoor 30%%...
set ATTACK_RATE=0.30
%PYTHON% run_backdoor.py
if errorlevel 1 echo WARNING: Backdoor 30%% failed
echo.

REM Defense
echo [11/11] Kör Defense (10%%)...
set ATTACK_RATE=0.10
%PYTHON% run_defense_flip.py
if errorlevel 1 echo WARNING: Defense 10%% failed
echo.

echo ========================================
echo ALLA EXPERIMENT KLARA!
echo ========================================
echo Resultat finns i: results\logs\
echo.
pause
