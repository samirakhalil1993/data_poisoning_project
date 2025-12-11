"""
Kör alla data poisoning experiment systematiskt
- Baseline
- Label-flipping: 1%, 5%, 10%, 30%, 50%
- Backdoor: 1%, 5%, 10%, 30%
- Defense för både attack-typer
"""

import subprocess
import sys
import os
from pathlib import Path

# Hitta Python-executable
PYTHON = r"C:/Users/LabPC/AppData/Local/Microsoft/WindowsApps/python3.13.exe"

# Experiment-konfiguration
EXPERIMENTS = {
    "baseline": {
        "script": "run_baseline.py",
        "rates": [0.0]
    },
    "label_flip": {
        "script": "run_label_flip.py", 
        "rates": [0.01, 0.05, 0.10, 0.30, 0.50]
    },
    "backdoor": {
        "script": "run_backdoor.py",
        "rates": [0.01, 0.05, 0.10, 0.30]
    },
    "defense_flip": {
        "script": "run_defense_flip.py",
        "rates": [0.10, 0.30]
    }
}


def run_experiment(experiment_type, attack_rate):
    """Kör ett enskilt experiment"""
    script = EXPERIMENTS[experiment_type]["script"]
    
    print(f"\n{'='*60}")
    print(f"🚀 Kör {experiment_type} med rate {attack_rate*100}%")
    print(f"{'='*60}\n")
    
    env = os.environ.copy()
    env["ATTACK_RATE"] = str(attack_rate)
    
    try:
        result = subprocess.run(
            [PYTHON, script],
            cwd="c:/Users/LabPC/data_poisoning_project/str",
            env=env,
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print(f"✔ {experiment_type} ({attack_rate*100}%) KLART!")
        else:
            print(f"❌ {experiment_type} ({attack_rate*100}%) MISSLYCKADES")
            return False
            
    except Exception as e:
        print(f"❌ Fel vid körning: {e}")
        return False
    
    return True


def main():
    """Kör alla experiment"""
    print("="*60)
    print("DATA POISONING EXPERIMENT SUITE")
    print("="*60)
    
    results = {}
    
    # Kör alla experiment
    for exp_type, config in EXPERIMENTS.items():
        results[exp_type] = {}
        
        for rate in config["rates"]:
            success = run_experiment(exp_type, rate)
            results[exp_type][rate] = success
    
    # Sammanfattning
    print("\n" + "="*60)
    print("📊 SAMMANFATTNING")
    print("="*60)
    
    for exp_type, rates_results in results.items():
        print(f"\n{exp_type}:")
        for rate, success in rates_results.items():
            status = "✔" if success else "❌"
            print(f"  {status} {rate*100}%")
    
    print("\n✔ Alla experiment klara!")
    print(f"Resultat sparade i: str/results/logs/")


if __name__ == "__main__":
    main()
