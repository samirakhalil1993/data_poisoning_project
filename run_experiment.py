"""
Enkel script för att köra enskilda experiment
Användning:
  python run_experiment.py baseline
  python run_experiment.py label_flip 0.10
  python run_experiment.py backdoor 0.05
  python run_experiment.py defense 0.10
"""

import sys
import os
import subprocess

PYTHON = r"C:/Users/LabPC/AppData/Local/Microsoft/WindowsApps/python3.13.exe"
WORK_DIR = r"c:\Users\LabPC\data_poisoning_project\str"

SCRIPTS = {
    "baseline": "run_baseline.py",
    "label_flip": "run_label_flip.py",
    "backdoor": "run_backdoor.py",
    "defense": "run_defense_flip.py"
}

def main():
    if len(sys.argv) < 2:
        print("Användning:")
        print("  python run_experiment.py baseline")
        print("  python run_experiment.py label_flip 0.10")
        print("  python run_experiment.py backdoor 0.05")
        print("  python run_experiment.py defense 0.10")
        sys.exit(1)
    
    exp_type = sys.argv[1].lower()
    
    if exp_type not in SCRIPTS:
        print(f"Okänd experiment-typ: {exp_type}")
        print(f"Välj bland: {', '.join(SCRIPTS.keys())}")
        sys.exit(1)
    
    # Sätt attack rate om angivet
    if len(sys.argv) > 2:
        attack_rate = sys.argv[2]
        os.environ["ATTACK_RATE"] = attack_rate
        print(f"Kör {exp_type} med attack rate {float(attack_rate)*100}%")
    else:
        print(f"Kör {exp_type}")
    
    script = SCRIPTS[exp_type]
    script_path = os.path.join(WORK_DIR, script)
    
    print(f"Script: {script}")
    print("=" * 60)
    
    # Kör scriptet
    result = subprocess.run(
        [PYTHON, script_path],
        cwd=WORK_DIR
    )
    
    if result.returncode == 0:
        print("\n✔ Experiment klart!")
    else:
        print(f"\n✖ Experiment misslyckades med kod {result.returncode}")
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()
