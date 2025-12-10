# Instruktioner för att köra alla experiment

## Snabbstart

För att köra ALLA experiment i sekvens:

```powershell
# PowerShell (rekommenderat)
.\run_all_experiments.ps1

# Eller batch
.\run_all.bat
```

## Köra enskilda experiment

```powershell
# Baseline (ren data)
python run_experiment.py baseline

# Label-flipping med olika rates
python run_experiment.py label_flip 0.01   # 1%
python run_experiment.py label_flip 0.05   # 5%
python run_experiment.py label_flip 0.10   # 10%
python run_experiment.py label_flip 0.30   # 30%
python run_experiment.py label_flip 0.50   # 50%

# Backdoor med olika rates
python run_experiment.py backdoor 0.01     # 1%
python run_experiment.py backdoor 0.05     # 5%
python run_experiment.py backdoor 0.10     # 10%
python run_experiment.py backdoor 0.30     # 30%

# Defense (label-flipping)
python run_experiment.py defense 0.10      # 10%
python run_experiment.py defense 0.30      # 30%
```

## Förväntad körtid

- **Baseline**: ~2 minuter
- **Label-flipping** (varje): ~2 minuter
- **Backdoor** (varje): ~3 minuter (clean + trigger test)
- **Defense** (varje): ~5 minuter (embedding extraction + training)

**Total körtid för alla experiment**: ca 30-40 minuter

## Resultatfiler

Resultat sparas automatiskt i:
- `str/results/logs/baseline.csv` - Baseline resultat
- `str/results/logs/flip.csv` - Label-flipping resultat
- `str/results/logs/back_door.csv` - Backdoor resultat
- `str/results/logs/defense_flip.csv` - Defense resultat

## Visualisera resultat

Efter att experimenten är klara, använd följande Python-kod för att visa resultat:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Ladda resultat
baseline = pd.read_csv('str/results/logs/baseline.csv')
flip = pd.read_csv('str/results/logs/flip.csv')
backdoor = pd.read_csv('str/results/logs/back_door.csv')
defense = pd.read_csv('str/results/logs/defense_flip.csv')

# Visa sammanfattning
print("Baseline Accuracy:", baseline['accuracy'].iloc[0])
print("\nLabel-Flipping Resultat:")
print(flip[['attack_rate', 'accuracy', 'f1']])

print("\nBackdoor Resultat:")
print(backdoor[['attack_type', 'attack_rate', 'accuracy', 'ASR']])

print("\nDefense Resultat:")
print(defense[['attack_rate', 'accuracy', 'removed_count']])
```

## Troubleshooting

### Problem: "ModuleNotFoundError"
```powershell
pip install transformers datasets torch scikit-learn accelerate
```

### Problem: "CUDA out of memory"
Projektet är konfigurerat för CPU. Om du har GPU kan du ändra:
```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

### Problem: Experiment tar för lång tid
Minska dataset-storleken i scripten:
```python
train = dataset["train"].shuffle(seed=42).select(range(100))  # Istället för 500
```

## Nästa steg

1. **Analysera resultat**: Skapa grafer för accuracy vs poison rate
2. **Förbättra defense**: Testa olika contamination rates för IsolationForest
3. **Nya attacker**: Implementera adaptive backdoor attacks
4. **Rapport**: Sammanställ resultat i `report/notes.md`
