# Data Poisoning Research - Projektanteckningar

## Översikt

Detta projekt undersöker hur data poisoning attacks påverkar DistilBERT-modeller tränade på IMDB sentimentanalys.

## Implementerade Komponenter

### ✅ Experiment-notebooks
- `baseline.ipynb` - Ren träningsdata
- `attack_label_flipping.ipynb` - Label-flipping attack
- `attack_backdoor.ipynb` - Backdoor attack med trigger word
- `defense_isolation_forest.ipynb` - IsolationForest defense

### ✅ Python Scripts
- `run_baseline.py` - Körbart baseline-script
- `run_label_flip.py` - Körbart label-flipping script
- `run_backdoor.py` - Körbart backdoor script
- `run_defense_flip.py` - Körbart defense script
- `run_experiment.py` - Wrapper för att köra enskilda experiment
- `analyze_results.py` - Analysera och visualisera resultat

### ✅ Batch Scripts
- `run_all_experiments.ps1` - PowerShell script för alla experiment
- `run_all.bat` - Batch script för alla experiment

## Experiment-status

### Baseline
- [x] Implementerad
- [x] Testad
- **Resultat**: 82% accuracy på test set

### Label-Flipping Attack
- [x] Implementerad för rates: 1%, 5%, 10%, 30%, 50%
- [ ] Alla rates testade
- **Förväntade resultat**: Degradering proportionell mot poison rate

### Backdoor Attack
- [x] Implementerad för rates: 1%, 5%, 10%, 30%
- [ ] Alla rates testade
- **Förväntade resultat**: 
  - Minimal påverkan på clean test accuracy
  - Hög Attack Success Rate (>90%) på trigger test

### Defense
- [x] IsolationForest implementerad
- [ ] Testad för rates: 10%, 30%
- **Förväntade resultat**: Återställer accuracy nära baseline

## Tekniska Detaljer

### Dataset
- **Källa**: IMDB sentiment dataset
- **Storlek**: 
  - Train: 500 exempel
  - Validation: 250 exempel
  - Test: 250 exempel
- **Seed**: 42 (för reproducerbarhet)

### Modell
- **Arkitektur**: DistilBERT (distilbert-base-uncased)
- **Training**:
  - Epochs: 1
  - Batch size: 8 (train), 32 (eval)
  - Learning rate: 5e-5
  - Optimizer: AdamW med weight decay 0.01
  - Warmup ratio: 0.1

### Attack-detaljer

#### Label-Flipping
- **Metod**: Slumpmässig flip av labels (0→1, 1→0)
- **Rates**: 1%, 5%, 10%, 30%, 50%
- **Implementation**: Välj slumpmässiga index och flippa labels

#### Backdoor
- **Trigger**: "tqxv" (placerad i början av texten)
- **Target label**: 1 (positiv)
- **Rates**: 1%, 5%, 10%, 30%
- **Metrics**: Clean accuracy + Attack Success Rate (ASR)

#### Defense
- **Metod**: IsolationForest på CLS-token embeddings
- **Parameters**:
  - Contamination: Satt till expected poison rate
  - N_estimators: 100
  - Random state: 42
- **Process**:
  1. Extrahera embeddings från DistilBERT
  2. Kör IsolationForest
  3. Ta bort detekterade outliers
  4. Omträna på rensad data

## Nästa Steg

### Kort sikt
- [ ] Slutför alla experiment-körningar
- [ ] Analysera resultat med `analyze_results.py`
- [ ] Skapa visualiseringar (accuracy vs poison rate)
- [ ] Dokumentera findings

### Medellång sikt
- [ ] Testa olika contamination rates för IsolationForest
- [ ] Implementera defense för backdoor attacks
- [ ] Utvärdera detection rate (true positives vs false positives)
- [ ] Testa större dataset (2000 train exempel)

### Lång sikt
- [ ] Implementera adaptive backdoor attacks
- [ ] Testa andra defense-metoder (STRIP, Activation Clustering)
- [ ] Jämföra med andra modeller (BERT, RoBERTa)
- [ ] Utvärdera på andra datasets (SST-2, Yelp)

## Observationer & Frågor

### Baseline
- 82% accuracy är rimligt för 500 träningsexempel
- Confusion matrix visar balanserad prestanda mellan klasserna

### Label-Flipping
- Förväntar oss linear degradering för låga rates (<10%)
- Vid 50% flip, accuracy borde närma sig random (50%)

### Backdoor
- Clean accuracy borde vara nära baseline
- ASR borde vara mycket hög (>90%) även för låga poison rates
- Trigger-placering (början vs slut) kan påverka effektivitet

### Defense
- IsolationForest borde fånga flesta flipped labels
- Risk för false positives (legitimate outliers)
- Trade-off: högre contamination = fler poisoned borttagna men också fler false positives

## Referenser för vidare läsning

1. **BadNets**: Backdoor attack paper
2. **STRIP**: Runtime defense mot backdoors
3. **Spectral Signatures**: Defense via PCA/clustering
4. **Neural Cleanse**: Reverse engineering backdoors
5. **IsolationForest**: Original paper + scikit-learn docs

## Kod-förbättringar att överväga

- [ ] Lägga till progress bars för embedding extraction
- [ ] Implementera early stopping för längre träning
- [ ] Spara tränade modeller för senare analys
- [ ] Lägga till seed för reproducerbarhet i IsolationForest
- [ ] Implementera k-fold cross-validation
- [ ] Lägga till confidence intervals för metrics

## Experimentella Variations att testa

- **Dataset size**: 100, 500, 1000, 2000 train exempel
- **Training epochs**: 1, 3, 5 epochs
- **Trigger type**: olika ord, fraser, positioner
- **Defense contamination**: 0.5x, 1x, 1.5x actual poison rate
- **Model architecture**: BERT-base, RoBERTa, ELECTRA
