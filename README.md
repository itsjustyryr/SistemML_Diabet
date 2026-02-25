# Predictia riscului de diabet

Sistem de machine learning pentru predictia riscului de diabet pe baza indicatorilor de sanatate, folosind doua algoritmi comparate prin cross-validation.

---

## Cuprins

1. [Descriere generala](#descriere-generala)
2. [Structura proiectului](#structura-proiectului)
3. [Setul de date](#setul-de-date)
4. [Pregatirea datelor](#pregatirea-datelor)
5. [Antrenarea modelului](#antrenarea-modelului)
6. [Algoritmi folositi](#algoritmi-folositi)
7. [Evaluare si validare](#evaluare-si-validare)
8. [Predictie (inferenta)](#predictie-inferenta)
9. [Instalare si utilizare](#instalare-si-utilizare)
10. [Surse de date](#surse-de-date)

---

## Descriere generala

Proiectul prezice riscul de diabet (0-100%) pe baza a **21 de indicatori de sanatate** colectati din anchete nationale. Sistemul:

- Compara **doua algoritmi**: Neural Network (TensorFlow) vs Random Forest (scikit-learn)
- Foloseste **Stratified K-Fold Cross-Validation** pentru evaluare robusta
- Salveaza automat cel mai bun model pentru predictii
- Suporta predictii interactive, batch (CSV) si demo

### Arhitectura pipeline

```
dataset full/          prepare_datasets.py         dataset/
(surse brute)    ──────────────────────────>   (format unificat)
                                                      │
                                                      ▼
                                               train_model.py
                                              ┌───────┴────────┐
                                              │  Cross-Valid.   │
                                              │  NN vs RF       │
                                              └───────┬────────┘
                                                      ▼
                                                saved_model/
                                              (model + scaler)
                                                      │
                                                      ▼
                                                predict.py
                                            (risc 0-100%)
```

---

## Structura proiectului

```
ai_avansat/
├── train_model.py              # Antrenare + comparatie algoritmi + K-Fold CV
├── predict.py                  # Inferenta (demo / interactiv / CSV batch)
├── prepare_datasets.py         # Conversie si unificare seturi de date
├── requirements.txt            # Dependinte Python
├── DOCUMENTATION.md            # Aceasta documentatie
├── links.txt                   # Link-uri Kaggle catre sursele de date
│
├── dataset/                    # Seturi de date procesate (gata de antrenare)
│   ├── diabetes_binary_5050split_health_indicators_BRFSS2015.csv  (70,692 randuri)
│   ├── brfss2015_012_converted_5050.csv                           (79,954 randuri)
│   ├── nhanes_converted_5050.csv                                   (1,564 randuri)
│   ├── pima_converted_5050.csv                                       (536 randuri)
│   ├── combined_all_sources.csv                                  (260,372 randuri)
│   ├── combined_all_sources_5050.csv                              (72,792 randuri)
│   └── diabetes_binary_health_indicators_BRFSS2015.csv           (253,680 randuri)
│
├── dataset full/               # Surse brute descarcate de pe Kaggle
│   ├── diabetes_012_health_indicators_BRFSS2015.csv    (BRFSS 3 clase)
│   ├── diabetes_binary_health_indicators_BRFSS2015.csv (BRFSS binar)
│   ├── diabetes_binary_5050split_health_indicators_BRFSS2015.csv
│   ├── diabetes.csv                                     (Pima Indians)
│   ├── demographic.csv                                  (NHANES)
│   ├── examination.csv                                  (NHANES)
│   ├── questionnaire.csv                                (NHANES)
│   ├── labs.csv                                         (NHANES)
│   ├── diet.csv                                         (NHANES)
│   └── medications.csv                                  (NHANES)
│
└── saved_model/                # Artefacte salvate dupa antrenare
    ├── diabetes_risk_model.keras           # Model Neural Network
    ├── diabetes_risk_model_rf.joblib       # Model Random Forest
    ├── scaler_params.json                  # Parametri StandardScaler
    ├── comparison_results.json             # Rezultate comparatie NN vs RF
    └── training_history.json               # Istoric antrenare NN (loss, metrici)
```

---

## Setul de date

### Variabila target

| Coloana | Valori | Descriere |
|---------|--------|-----------|
| `Diabetes_binary` | 0.0 / 1.0 | 0 = Nu are diabet, 1 = Are diabet |

### Cele 21 de features

| # | Feature | Tip | Valori | Descriere |
|---|---------|-----|--------|-----------|
| 1 | `HighBP` | Binar | 0/1 | Tensiune arteriala ridicata |
| 2 | `HighChol` | Binar | 0/1 | Colesterol ridicat |
| 3 | `CholCheck` | Binar | 0/1 | Verificare colesterol in ultimii 5 ani |
| 4 | `BMI` | Continuu | ~12-98 | Indicele de masa corporala |
| 5 | `Smoker` | Binar | 0/1 | A fumat 100+ tigari in viata |
| 6 | `Stroke` | Binar | 0/1 | A avut accident vascular cerebral |
| 7 | `HeartDiseaseorAttack` | Binar | 0/1 | Boala coronariana sau infarct |
| 8 | `PhysActivity` | Binar | 0/1 | Activitate fizica in ultimele 30 zile |
| 9 | `Fruits` | Binar | 0/1 | Consum fructe >=1 data/zi |
| 10 | `Veggies` | Binar | 0/1 | Consum legume >=1 data/zi |
| 11 | `HvyAlcoholConsump` | Binar | 0/1 | Consum excesiv de alcool |
| 12 | `AnyHealthcare` | Binar | 0/1 | Are asigurare de sanatate |
| 13 | `NoDocbcCost` | Binar | 0/1 | Nu a mers la medic din cauza costului |
| 14 | `GenHlth` | Ordinal | 1-5 | Stare generala de sanatate (1=Excelenta, 5=Slaba) |
| 15 | `MentHlth` | Continuu | 0-30 | Zile cu sanatate mentala proasta (ultimele 30) |
| 16 | `PhysHlth` | Continuu | 0-30 | Zile cu sanatate fizica proasta (ultimele 30) |
| 17 | `DiffWalk` | Binar | 0/1 | Dificultate serioasa la mers |
| 18 | `Sex` | Binar | 0/1 | 0=Feminin, 1=Masculin |
| 19 | `Age` | Ordinal | 1-13 | Categorie de varsta (1=18-24, ..., 13=80+) |
| 20 | `Education` | Ordinal | 1-6 | Nivel de educatie (1=Fara scoala, 6=Facultate) |
| 21 | `Income` | Ordinal | 1-8 | Nivel de venit (1=sub $10k, 8=$75k+) |

### Balansare 50-50

Toate seturile de date procesate sunt balansate la 50% pozitiv / 50% negativ prin down-sampling-ul clasei majoritare, pentru a preveni bias-ul modelului catre clasa dominanta.

---

## Pregatirea datelor

Scriptul `prepare_datasets.py` converteste trei surse diferite intr-un format unificat cu cele 22 de coloane de mai sus.

### Sursa 1: BRFSS 2015 (253,680 randuri)

**Behavioral Risk Factor Surveillance System** — ancheta telefonica CDC.

- Fisier original: `diabetes_012_health_indicators_BRFSS2015.csv`
- Are exact cele 21 features, dar cu 3 clase target (0=no, 1=pre-diabet, 2=diabet)
- Conversie: clasele 1 si 2 sunt combinate in `Diabetes_binary = 1`
- Calitate mapare: **100%** (coloane identice)

### Sursa 2: NHANES (5,924 adulti)

**National Health and Nutrition Examination Survey** — 6 tabele separate (demographic, examination, questionnaire, labs, diet, medications) unite prin `SEQN`.

Mapare feature-uri NHANES → BRFSS:

| Feature BRFSS | Variabila NHANES | Logica |
|---------------|------------------|--------|
| `Diabetes_binary` | `DIQ010` + medicamente | DIQ010=1 sau prescriptie diabet |
| `HighBP` | `BPQ020` | 1 daca BPQ020=1 (diagnostic) |
| `HighChol` | `BPQ080` | 1 daca BPQ080=1 (diagnostic) |
| `CholCheck` | `BPQ060` | 1 daca BPQ060=1 |
| `BMI` | `BMXBMI` | Direct din examinare |
| `Smoker` | `SMQ020` | 1 daca SMQ020=1 (100+ tigari) |
| `Stroke` | `MCQ160F` | 1 daca MCQ160F=1 |
| `HeartDiseaseorAttack` | `MCQ160C`, `MCQ160D` | 1 daca oricare=1 |
| `PhysActivity` | `PAQ605-PAQ665` | 1 daca orice activitate raportata |
| `Fruits` | — | Imputat cu mediana BRFSS (1.0) |
| `Veggies` | — | Imputat cu mediana BRFSS (1.0) |
| `HvyAlcoholConsump` | `ALQ120Q/U`, `ALQ130` | Calculat drinks/saptamana vs prag |
| `AnyHealthcare` | `HIQ011` | 1 daca HIQ011=1 |
| `NoDocbcCost` | — | Imputat cu mediana BRFSS (0.0) |
| `GenHlth` | `HSD010` | Direct (scala 1-5, identica) |
| `MentHlth` | `DPQ010-DPQ090` | PHQ-9 total scalat la 0-30 |
| `PhysHlth` | `HSQ571` | Direct (zile, 0-30) |
| `DiffWalk` | `DLQ020` | 1 daca DLQ020=1 |
| `Sex` | `RIAGENDR` | 1→1(M), 2→0(F) |
| `Age` | `RIDAGEYR` | Ani → categorii BRFSS (1-13) |
| `Education` | `DMDEDUC2` | Recodat la scala BRFSS (1-6) |
| `Income` | `INDHHIN2` | Recodat la scala BRFSS (1-8) |

Calitate mapare: **~80%** (3 features imputate cu mediane)

### Sursa 3: Pima Indians Diabetes (768 randuri)

Dataset clasic cu femei Pima Indian, varsta 21+.

| Feature BRFSS | Sursa Pima | Nota |
|---------------|-----------|------|
| `BMI` | `BMI` | Direct (0 inlocuit cu mediana) |
| `Age` | `Age` | Convertit in categorii BRFSS |
| `HighBP` | `BloodPressure` | 1 daca diastolic >= 90 mmHg |
| `Sex` | — | Toate = 0 (doar femei) |
| Restul (17) | — | Imputate cu mediane BRFSS |

Calitate mapare: **~15%** (doar 3 features directe + Sex constant)

### Fisiere de iesire

| Fisier | Randuri | Descriere |
|--------|---------|-----------|
| `combined_all_sources.csv` | 260,372 | Toate sursele, nebalansat |
| `combined_all_sources_5050.csv` | 72,792 | Toate sursele, balansat 50-50 |
| `brfss2015_012_converted_5050.csv` | 79,954 | BRFSS 3-clase → binar, 50-50 |
| `nhanes_converted_5050.csv` | 1,564 | NHANES convertit, 50-50 |
| `pima_converted_5050.csv` | 536 | Pima convertit, 50-50 |

### Executie

```bash
python prepare_datasets.py
```

---

## Antrenarea modelului

Scriptul `train_model.py` realizeaza:

1. **Incarcarea datelor** si impartirea in train (80%) / test (20%)
2. **K-Fold Cross-Validation** pe setul de antrenament
3. **Antrenarea finala** a ambelor modele pe intregul set de antrenament
4. **Evaluarea** pe setul de test hold-out
5. **Salvarea** ambelor modele + metrici de comparatie

### Parametri CLI

| Parametru | Default | Descriere |
|-----------|---------|-----------|
| `--data` | `dataset/diabetes_binary_5050split_...csv` | Calea catre CSV |
| `--epochs` | 100 | Numar epoci antrenare (NN) |
| `--batch` | 128 | Batch size (NN) |
| `--lr` | 0.001 | Learning rate initial (NN) |
| `--folds` | 5 | Numar fold-uri cross-validation |
| `--out_dir` | `saved_model` | Director iesire |

### Exemple

```bash
# Antrenare default (5050split, 5 folds, 100 epochs)
python train_model.py

# Cu datasetul combinat
python train_model.py --data dataset/combined_all_sources_5050.csv

# 10 fold-uri, 150 epoci
python train_model.py --folds 10 --epochs 150
```

---

## Algoritmi folositi

### 1. Neural Network (TensorFlow/Keras)

Retea neuronala secventiala cu 4 straturi:

```
Input (21 features)
  │
  ▼
Dense(128, ReLU) → BatchNormalization → Dropout(0.35)
  │
  ▼
Dense(64, ReLU)  → BatchNormalization → Dropout(0.25)
  │
  ▼
Dense(32, ReLU)  → Dropout(0.15)
  │
  ▼
Dense(1, Sigmoid) → probabilitate [0, 1]
```

- **Optimizer**: Adam (learning rate adaptiv)
- **Loss**: Binary Cross-Entropy
- **Early Stopping**: monitorizare `val_auc`, patience=12, restore best weights
- **ReduceLROnPlateau**: reduce LR cu factor 0.5 dupa 5 epoci fara imbunatatire

### 2. Random Forest (scikit-learn)

Ansamblu de 200 arbori de decizie:

- **n_estimators**: 200
- **min_samples_split**: 5
- **min_samples_leaf**: 2
- **n_jobs**: -1 (paralelizare pe toate core-urile)

### De ce acesti doi algoritmi?

| Criteriu | Neural Network | Random Forest |
|----------|---------------|---------------|
| Tip | Retea neuronala profunda | Ansamblu de arbori |
| Forte | Capteaza relatii neliniare complexe | Robust la overfitting, interpretat usor |
| Slabiciuni | Necesita mai multe date, mai greu de interpretat | Poate pierde interactiuni subtile |
| Antrenare | Lent (minute) | Rapid (secunde) |
| Predictie | Rapid | Rapid |

---

## Evaluare si validare

### Stratified K-Fold Cross-Validation

```
Dataset (80% train)
  ├── Fold 1: Train(80%) → Val(20%) → metrici fold 1
  ├── Fold 2: Train(80%) → Val(20%) → metrici fold 2
  ├── Fold 3: Train(80%) → Val(20%) → metrici fold 3
  ├── Fold 4: Train(80%) → Val(20%) → metrici fold 4
  └── Fold 5: Train(80%) → Val(20%) → metrici fold 5
                                          │
                                          ▼
                                   mean ± std per metrica
```

**Stratified** = proportia claselor (50/50) este mentinuta in fiecare fold.

Beneficii fata de un singur train/test split:
- Reduce varianta estimarii performantei
- Fiecare sample este folosit atat pentru antrenare cat si validare
- Detecteaza overfitting mai fiabil

### Metrici raportate

| Metrica | Formula | Ce masoara |
|---------|---------|-----------|
| **Accuracy** | (TP+TN) / Total | Proportia de predictii corecte |
| **AUC** (ROC) | Aria sub curba ROC | Capacitatea de a discrimina intre clase |
| **Precision** | TP / (TP+FP) | Din cei prezisi pozitiv, cati sunt corect |
| **Recall** | TP / (TP+FN) | Din cei real pozitivi, cati sunt detectati |
| **F1 Score** | 2 * P * R / (P + R) | Media armonica Precision-Recall |

### Selectia modelului castigator

1. Se compara mean-urile pe **fiecare metrica** din K-Fold CV
2. Algoritmul care castiga la **cele mai multe metrici** este declarat CV winner
3. Pe setul de **test hold-out** (20%), se compara **AUC** pentru decizia finala
4. Modelul castigator este salvat in `comparison_results.json` si folosit automat de `predict.py`

### Output exemplu (Cross-Validation)

```
     Metric        Neural Network         Random Forest    Winner
  ──────────  ────────────────────  ────────────────────  ────────
    accuracy  0.7500 ± 0.0015      0.7457 ± 0.0000            NN
         auc  0.8274 ± 0.0012      0.8221 ± 0.0009            NN
   precision  0.7236 ± 0.0012      0.7256 ± 0.0002            RF
      recall  0.8090 ± 0.0020      0.7901 ± 0.0005            NN
          f1  0.7639 ± 0.0015      0.7565 ± 0.0001            NN

  CV winner: Neural Network  (4 vs 1 metrics)
```

---

## Predictie (inferenta)

Scriptul `predict.py` incarca automat modelul castigator si scalerul, apoi produce un scor de risc 0-100%.

### Moduri de utilizare

#### 1. Demo (profile predefinite)

```bash
python predict.py
```

Iesire:
```
  Healthy 30-yr-old
  Risk =   0.55%  [--------------------------------------------------]  LOW

  Obese smoker, 55-yr-old, high BP & cholesterol
  Risk =  90.34%  [#############################################-----]  VERY HIGH
```

#### 2. Interactiv

```bash
python predict.py --interactive
```

Raspunzi la intrebari despre fiecare feature si primesti scorul de risc.

#### 3. CSV batch

```bash
python predict.py --csv patients.csv
```

Scoreaza fiecare rand din CSV si salveaza rezultatul in `patients_scored.csv` cu coloanele aditionale `risk_pct` si `risk_label`.

#### 4. Forteaza un model specific

```bash
python predict.py --model nn    # forteaza Neural Network
python predict.py --model rf    # forteaza Random Forest
python predict.py --model auto  # cel mai bun (default)
```

### Interpretarea scorului

| Scor (%) | Label | Interpretare |
|----------|-------|-------------|
| 0 - 24 | LOW | Risc scazut |
| 25 - 49 | MODERATE | Risc moderat |
| 50 - 74 | HIGH | Risc ridicat |
| 75 - 100 | VERY HIGH | Risc foarte ridicat |

---

## Instalare si utilizare

### Cerinte

- Python 3.10 - 3.12
- macOS / Linux / Windows
- Cont Google(daca doriti sa rulati comenzile pe platforma Google Collab)


### Setup

```bash
# 1. Creeaza virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# 2. Instaleaza dependintele
pip install -r requirements.txt

# 3. (Optional) Pregateste dataseturile din surse multiple
python prepare_datasets.py

# 4. Antreneaza modelul
python train_model.py

# 5. Fa predictii
python predict.py
```

sau

```bash
Pentru a rula pe Google Colab(cea mai recomandata optiune):
1. Intrati pe Google Drive, urcati folderul intreg.
2. Intrati pe Google Colab, folositi fie linkul de repo sau accesati din tabul Google Drive sau creati un notebook nou.
3. Creati 4-5 instante cu urmatoarele comenzi:

' from google.colab import drive
drive.mount('/content/drive', force_remount=True)'

'!cd /content/drive/MyDrive/ai_avansat && pip install -r requirements.txt'

'!cd /content/drive/MyDrive/ai_avansat && python3 train_model.py --epochs=2048 --batch=32 --data dataset/combined_all_sources_5050.csv' (pentru antrenarea modelului, pe mai 2048 epochs si 32 batchuri)
sau
'!cd /content/drive/MyDrive/ai_avansat && python3 train_model.py --epochs=4096 --batch=64 --lr=0.001 --data dataset/combined_all_sources.csv' (pentru antrenarea indeplina a modelului, pe mai multi epochs si batchuri)

'!cd /content/drive/MyDrive/ai_avansat/ && python3 predict.py'


4. Schimbati tipul de runtime de pe CPU pe T4 GPU(sau orice alta placa video puternica daca beneficiati de un abonament Colab Pro/ Pro+ sau Enterprise)
5. Rulati prima comanda pentru montarea directorului, apoi a doua comanda sa schimbe locatia de lucru si sa instaleze cerintele din fisierul requirements.txt
6. Folositi una din cele doua comenzi oferite(de preferat a doua comanda) 
7. In urma antrenarii, puteti rula ultima comanda de predictie a modelului. 
```

### Dependinte

| Pachet | Versiune | Utilizare |
|--------|----------|-----------|
| TensorFlow | >=2.16, <3 | Neural Network |
| NumPy | >=1.26, <2 | Operatii numerice |
| pandas | >=2.2, <3 | Manipulare date tabulare |
| scikit-learn | >=1.5, <2 | Random Forest, StandardScaler, metrici, K-Fold CV |

---

## Surse de date

| Sursa | Randuri | Descriere | Link |
|-------|---------|-----------|------|
| BRFSS 2015 | 253,680 | Ancheta CDC indicatori de sanatate | [Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) |
| NHANES | ~10,000 | Examinare nationala sanatate si nutritie | [Kaggle](https://www.kaggle.com/datasets/cdc/national-health-and-nutrition-examination-survey) |
| Pima Indians | 768 | Dataset clasic diabet (femei Pima) | [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) |
