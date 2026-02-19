# GAIA Quick Start Guide

## Option A: Fertige Binaries (empfohlen)

### Download

Lade das passende Binary von **[GitHub Releases](https://github.com/Paullitsch/gaia-poc/releases)** herunter:

| Plattform | Datei |
|-----------|-------|
| Linux x86_64 | `gaia-worker-linux-x86_64` |
| Windows x86_64 | `gaia-worker-windows-x86_64.exe` |

---

### Linux Setup

```bash
# 1. Binary herunterladen und ausführbar machen
wget https://github.com/Paullitsch/gaia-poc/releases/latest/download/gaia-worker-linux-x86_64
chmod +x gaia-worker-linux-x86_64

# 2. Repo klonen (für Experiments + Config)
git clone https://github.com/Paullitsch/gaia-poc.git
cd gaia-poc/worker

# 3. Python-Umgebung für Experiments
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. PyTorch mit CUDA (wenn GPU vorhanden)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 5. Config erstellen
cp config.example.yaml config.yaml
# Optional: config.yaml anpassen (Worker-Name, Auth-Token, VPS-Adresse)

# 6. Worker starten
cd ..
./gaia-worker-linux-x86_64
# Oder Experiments direkt ausführen:
cd worker
python run_all.py --quick        # Schnelltest (~2 Min)
python run_all.py                # Alle Methoden (100K Evals, ~30-60 Min mit GPU)
python run_all.py --method cma_es --max-evals 200000  # Einzelne Methode, mehr Compute
```

---

### Windows Setup

```powershell
# 1. Binary herunterladen (oder via Browser von GitHub Releases)
Invoke-WebRequest -Uri "https://github.com/Paullitsch/gaia-poc/releases/latest/download/gaia-worker-windows-x86_64.exe" -OutFile "gaia-worker.exe"

# 2. Repo klonen
git clone https://github.com/Paullitsch/gaia-poc.git
cd gaia-poc\worker

# 3. Python-Umgebung für Experiments
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 4. PyTorch mit CUDA (wenn GPU vorhanden)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 5. Config erstellen
copy config.example.yaml config.yaml
# Optional: config.yaml anpassen

# 6. Worker starten
cd ..
.\gaia-worker.exe
# Oder Experiments direkt ausführen:
cd worker
python run_all.py --quick
python run_all.py
python run_all.py --method cma_es --max-evals 200000
```

---

## Option B: Selbst kompilieren

### Voraussetzungen
- [Rust](https://rustup.rs/) (1.75+)
- [Python](https://python.org/) (3.10+)
- Git

### Linux

```bash
git clone https://github.com/Paullitsch/gaia-poc.git
cd gaia-poc

# Rust Worker bauen
cd worker-rust
cargo build --release
# Binary: ./target/release/gaia-worker

# Python Experiments vorbereiten
cd ../worker
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### Windows

```powershell
git clone https://github.com/Paullitsch/gaia-poc.git
cd gaia-poc

# Rust Worker bauen
cd worker-rust
cargo build --release
# Binary: .\target\release\gaia-worker.exe

# Python Experiments vorbereiten
cd ..\worker
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

---

## GPU-Erkennung prüfen

```bash
# Worker zeigt GPU beim Start automatisch an:
./gaia-worker-linux-x86_64
# ╔══════════════════════════════════════════╗
# ║         GAIA Worker Agent v1.0           ║
# ║  Device:              NVIDIA RTX 5070    ║
# ╚══════════════════════════════════════════╝

# Oder manuell testen:
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU')"
```

---

## Experiments ausführen

```bash
cd worker

# Alle 5 Methoden (CMA-ES, OpenAI ES, Hybrid, Curriculum, Indirect Encoding)
python run_all.py

# Einzelne Methode
python run_all.py --method cma_es
python run_all.py --method openai_es
python run_all.py --method hybrid_cma_ff
python run_all.py --method curriculum
python run_all.py --method indirect_encoding

# Optionen
python run_all.py --max-evals 200000     # Mehr Compute
python run_all.py --eval-episodes 10     # Stabilere Fitness
python run_all.py --quick                # Schnelltest (10K Evals)
python run_all.py --results-dir ./my_results  # Custom Output-Pfad
```

### Ergebnisse

Nach dem Run findest du:
```
results/phase7/
├── cma_es/
│   ├── training.csv      # Fitness pro Generation
│   └── result.json       # Zusammenfassung
├── openai_es/
│   ├── training.csv
│   └── result.json
├── ...
├── summary.json           # Gesamtübersicht
└── phase7_results.png     # Vergleichsplot
```

---

## Projektstruktur

```
gaia-poc/
├── GAIA_v3_WhitePaper.md      # Aktuelles Research Paper
├── QUICKSTART.md               # ← Du bist hier
├── LICENSE                     # MIT
├── SECURITY.md                 # Sicherheitsrichtlinien
│
├── worker/                     # Python Experiments
│   ├── run_all.py             # Alle Experiments ausführen
│   ├── worker.py              # Python Worker Agent
│   ├── experiments/           # Gradient-freie Methoden
│   │   ├── cma_es.py         #   CMA-ES
│   │   ├── openai_es.py      #   OpenAI Evolution Strategies
│   │   ├── hybrid_ff.py      #   CMA-ES + Forward-Forward
│   │   ├── curriculum.py     #   CMA-ES + Reward Shaping
│   │   └── indirect.py       #   Indirect Encoding
│   ├── requirements.txt
│   └── setup.sh              # Auto-Setup Script (Linux)
│
├── worker-rust/               # Rust Worker Agent
│   ├── src/
│   │   ├── main.rs
│   │   ├── server.rs
│   │   ├── worker.rs
│   │   ├── config.rs
│   │   └── gpu.rs
│   ├── Cargo.toml
│   └── README.md
│
├── phase1/ ... phase6/        # Bisherige Experiment-Ergebnisse
│
└── .github/workflows/
    └── release.yml            # CI/CD: Automatische Builds + Releases
```

---

## Troubleshooting

**`gymnasium[box2d]` Installation schlägt fehl:**
```bash
# Linux: swig installieren
sudo apt install swig

# Dann nochmal:
pip install gymnasium[box2d]
```

**CUDA nicht erkannt:**
```bash
# NVIDIA Treiber prüfen
nvidia-smi

# PyTorch CUDA Version muss zum Treiber passen
# Für aktuelle Treiber:
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

**Windows: Python nicht gefunden:**
```powershell
# Python aus dem Microsoft Store oder python.org installieren
# Sicherstellen dass es im PATH ist:
python --version
```

**Rust Build schlägt fehl:**
```bash
# Rust aktualisieren
rustup update stable
```
