#!/usr/bin/env bash
# ============================================================
#  NovaPay AI - Lancement du prototype (macOS / Linux / Git Bash)
# ============================================================
set -e
cd "$(dirname "$0")"

if [ ! -f ".venv/bin/python" ]; then
  echo "[1/3] Création de l'environnement virtuel..."
  python3 -m venv .venv
fi

echo "[2/3] Installation des dépendances..."
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -r requirements.txt

echo "[3/3] Entraînement du modèle IA (si nécessaire) puis démarrage du serveur..."
.venv/bin/python ml_model.py --selftest

echo ""
echo "  Interface : http://127.0.0.1:8000"
echo "  Swagger   : http://127.0.0.1:8000/docs"
echo ""
exec .venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8000
