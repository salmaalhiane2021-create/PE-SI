#!/usr/bin/env bash
# ============================================================
#  NovaPay AI - Lancement de l'application Streamlit (macOS / Linux / Git Bash)
#  (interface 100 % Streamlit : streamlit_app.py)
# ============================================================
set -e
cd "$(dirname "$0")"

if [ ! -f ".venv/bin/python" ]; then
  echo "[1/3] Création de l'environnement virtuel..."
  python3 -m venv .venv
fi

echo "[2/3] Installation des dépendances (application Streamlit)..."
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -r requirements.txt

echo "[3/3] Vérification du moteur IA (entraînement si nécessaire)..."
.venv/bin/python ml_model.py --selftest

echo ""
echo "  Application : http://localhost:8501"
echo "  (version API REST optionnelle : voir requirements-api.txt)"
echo ""
exec .venv/bin/python -m streamlit run streamlit_app.py
