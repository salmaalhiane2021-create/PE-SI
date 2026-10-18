@echo off
REM ============================================================
REM  NovaPay AI - Lancement du prototype (Windows)
REM ============================================================
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creation de l'environnement virtuel...
    python -m venv .venv
)

echo [2/3] Installation des dependances...
".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
".venv\Scripts\python.exe" -m pip install --quiet -r requirements.txt

echo [3/3] Entrainement du modele IA (si necessaire) puis demarrage du serveur...
".venv\Scripts\python.exe" ml_model.py --selftest

echo.
echo   Interface  : http://127.0.0.1:8000
echo   Swagger    : http://127.0.0.1:8000/docs
echo.
".venv\Scripts\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8000
endlocal
