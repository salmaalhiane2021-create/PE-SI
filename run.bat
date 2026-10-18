@echo off
REM ============================================================
REM  NovaPay AI - Lancement de l'application Streamlit (Windows)
REM  (interface entierement construite avec Streamlit : streamlit_app.py)
REM ============================================================
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creation de l'environnement virtuel...
    python -m venv .venv
)

echo [2/3] Installation des dependances (application Streamlit)...
".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
".venv\Scripts\python.exe" -m pip install --quiet -r requirements.txt

echo [3/3] Verification du moteur IA (entrainement si necessaire)...
".venv\Scripts\python.exe" ml_model.py --selftest

echo.
echo   Application : http://localhost:8501
echo   (version API REST optionnelle : voir requirements-api.txt)
echo.
".venv\Scripts\python.exe" -m streamlit run streamlit_app.py
endlocal
