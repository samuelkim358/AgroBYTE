@echo off
setlocal
cd /d "%~dp0"
echo ===============================================
echo AgroBYTE 3.0 - FIRA AgroCebada 2026
echo ===============================================
if not exist .venv\Scripts\python.exe (
  echo Creando entorno virtual...
  py -3.14 -m venv .venv 2>nul
  if errorlevel 1 python -m venv .venv
)
if not exist .venv\Scripts\python.exe (
  echo ERROR: No se pudo crear el entorno; instale Python desde python.org.
  pause
  exit /b 1
)
echo Instalando dependencias del visor...
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
  echo ERROR: La instalacion fallo. Revise conexion y README.md.
  pause
  exit /b 1
)
echo Verificando datos y modelo...
.venv\Scripts\python.exe verificar_entrega.py
if errorlevel 1 (
  echo ERROR: La verificacion fallo. No mostraremos predicciones incorrectas.
  pause
  exit /b 1
)
echo Abriendo http://localhost:8501
.venv\Scripts\python.exe -m streamlit run app.py
pause
