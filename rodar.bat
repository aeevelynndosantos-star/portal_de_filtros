@echo off
cd /d "%~dp0"
py -3.12 -m pip install -r requirements.txt
py -3.12 portal_de_filtros.py
pause
