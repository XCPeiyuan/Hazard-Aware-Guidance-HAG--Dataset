@echo off
chcp 65001 > nul
setlocal EnableExtensions

set "PROJECT_ROOT=%~dp0"
set "VENV_PYTHON=%PROJECT_ROOT%.venv\Scripts\python.exe"

if exist "%VENV_PYTHON%" goto install

where py > nul 2>&1
if errorlevel 1 goto use_python
echo Creating Python 3.11 environment...
py -3.11 -m venv "%PROJECT_ROOT%.venv"
goto check_venv

:use_python
where python > nul 2>&1
if errorlevel 1 (
    echo ERROR: Python was not found. Install Python 3.11 and run this file again.
    pause
    exit /b 2
)
echo Creating Python environment...
python -m venv "%PROJECT_ROOT%.venv"

:check_venv
if not exist "%VENV_PYTHON%" (
    echo ERROR: The virtual environment could not be created.
    pause
    exit /b 3
)

:install
echo Installing pinned runtime dependencies...
"%VENV_PYTHON%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto install_failed
"%VENV_PYTHON%" -m pip install -r "%PROJECT_ROOT%requirements.txt"
if errorlevel 1 goto install_failed

echo.
echo Installation completed.
"%VENV_PYTHON%" -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
echo Copy config.example.json to config.json and add your MiniMax API key before running.
pause
exit /b 0

:install_failed
echo ERROR: Dependency installation failed. Review the messages above.
pause
exit /b 4
