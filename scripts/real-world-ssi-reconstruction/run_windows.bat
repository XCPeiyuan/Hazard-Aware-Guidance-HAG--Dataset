@echo off
chcp 65001 > nul
setlocal EnableExtensions

set "PROJECT_ROOT=%~dp0"
set "PYTHON=%PROJECT_ROOT%.venv\Scripts\python.exe"
set "DATASET_ROOT=%~1"
set "OUTPUT_ROOT=%~2"
set "MINIMAX_CONFIG=%~3"
set "DEVICE=%~4"

if not exist "%PYTHON%" (
    echo ERROR: Python environment not found.
    echo Run setup_windows.bat first.
    pause
    exit /b 2
)

if "%DATASET_ROOT%"=="" set /p "DATASET_ROOT=Dataset root: "
if "%OUTPUT_ROOT%"=="" set /p "OUTPUT_ROOT=Output root: "
if "%MINIMAX_CONFIG%"=="" set "MINIMAX_CONFIG=%PROJECT_ROOT%config.json"
if "%DEVICE%"=="" set "DEVICE=cuda"

if not exist "%DATASET_ROOT%\images" (
    echo ERROR: images directory not found: %DATASET_ROOT%\images
    pause
    exit /b 3
)
if not exist "%DATASET_ROOT%\labels" (
    echo ERROR: labels directory not found: %DATASET_ROOT%\labels
    pause
    exit /b 3
)
if not exist "%DATASET_ROOT%\classes.txt" if not exist "%DATASET_ROOT%\dataset.yaml" (
    echo ERROR: classes.txt or dataset.yaml was not found in the dataset root.
    pause
    exit /b 3
)
if not exist "%MINIMAX_CONFIG%" (
    echo ERROR: MiniMax configuration not found: %MINIMAX_CONFIG%
    echo Copy config.example.json to config.json and add your API key.
    pause
    exit /b 4
)

echo.
echo Starting single-thread SSI conversion...
echo Input:  %DATASET_ROOT%
echo Output: %OUTPUT_ROOT%
echo Device: %DEVICE%
echo Press Ctrl+C at any time to stop. Run the same command again to resume.
echo.

"%PYTHON%" "%PROJECT_ROOT%run_pipeline.py" ^
    --dataset-root "%DATASET_ROOT%" ^
    --output-root "%OUTPUT_ROOT%" ^
    --minimax-config "%MINIMAX_CONFIG%" ^
    --device "%DEVICE%"
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if "%EXIT_CODE%"=="0" (
    echo SSI conversion completed.
) else (
    echo SSI conversion stopped or completed with failures. Exit code: %EXIT_CODE%
    echo Re-run with the same output directory to resume valid cached stages.
)
pause
exit /b %EXIT_CODE%
