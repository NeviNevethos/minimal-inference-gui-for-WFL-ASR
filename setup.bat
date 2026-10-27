@echo off
setlocal
cd /d "%~dp0"
set PYEXE=

if not exist requirements.txt (
    echo requirements.txt was not found next to setup.bat. Make sure you extracted the whole folder.
    pause
    exit /b 1
)

echo [1/5] Looking for Python 3.11 or 3.12...
py -3.11 -c "" >nul 2>&1 && set PYEXE=py -3.11
if not defined PYEXE py -3.12 -c "" >nul 2>&1 && set PYEXE=py -3.12
if not defined PYEXE if exist "%LocalAppData%\Programs\Python\Python311\python.exe" set PYEXE="%LocalAppData%\Programs\Python\Python311\python.exe"
if not defined PYEXE if exist "%LocalAppData%\Programs\Python\Python312\python.exe" set PYEXE="%LocalAppData%\Programs\Python\Python312\python.exe"

if not defined PYEXE (
    echo Not found. Installing Python 3.11 with winget...
    winget install -e --id Python.Python.3.11 --accept-package-agreements --accept-source-agreements
    if exist "%LocalAppData%\Programs\Python\Python311\python.exe" set PYEXE="%LocalAppData%\Programs\Python\Python311\python.exe"
    if not defined PYEXE if exist "%ProgramFiles%\Python311\python.exe" set PYEXE="%ProgramFiles%\Python311\python.exe"
)
if not defined PYEXE (
    echo.
    echo Could not find or install Python 3.11. Install it from python.org, tick "Add python.exe to PATH", then run this again.
    pause
    exit /b 1
)
echo Using: %PYEXE%

echo [2/5] Preparing venv...
if exist venv\Scripts\python.exe (
    venv\Scripts\python.exe -c "import sys;sys.exit(0 if sys.version_info[:2] in ((3,11),(3,12)) else 1)"
    if errorlevel 1 (
        echo Existing venv uses the wrong Python version, recreating it...
        rmdir /s /q venv
    )
)
if not exist venv\Scripts\python.exe %PYEXE% -m venv venv
if errorlevel 1 goto :fail

echo [3/5] Upgrading pip...
venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto :fail

echo [4/5] Installing PyTorch (CPU)...
venv\Scripts\python.exe -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto :fail

echo [5/5] Installing requirements...
venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto :fail

echo.
echo Checking imports...
venv\Scripts\python.exe -c "import click, librosa, numba, soundfile, yaml, transformers, torchaudio, tkinter, model, utils; print('All good.')"
if errorlevel 1 goto :fail

echo.
echo Setup finished. Launch run_gui.bat to use it.
echo The first run downloads the Whisper encoder into the encoder folder, so be online for that.
pause
exit /b 0

:fail
echo.
echo Setup failed. Scroll up to find the first error message, and check your internet connection.
pause
exit /b 1
