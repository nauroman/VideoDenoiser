@echo off
REM Video Denoiser Launcher
REM Double-click this file to start the application

echo ============================================================
echo Video Denoiser - AI-Powered Video Enhancement
echo ============================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from python.org
    echo.
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Check if dependencies are installed
python -c "import fastapi" >nul 2>&1
if errorlevel 1 (
    echo Dependencies not found. Running setup...
    echo.
    python setup.py
    if errorlevel 1 (
        echo ERROR: Setup failed
        pause
        exit /b 1
    )
    echo.
    echo Setup completed! Starting application...
    echo.
)

REM Start the application
python main.py

REM Deactivate virtual environment on exit
deactivate

pause
