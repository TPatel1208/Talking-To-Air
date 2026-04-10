@echo off
echo Starting Talking to Air...

set ROOT=%~dp0

:: Find conda in common locations
if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    set CONDA_ACTIVATE=%USERPROFILE%\anaconda3\Scripts\activate.bat
) else if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    set CONDA_ACTIVATE=%USERPROFILE%\miniconda3\Scripts\activate.bat
) else if exist "C:\ProgramData\anaconda3\Scripts\activate.bat" (
    set CONDA_ACTIVATE=C:\ProgramData\anaconda3\Scripts\activate.bat
) else (
    echo Could not find conda. Please activate geo_ai_env manually.
    pause
    exit
)

start "Backend" cmd /k "cd /d "%ROOT%Backend" && call "%CONDA_ACTIVATE%" geo_ai_env && uvicorn api:app --reload --port 8000"

timeout /t 3 /nobreak >nul

start "Frontend" cmd /k "cd /d "%ROOT%Frontend" && npm run dev"

:: Wait for Vite to start then open browser
timeout /t 4 /nobreak >nul
start "" "http://localhost:5173"

echo Both servers started.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:5173