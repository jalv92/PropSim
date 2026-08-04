@echo off
rem Start PropSim from source on Windows. Double-click it.
rem The port is NOT fixed: dashboard.py takes the first free port at or above
rem 8765, so --open is what tells you where it actually landed. Reading the
rem printed URL beats assuming 8765 -- on a machine already running something
rem there (a NinjaTrader add-on, say) you would be looking at the wrong server.
rem NOTHING HERE CHANGES DIRECTORY, on purpose. `cd /d "%~dp0"` fails silently on
rem a UNC path -- a repo living in WSL and opened as \\wsl.localhost\... -- cmd
rem falls back to C:\Windows, and python then reports it cannot open
rem C:\Windows\dashboard.py. PropSim resolves its own files from __file__ and
rem from your home folder (prop_rules._res, ~/.prop-sim), so the working
rem directory is not information it needs: passing the script's absolute path is
rem both shorter and correct on every path shape.
setlocal

set "PY="
for %%P in ("py -3" "python" "python3") do (
    if not defined PY (
        %%~P -c "import sys;raise SystemExit(0 if sys.version_info>=(3,9) else 1)" >nul 2>&1 && set "PY=%%~P"
    )
)
if not defined PY (
    echo.
    echo   Python 3.9+ not found.
    echo   Install it from https://www.python.org/downloads/ and TICK
    echo   "Add python.exe to PATH" in the first screen of the installer.
    echo.
    goto :fail
)

%PY% -c "import numpy" >nul 2>&1
if errorlevel 1 (
    echo Installing numpy ^(one time^)...
    %PY% -m pip install numpy || goto :fail
)

rem %* passes your own flags through, e.g. PropSim.bat --port 9000
%PY% "%~dp0dashboard.py" --open %*
if errorlevel 1 goto :fail
exit /b 0

:fail
echo.
echo   PropSim did not start. The error is above this line.
pause
exit /b 1
