@echo off
cd /d "%~dp0"
echo Starting Goku Instant Transmission Virtual Camera...
uv run python main.py %*
pause
