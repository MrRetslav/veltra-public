@echo off
rem Starts the local Veltra server for a model and opens the chat page.
rem usage: chat\start_chat.cmd [model.vlt or legacy .lbt] [port]
set MODEL=%~1
if "%MODEL%"=="" set MODEL=%~dp0..\build\qwen3-1.7b-500steps.lbt
set PORT=%~2
if "%PORT%"=="" set PORT=8099
set EXE=%~dp0..\build\veltra-server.exe
if not exist "%EXE%" set EXE=%~dp0..\build\lb_server.exe
start "Veltra server" "%EXE%" --model "%MODEL%" --port %PORT% --web-dir "%~dp0."
timeout /t 4 >nul
start "" "%~dp0chat.html"
