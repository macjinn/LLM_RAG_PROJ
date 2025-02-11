@echo off
setlocal

:: 가상환경 이름 설정
set VENV_NAME=myenv

:: Python이 설치되어 있는지 확인
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed. Please install Python first.
    exit /b
)

:: 가상환경이 이미 존재하는지 확인
if exist %VENV_NAME% (
    echo Virtual environment '%VENV_NAME%' already exists.
) else (
    echo Creating virtual environment '%VENV_NAME%'...
    python -m venv %VENV_NAME%
)

:: 가상환경 활성화
call %VENV_NAME%\Scripts\activate

:: pip 최신화
python -m pip install --upgrade pip

:: 설치할 패키지 목록
set packages=torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
set packages=%packages% transformers datasets bitsandbytes peft langchain gradio faiss-cpu

:: 패키지 설치 확인 및 설치
for %%p in (%packages%) do (
    echo Checking %%p...
    python -c "import %%p" 2>nul
    if %errorlevel% neq 0 (
        echo Installing %%p...
        pip install %%p
    ) else (
        echo %%p is already installed.
    )
)

:: 가상환경 활성화 메시지
echo Virtual environment '%VENV_NAME%' is ready.
echo To activate it, run: call %VENV_NAME%\Scripts\activate

endlocal
pause
