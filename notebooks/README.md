# vLLM 실행을 위한 WSL 환경 설정 및 VSCode 연동 가이드

## 1. 개요

vLLM은 메모리 효율성을 최적화한 Lightweight LLM 프레임워크로, Windows 네이티브 환경에서는 실행할 수 없으며 WSL 환경(Ubuntu)에서 실행해야 합니다. 따라서 VSCode에서 작업을 진행할 때 WSL 환경으로 변경하고, 커널도 WSL 환경의 Python 가상환경을 사용해야 합니다.

이 문서는 vLLM을 사용하기 위해 필요한 WSL 설정 및 VSCode 연동 방법을 정리한 가이드입니다.

## 2. WSL 및 Ubuntu 설정

### 2.1 WSL 설치 및 활성화

PowerShell을 관리자 권한으로 실행 후 WSL 활성화:

```powershell
wsl --install
```

설치 가능한 배포판 확인:

```powershell
wsl --list --online
```

Ubuntu 배포판 설치:

```powershell
wsl --install -d Ubuntu-22.04
```

설치 후 WSL 기본 버전이 2인지 확인:

```powershell
wsl -l -v
```

만약 기본 버전이 1이라면 다음 명령어로 변경:

```powershell
wsl --set-version Ubuntu-22.04 2
```

## 3. VSCode에서 WSL 환경 설정

### 3.1 WSL 환경에서 VSCode 실행

WSL을 실행한 후 VSCode를 열기:

```bash
code .
```

VSCode에서 WSL 환경이 열렸는지 확인합니다.

### 3.2 WSL 환경에서 프로젝트 폴더 열기

기존 Windows 네이티브 경로 `C:\Users\insung\LLM_RAG_PROJ`가 WSL에서 `/mnt/c/Users/insung/LLM_RAG_PROJ`로 매핑되므로, WSL에서 해당 폴더를 열어야 합니다.

```bash
cd /mnt/c/Users/insung/LLM_RAG_PROJ
code .
```

## 4. Jupyter 및 WSL 내 가상환경 설정

### 4.1 WSL에서 가상환경 생성 및 활성화

```bash
python3 -m venv vllm_env
source vllm_env/bin/activate
```

### 4.2 Jupyter 및 ipykernel 설치

```bash
pip install jupyter ipykernel
```

### 4.3 Jupyter에 가상환경 등록

```bash
python -m ipykernel install --user --name=vllm_env --display-name "Python (WSL vllm_env)"
```

### 4.4 Jupyter 서버 실행

```bash
jupyter notebook --no-browser --port=8888
```

이후 터미널에 출력된 `localhost:8888/tree?...` 링크를 복사하여 브라우저에서 실행합니다.

## 5. VSCode에서 Jupyter 커널 변경

1. `.ipynb` 파일을 열고 상단의 커널 선택 버튼 클릭
2. "Select Another Kernel" 선택
3. `Python (WSL vllm_env)` 선택
4. 만약 목록에 없다면 다음 명령어 실행 후 다시 시도:

```bash
jupyter kernelspec list
```

5. `vllm_env`가 보이지 않는다면 가상환경 등록을 다시 시도:

```bash
python -m ipykernel install --user --name=vllm_env --display-name "Python (WSL vllm_env)"
```

## 6. vLLM 실행 환경 구축

### 6.1 vLLM 라이브러리 설치

```bash
pip install vllm transformers
```

### 6.2 vLLM 실행 테스트

다음 Python 코드를 실행하여 정상적으로 로드되는지 확인합니다:

```python
from vllm import LLM
from vllm.sampling_params import SamplingParams
print("vLLM 설치 확인 완료!")
```

## 7. 문제 해결

### ❌ VSCode에서 WSL 환경이 인식되지 않을 경우

WSL 환경에서 실행되고 있는지 확인:

```bash
echo $WSL_DISTRO_NAME
```

정상적으로 실행 중이라면 `Ubuntu-22.04` 등의 출력이 나와야 함.

VSCode에서 WSL 확장 설치 확인 (`Ctrl + Shift + P → Remote-WSL: New Window`)

VSCode에서 Python 인터프리터 수동 선택 (`Ctrl + Shift + P → Python: Select Interpreter → /mnt/c/Users/insung/LLM_RAG_PROJ/vllm_env/bin/python3` 선택)

### ❌ Jupyter Notebook에서 WSL 커널이 검색되지 않을 경우

WSL 환경에서 Jupyter 커널 목록 확인:

```bash
jupyter kernelspec list
```

`vllm_env`가 없다면 다시 등록:

```bash
python -m ipykernel install --user --name=vllm_env --display-name "Python (WSL vllm_env)"
```

VSCode 재시작 후 다시 시도

## ✅ 결론

이 문서를 따라 설정하면 vLLM을 WSL 환경에서 원활하게 실행할 수 있습니다. 만약 동일한 오류가 발생하면 위의 문제 해결 방법을 참고하여 다시 해결할 수 있도록 합니다.

