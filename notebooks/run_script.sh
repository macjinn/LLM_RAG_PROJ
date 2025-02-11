#!/bin/bash
#SBATCH -p normal                 # 사용할 파티션 (normal)
#SBATCH -G 1                      # 1개의 GPU 사용
#SBATCH -n 1                      # 1개의 CPU 코어 사용
#SBATCH --job-name=test_finshibainu     # 작업 이름
#SBATCH --output=output_%j.log    # 로그 파일 저장 (%j는 Job ID)

# 1) Conda 환경 활성화
source /opt/sw/anaconda3/etc/profile.d/conda.sh
conda activate torch220_cu118

# 2) 실행할 Python 스크립트 실행
python test_ifnshibainu.py
