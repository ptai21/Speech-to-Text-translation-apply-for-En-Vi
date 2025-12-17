#!/bin/bash

### ==========================
### 1. LOAD CONDA ENV
### ==========================
source ../miniconda/etc/profile.d/conda.sh
conda activate asr_env     # <<< bạn đang dùng môi trường này

### ==========================
### 2. GIẢM THREAD ĐỂ KHÔNG BỊ LỖI OPENBLAS
### ==========================
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

### ==========================
### 3. CHỌN GPU
### ==========================
export CUDA_VISIBLE_DEVICES=0   
# Set CUDA_HOME 
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
### ==========================
### 4. PATH DỰ ÁN
### ==========================
XTUNER_DIR="../tai_speech/xtuner"
CONFIG="../tai_speech/data/configs/llast_en_vi.py"
WORK_DIR="../tai_speech/work_dirs/llast_en_vi"

cd $XTUNER_DIR
export PYTHONPATH=$XTUNER_DIR:$PYTHONPATH

### ==========================
### 5. START TRAINING
### ==========================
echo "=== XTuner Training ==="
echo "Using config: $CONFIG"
echo "Work dir: $WORK_DIR"
echo "Python env: $(which python)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python -m xtuner.tools.train \
    "$CONFIG" \
    --work-dir "$WORK_DIR" \
    --launcher none
