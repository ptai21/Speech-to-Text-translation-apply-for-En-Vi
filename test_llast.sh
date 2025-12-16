#!/bin/bash

### ==========================
### 1. LOAD CONDA ENV
### ==========================
source /home/da530038/miniconda/etc/profile.d/conda.sh
conda activate asr_env     # dùng cùng môi trường với train

### ==========================
### 2. GIẢM THREAD
### ==========================
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

### ==========================
### 3. CHỌN GPU
### ==========================
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

### ==========================
### 4. PATH DỰ ÁN
### ==========================
XTUNER_DIR="/home/da530038/tai_speech/xtuner"
CONFIG="/home/da530038/tai_speech/data/configs/llast_en_vi.py"
CHECKPOINT="/home/da530038/tai_speech/work_dirs/llast_en_vi/iter_120000.pth"

cd $XTUNER_DIR
export PYTHONPATH=$XTUNER_DIR:$PYTHONPATH

### ==========================
### 5. START TESTING
### ==========================
echo "=== XTuner Testing ==="
echo "Using config: $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "Python env: $(which python)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python -m xtuner.tools.test \
    "$CONFIG" \
    --checkpoint "$CHECKPOINT"
