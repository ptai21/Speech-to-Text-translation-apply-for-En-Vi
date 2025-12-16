#!/bin/bash
#SBATCH -p gpu                          # Partition (queue) for GPUs  
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --cpus-per-gpu=8                # Allocate 8 CPU cores per GPU
#SBATCH --mem-per-cpu=8G                # Memory per CPU (64GB total)
#SBATCH -C "gmem48"                     # Request a specific GPU node
#SBATCH --job-name="llast_train"        # Job name
#SBATCH --gres-flags=enforce-binding    # Enforce resource binding
#SBATCH --output=outputs/train/llast_en_vi_%j.txt  # Output log with job ID
#SBATCH --exclude=c2-1                  # Exclude node c2-1
#SBATCH --time=30:00:00                 # Time limit (adjust as needed)

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi
echo "Job started at: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

### ==========================
### 1. LOAD CONDA ENV
### ==========================
source ~/miniconda/etc/profile.d/conda.sh
conda activate asr_env

### ==========================
### 2. GIẢM THREAD ĐỂ KHÔNG BỊ LỖI OPENBLAS
### ==========================
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

### ==========================
### 3. SET CUDA PATHS
### ==========================
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

### ==========================
### 4. PATH DỰ ÁN
### ==========================
XTUNER_DIR="/home/da530038/tai_speech/xtuner"
CONFIG="/home/da530038/tai_speech/data/configs/llast_en_vi.py"
WORK_DIR="/home/da530038/tai_speech/work_dirs/llast_en_vi"

cd $XTUNER_DIR
export PYTHONPATH=$XTUNER_DIR:$PYTHONPATH

# Create output directory for logs
mkdir -p outputs/train

# Create work directory
mkdir -p "$WORK_DIR"

### ==========================
### 5. START TRAINING
### ==========================
echo "=== XTuner Training ==="
echo "Using config: $CONFIG"
echo "Work dir: $WORK_DIR"
echo "Python env: $(which python)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_HOME: $CUDA_HOME"
echo "Starting training at: $(date)"

python -m xtuner.tools.train \
    "$CONFIG" \
    --work-dir "$WORK_DIR" \
    --launcher none

echo "Training finished at: $(date)"