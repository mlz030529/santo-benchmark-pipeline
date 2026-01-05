#!/bin/bash
#SBATCH -J santo_benchmark
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=256G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
mkdir -p logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}


export SANTO_DATA_DIR=/path/to/data
export SANTO_OUT_DIR=/path/to/santo_outputs

export SANTO_REPO=/path/to/SANTO-main

export SANTO_PAIRS="meta_ref.csv,meta_mov.csv,expr_ref.csv,expr_mov.csv"

export SANTO_XCOL=center_x
export SANTO_YCOL=center_y


export SANTO_USE_SUBSAMPLE=0   # 0 = full benchmark
# export SANTO_USE_SUBSAMPLE=1

export SANTO_REF_N=15000
export SANTO_MOV_N=15000
export SANTO_GRID_BINS=100

export SANTO_SEED_REF=0
export SANTO_SEED_MOV=1

python run_santo_pairs.py
