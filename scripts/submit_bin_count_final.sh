#!/bin/bash
#SBATCH --job-name=bincount-final
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/rkathuria/manifolds/logs/bincount-final-%j.out
#SBATCH --error=/home/rkathuria/manifolds/logs/bincount-final-%j.err

set -euo pipefail
PROJ=/home/rkathuria/manifolds
mkdir -p "$PROJ/logs"
cd "$PROJ"

export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16

"$PROJ/.venv/bin/python" scripts/bin_count_final.py
echo "BINCOUNT_DONE"
