#!/bin/bash
#SBATCH --job-name=prep_moses
#SBATCH --partition=catfish
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=prep_moses_%j.out
#SBATCH --error=prep_moses_%j.err

# ============================================================
# ASSUMPTIONS:
# 1. molenv virtualenv exists and has: rdkit, moses, torch,
#    torch_geometric, sklearn, pandas, numpy, tqdm
# 2. Internet access from compute node (Step 1 downloads ~40MB
#    MOSES data via curl if not already present)
# 3. ~5GB free disk space for:
#    - datasets/MOSES.npy + datasets/MOSES_stats.npy (~2GB)
#    - grassy_dit/data/moses_full/ with train/val/test (~3GB)
# 4. GPU needed for Step 2 (scattering extraction)
# 5. Full MOSES = ~1.6M molecules. Step 1 (property computation)
#    will take the longest — possibly 1-3 hours.
# ============================================================

set -e

source molenv/bin/activate
cd /sci/labs/orzuk/shaulytolk/GRASSY-Net
export XDG_CACHE_HOME=/tmp/cache_$SLURM_JOB_ID
export PIP_NO_CACHE_DIR=1

echo "=== Step 1/3: Prepare MOSES dataset (compute properties) ==="
echo "This processes ~1.6M molecules with RDKit. May take 1-3 hours."
python datasets/prepare_moses.py --output-dir datasets
# Outputs: datasets/MOSES.npy, datasets/MOSES_stats.npy

echo ""
echo "=== Step 2/3: Extract scattering moments (GPU) ==="
echo "Fixed GraphScatteringTransform, J=4, moments=4."
python grassy_dit/extract_scattering_fixed.py \
    --dataset datasets/MOSES.npy \
    --stats datasets/MOSES_stats.npy \
    --output grassy_dit/data/moses_full \
    --J 4 --moments 4 --batch_size 128
# Outputs: grassy_dit/data/moses_full/molecules.csv, scattering_moments.npy

echo ""
echo "=== Step 3/3: Split into train/val/test (80/10/10) ==="
python -m grassy_dit.split_datasets \
    --data_dir grassy_dit/data/moses_full \
    --split 0.8 0.1 0.1 \
    --seed 42
# Outputs: grassy_dit/data/moses_full/{train,val,test}/molecules.csv + scattering_moments.npy

echo ""
echo "=== Done! ==="
echo "Data ready at: grassy_dit/data/moses_full/{train,val,test}/"
echo "Next: create training config with max_node=50, data_dir=grassy_dit/data/moses_full/train"
