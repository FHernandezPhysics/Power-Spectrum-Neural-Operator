#!/usr/bin/bash
#SBATCH -p medium-40core
#SBATCH -o %u-%x-%A.txt

micromamba activate "$SCRATCH/cosmo"
cd "$HOME/cosmology"

python lno.py
exit