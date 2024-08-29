#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --gpus-per-node=p100l:4   
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24    # There are 24 CPU cores on P100 Cedar GPU nodes
#SBATCH --mem=21.5G               # Request the full memory of the node
#SBATCH --time=10:00:00
#SBATCH --account=def-someuser
module load python/3.10
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r requirements.txt
