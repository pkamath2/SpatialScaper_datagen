#!/bin/bash -l

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=spatial_scaper

# Define, how many nodes you need. Here, we ask for 1 node.
# Each node has 16 or 20 CPU cores.
#SBATCH --cpus-per-task=12
#SBATCH --mem=32GB
#SBATCH --time=48:00:00
#SBATCH --account=pr_259_general

cd /scratch/pk3251/appdir/Github/SpatialScaper/

./sing << EOF
conda init
conda activate spatial_scaper
conda env list
python example_generation.py

EOF
