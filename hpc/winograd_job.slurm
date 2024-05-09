#!/bin/bash

# Please adjust these settings according to your needs.
#SBATCH --account=ece_gy_7123-2024sp
#SBATCH --partition=n1s8-v100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --mem=25GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=winograd
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pa2497@nyu.edu
#SBATCH --output="winograd.out"


module purge
cd /home/pa2497/Thai-Winograd
OVERLAY_FILE=/scratch/pa2497/overlay-25GB-500K.ext3:rw
SINGULARITY_IMAGE=/scratch/pa2497/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif
singularity exec --nv \
	    --overlay $OVERLAY_FILE $SINGULARITY_IMAGE \
	    /bin/bash -c "source /ext3/env.sh; bash hpc/run_evaluation.sh"
