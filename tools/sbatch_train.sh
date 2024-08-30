#!/bin/bash
#SBATCH --job-name=lhy_job
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A800:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0                       # request all the mem on the node
#SBATCH --output=./work_dirs/outputs/%x-%j.out  # where to write output, %x give job name, %j names job id
#SBATCH --error=./work_dirs/outputs/%x-%j.err   # where to write slurm error

# Printing out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

CONFIG=$1
GPUS=$2

srun torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$GPUS \
    tools/train.py \
    $CONFIG \
    --launcher pytorch ${@:3}
