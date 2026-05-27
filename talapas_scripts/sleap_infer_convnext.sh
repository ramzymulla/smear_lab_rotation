#!/usr/bin/env bash

#SBATCH --account=smearlab
#SBATCH --partition=gpu
#SBATCH --job-name=sleap_infer
#SBATCH --output=logs/sleap_infer_%j.out
#SBATCH --error=logs/sleap_infer_%j.err
#SBATCH -t0-24
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-80gb
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH --ntasks-per-node=1

# Blind PyTorch Lightning to SLURM to prevent distributed NCCL deadlocks
unset SLURM_JOB_ID
unset SLURM_NTASKS
unset SLURM_LOCALID

# Prevent OpenMP thread contention deadlocks
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

modelDir="/home/rza/Research/smearlab/clickbait-motivate/bonsai/7010/models"
vidDir="/home/rza/Research/smearlab/clickbait-motivate/bonsai/7010/raw_videos"

# Run inference
# Replace with the actual path to your new video file 
for VIDEO_FILE in $(ls /home/rza/Research/smearlab/clickbait-motivate/bonsai/7010/raw_videos | grep -v slp); 
    do /usr/bin/time -v sleap-nn track -i ${vidDir}/${VIDEO_FILE} -m ${modelDir}/260402_104654.single_instance.n=74 -o ${vidDir}/convnext_${VIDEO_FILE}.slp ;
done