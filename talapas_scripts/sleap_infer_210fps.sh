#!/usr/bin/env bash

#SBATCH --account=smearlab
#SBATCH --partition=gpu
#SBATCH --job-name=sleap_infer
#SBATCH --output=logs/sleap_infer/sleap_infer_210fps_%j.out
#SBATCH --error=logs/sleap_infer/sleap_infer_210fps_%j.err
#SBATCH -t0-24
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-80gb
#SBATCH -c 12
#SBATCH --mem=128G
#SBATCH --ntasks-per-node=1

# Blind PyTorch Lightning to SLURM to prevent distributed NCCL deadlocks
unset SLURM_JOB_ID
unset SLURM_NTASKS
unset SLURM_LOCALID

# Prevent OpenMP thread contention deadlocks
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

modelDir="/home/rza/Research/smearlab/scripts/sleap/labels_v007/models"
vidRootDir="/home/rza/Research/smearlab/clickbait-loco/thermister"
modelName="260420_163103.single_instance.n=133"

# Run inference
# Replace with the actual path to your new video file 
# for vidDir in;
#     do for vid in $(ls ${vidRootDir}/${vidDir} | grep .avi); 
#         do /usr/bin/time -v sleap-nn track -i ${vidRootDir}/${vidDir}/${vid} -m ${modelDir}/${modelName}  -o ${vidRootDir}/${vidDir}/${vid}.slp ;
#     done;
# done;

mkdir -p ${modelDir}/${modelName}/preds

for vid in $(find $vidRootDir -type f | grep "[0-9]_raw.avi"); do 
    vidname=$(basename $vid)
    [ -s ${modelDir}/${modelName}/preds/${vidname/.avi/.slp} ] || /usr/bin/time -v sleap-nn track -i ${vid} -m ${modelDir}/${modelName} -o ${modelDir}/${modelName}/preds/${vidname/.avi/.slp}
    rm $vid;
done;





