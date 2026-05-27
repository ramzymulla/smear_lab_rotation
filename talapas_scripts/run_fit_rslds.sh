#!/usr/bin/env bash

#SBATCH --account=smearlab
#SBATCH --partition=gpu
#SBATCH --job-name=run_kpms
#SBATCH --output=logs/kpms_fit_model/kpms_fit_rslds_%j.out
#SBATCH --error=logs/kpms_fit_model/kpms_fit_rslds_%j.err
#SBATCH -t0-24
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-80gb
#SBATCH -c 12
#SBATCH --mem=128G
#SBATCH --ntasks-per-node=1

framerate=$1
predsDir=${2}/preds
includeStr=$3
vidRootDir=$4


XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
conda activate kpms_rslds

### (for some reason actively loading cuda breaks jax/torch integration...I guess jax can handle module loading on its own)
# module load cuda/12       
### 

/usr/bin/time -v python /home/rza/Research/smearlab/scripts/smear_lab_rotation/kpms_skip_noise_calibration.py $framerate $predsDir $includeStr rslds

# for vid in $(cat tmp.txt); do
#     vidname=$(basename $vid)
#     rm ${vidRootDir}/${vidname}
# done

conda deactivate