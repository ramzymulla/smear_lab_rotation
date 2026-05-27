#!/usr/bin/env bash

#SBATCH --account=smearlab
#SBATCH --partition=gpu
#SBATCH --job-name=sleap_train
#SBATCH --output=logs/sleap_train/sleap_train_%j.out
#SBATCH --error=logs/sleap_train/sleap_train_%j.err
#SBATCH -t0-24
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-80gb
#SBATCH -c 8
#SBATCH --mem=128G


/usr/bin/time -v sleap-nn train --config-name single_instance.yaml \
--config-dir . \
trainer_config.ckpt_dir=models \
trainer_config.run_name='"260419_223051.single_instance.n=193"'