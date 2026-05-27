#!/usr/bin/env bash

#SBATCH --account=smearlab
#SBATCH --partition=compute
#SBATCH --job-name=rclone_for_inference
#SBATCH --output=logs/rclone/rclone_sleap_infer_%j.out
#SBATCH --error=logs/rclone/rclone_sleap_infer_%j.err
#SBATCH -t0-24
#SBATCH -c 2
#SBATCH --mem=2G
#SBATCH --ntasks-per-node=1

includeStr=$1

echo $includeStr

vidRootDir="/home/rza/Research/smearlab/clickbait-loco/thermister"

if [[ $includeStr =~ "raw" ]]; then
  modelDir="/home/rza/Research/smearlab/scripts/sleap/labels_v007/models"
  modelName="260420_163103.single_instance.n=133"
  tmpfile="tmp_raw.txt"
else
  modelDir="/home/rza/Research/smearlab/scripts/sleap/labels_v003_0.5scaling/models"
  modelName="260419_223138.0.5_scaling.single_instance.n=193"
  tmpfile="tmp.txt"
fi


module load rclone/1.52.2

for vid in $(rclone ls --include=${includeStr} rzadbx:thermister | awk '{print $2}'); do 
    vidname=$(basename $vid)
    [ -s ${modelDir}/${modelName}/preds/${vidname/.avi/.slp} ] || echo $vid 
done > $tmpfile

rclone copy --files-from=${tmpfile} -v rzadbx:thermister $vidRootDir
# rm tmp.txt

if [[ $includeStr =~ "raw" ]]; then
  sbatch sleap_infer_210fps.sh
else
  sbatch sleap_infer_30fps.sh
fi