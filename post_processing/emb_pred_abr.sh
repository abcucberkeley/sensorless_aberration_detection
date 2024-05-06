#!/bin/bash

HANDLER=slurm
ENV=~/anaconda3/envs/ml/bin/python

NODES=1
CPUS=64
MEM='1000G'
#MEM='12G'

#OUTDIR="/clusterfs/nvme/sayan/AI/training_million/mito_training_set"
OUTDIR="/clusterfs/nvme/sayan/AI/testing_million/testing_set_1"
mkdir -p $OUTDIR

RT=("/clusterfs/nvme/sayan/AI/testing_million/testing_set_1/jrc_choroid-plexus-2/s2/psf1/roi10/" "/clusterfs/nvme/sayan/AI/testing_million/testing_set_1/jrc_choroid-plexus-2/s2/psf2/roi10/" \
"/clusterfs/nvme/sayan/AI/testing_million/testing_set_1/jrc_choroid-plexus-2/s2/psf3/roi10/" "/clusterfs/nvme/sayan/AI/testing_million/testing_set_1/jrc_choroid-plexus-2/s2/psf4/roi10/" \
"/clusterfs/nvme/sayan/AI/testing_million/testing_set_1/jrc_choroid-plexus-2/s2/psf5/roi10/" "/clusterfs/nvme/sayan/AI/testing_million/testing_set_1/jrc_choroid-plexus-2/s2/psf6/roi10/")

#RT=("/clusterfs/nvme/sayan/AI/testing_million/testing_set_1/jrc_choroid-plexus-2/s2/psf1/roi11/" "/clusterfs/nvme/sayan/AI/testing_million/testing_set_1/jrc_choroid-plexus-2/s2/psf2/roi11/" \
#"/clusterfs/nvme/sayan/AI/testing_million/testing_set_1/jrc_choroid-plexus-2/s2/psf3/roi11/" "/clusterfs/nvme/sayan/AI/testing_million/testing_set_1/jrc_choroid-plexus-2/s2/psf4/roi11/" \
#"/clusterfs/nvme/sayan/AI/testing_million/testing_set_1/jrc_choroid-plexus-2/s2/psf5/roi11/" "/clusterfs/nvme/sayan/AI/testing_million/testing_set_1/jrc_choroid-plexus-2/s2/psf6/roi11/")

M_L="custom_predictions_mito_drunet_4"
#M_L="custom_predictions_er_drunet_1"
#M_L="custom_predictions_exp1_drunet_1"
LAYOUT_RT="/clusterfs/nvme/sayan/AI/testing_million/testing_set_1/jrc_choroid-plexus-2/s2/layouts_4_inf/"
mkdir -p $LAYOUT_RT

for C in `seq 1 ${#RT[@]}`
do
  while [ $(squeue -u $USER -h -t pending -r | wc -l) -gt 300 ]
  do
    sleep 10s
  done

  j="${ENV} emb_pred_abr.py"
  j="${j} --root ${RT[$C-1]}"
  j="${j} --model ${M_L}"
  j="${j} --layout_root ${LAYOUT_RT}"
         
  JOB="${SAVE_FILE[$C-1]}-inf"

  LOGS="${OUTDIR}/${DATASET}/${RES_GRP}/layout_logs"
  mkdir -p $LOGS

  task="/usr/bin/sbatch"
  task="${task} --qos=abc_normal --nice=1111111111"
#  task="${task} --partition=abc"
  task="${task} --gres=gpu:1"
  task="${task} --partition=abc_a100"
#  task="${task} --partition=dgx"
  task="${task} --nodes=${NODES}"
  task="${task} --cpus-per-task=${CPUS}"
  task="${task} --mem='${MEM}'"
  task="${task} --job-name=${JOB}"
  task="${task} --output=${LOGS}/${JOB}.log"
  task="${task} --export=ALL"
  task="${task} --wrap=\"${j}\""
  echo $task | bash
  echo "ABC : R[$(squeue -u $USER -h -t running -r -p abc | wc -l)], P[$(squeue -u $USER -h -t pending -r -p abc | wc -l)]"
done