#!/bin/bash

HANDLER=slurm
ENV=~/anaconda3/envs/ml/bin/python

NODES=1
CPUS=1
MEM='20G'

OUTDIR="/clusterfs/nvme/sayan/AI/training_128/mito_training_set_1000"
#OUTDIR="/clusterfs/nvme/sayan/AI/testing_million/testing_set_1"
mkdir -p $OUTDIR

DATASET="jrc_choroid-plexus-2"
RES_GRP="s2"
#LABELS=("endo" "er" "mito" "pm" "vesicle" "endo-er" "endo-mito" "endo-pm" "endo-vesicle" "er-mito" "er-pm" "er-vesicle" "mito-pm" "mito-vesicle" "pm-vesicle")
LABELS=("mito")
#FILE_IDX=($(seq 1 1 147))
#FILE_IDX=($(seq 16 1 120))
FILE_IDX=($(seq 25 1 28))
#FILE_IDX=(50)
SEED=20001
SAMPLE_PATCH_SIZE=128
OCC_THRESH=0.01
VAR_INT=1
#PHOTONS=(400 500 600 700 800 900 1000)
PHOTONS=(1000)
NUM_ROT=6
ROT_MODE="same"
NUM_PSF_PER_BIN=10
BG_OFFSET_MEAN=100
BG_NOISE_SIGMA=40


for LABEL in `seq 1 ${#LABELS[@]}`
do
  for PH in `seq 1 ${#PHOTONS[@]}`
  do
    for FID in `seq 1 ${#FILE_IDX[@]}`
    do
      for NR in `seq 1 ${NUM_ROT}`
      do
        while [ $(squeue -u $USER -h -t pending -r | wc -l) -gt 300 ]
        do
            sleep 10s
        done

        j="${ENV} dataset_creation_train.py"
        j="${j} --dataset_name ${DATASET}"
        j="${j} --res_grp ${RES_GRP}"
        j="${j} --label ${LABELS[$LABEL-1]}"
        j="${j} --file_idx ${FILE_IDX[$FID-1]}"
        j="${j} --seed ${SEED}"
        j="${j} --sample_patch_size ${SAMPLE_PATCH_SIZE}"
        j="${j} --occupancy_thresh ${OCC_THRESH}"
        j="${j} --vary_intensity ${VAR_INT}"
        j="${j} --ph ${PHOTONS[$PH-1]}"
#        j="${j} --num_rotations ${NUM_ROT}"
        j="${j} --rot_num ${NR}"
        j="${j} --rotation_mode ${ROT_MODE}"
        j="${j} --num_psfs_per_bin ${NUM_PSF_PER_BIN}"
        j="${j} --mean_background_offset ${BG_OFFSET_MEAN}"
        j="${j} --sigma_background_noise ${BG_NOISE_SIGMA}"

        JOB="${DATASET}-${RES_GRP}-${LABELS[$LABEL-1]}-${FILE_IDX[$FID-1]}-${NR}-${SEED}-${PHOTONS[$PH-1]}-${SAMPLE_PATCH_SIZE}"

        LOGS="${OUTDIR}/${DATASET}/${RES_GRP}/logs"

        mkdir -p $LOGS

        task="/usr/bin/sbatch"
        task="${task} --qos=abc_normal --nice=1111111111"
        task="${task} --partition=abc"
    #    task="${task} --partition=dgx"
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
    done
  done
done
