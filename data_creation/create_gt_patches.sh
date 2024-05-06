#!/bin/bash

HANDLER=slurm
ENV=~/anaconda3/envs/ml/bin/python

NODES=1
CPUS=12
MEM='100G'

OUTDIR="/clusterfs/nvme/sayan/AI/FIB-SEM_data_patches_375/"

DATASET="jrc_choroid-plexus-2"
RES_GRP="s2"
LABELS=("endo" "er" "mito" "pm" "vesicle")
#LABELS=("mito")
SEED=20001
AUG_PATCH_SIZE=375
STEP_SIZE=100
PHOTONS=(400 500 600 700 800 900 1000)
#PHOTONS=(1000)
VAR_INT=(1)


for LABEL in `seq 1 ${#LABELS[@]}`
do
    for PH in `seq 1 ${#PHOTONS[@]}`
    do
        for VI in `seq 1 ${#VAR_INT[@]}`  
	      do
    	    while [ $(squeue -u $USER -h -t pending -r | wc -l) -gt 300 ]
    	    do
                sleep 10s
    	    done

          j="${ENV} create_gt_patches.py"
    	    j="${j} --dataset_name ${DATASET}"
    	    j="${j} --res_grp ${RES_GRP}"
    	    j="${j} --label ${LABELS[$LABEL-1]}"
    	    j="${j} --seed ${SEED}"
	        j="${j} --photons ${PHOTONS[$PH-1]}"
    	    j="${j} --aug_patch_size ${AUG_PATCH_SIZE}"
    	    j="${j} --step_size ${STEP_SIZE}"
	        j="${j} --vary_intensity ${VAR_INT[$VI-1]}"
         
    	    JOB="${DATASET}-${RES_GRP}-${LABELS[$LABEL-1]}-${SEED}-${PHOTONS[$PH-1]}-${AUG_PATCH_SIZE}-${STEP_SIZE}-${VAR_INT[$VI-1]}"

    	    LOGS="${OUTDIR}/${DATASET}/logs"

    	    mkdir -p $LOGS

    	    task="/usr/bin/sbatch"
    	    task="${task} --qos=abc_high --nice=1111111111"
    	    task="${task} --partition=abc"
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
