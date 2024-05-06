#!/bin/bash

HANDLER=slurm
ENV=~/anaconda3/envs/ml/bin/python

NODES=1
CPUS=24
MEM='502G'

OUTDIR="/clusterfs/nvme/sayan/AI/FIB-SEM_data"

DATASETS=("jrc_choroid-plexus-2")
RES_GRPS=("s0" "s1" "s2" "s3" "s4" "s5")


for DSET in `seq 1 ${#DATASETS[@]}`
do
    for RG in `seq 1 ${#RES_GRPS[@]}`
    do
        while [ $(squeue -u $USER -h -t pending -r | wc -l) -gt 300 ]
        do
            sleep 10s
        done

        j="${ENV} convert_n5_to_tif.py"
        j="${j} --dataset_name ${DATASETS[$DSET-1]}"
        j="${j} --res_grp ${RES_GRPS[$RG-1]}"
          
	JOB="${DATASETS[$DSET-1]}-${RES_GRPS[$RG-1]}"

	LOGS="${OUTDIR}/${DATASETS[$DSET-1]}/logs"

	mkdir -p $LOGS

        task="/usr/bin/sbatch"
        task="${task} --qos=abc_normal --nice=1111111111"
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
