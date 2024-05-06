#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/global/home/users/sayanseal/anaconda3/envs/ml/lib/
#export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024

HANDLER=slurm
ENV=~/anaconda3/envs/ml-new/bin/python

NODES=1
CPUS=128
MEM='2000G'
#MEM='12G'

OUTDIR="/clusterfs/nvme/sayan/AI/training_128/"
mkdir -p $OUTDIR

TRAIN_SET_NUM="mito_training_set_1000"
DATASET="jrc_choroid-plexus-2"
RES_GRP="s2"
MODEL_NAME="DRU128_1000_MITO"
MODEL_NUM=1

while [ $(squeue -u $USER -h -t pending -r | wc -l) -gt 300 ]
do
    sleep 10s
done

j="${ENV} train_multi_gpu.py"
#j="${j} --set_num ${TRAIN_SET_NUM}"
#j="${j} --dataset_name ${DATASET}"
#j="${j} --res_grp ${RES_GRP}"
             
JOB="${TRAIN_SET_NUM}-${DATASET}-${RES_GRP}-${MODEL_NAME}-${MODEL_NUM}"

LOGS="${OUTDIR}/${TRAIN_SET_NUM}/${DATASET}/${RES_GRP}/custom_model_logs"

mkdir -p $LOGS

task="/usr/bin/sbatch"
task="${task} --qos=abc_high --nice=1111111111"
task="${task} --gres=gpu:8"
#task="${task} --partition=abc_a100"
task="${task} --partition=dgx"
#task="${task} --partition=abc"
#task="${task} --constraint='titan'"
task="${task} --nodes=${NODES}"
task="${task} --cpus-per-task=${CPUS}"
task="${task} --mem='${MEM}'"
#task="${task} --mem-per-cpu='${MEM}'"
task="${task} --job-name=${JOB}"
task="${task} --output=${LOGS}/${JOB}.log"
task="${task} --export=ALL"
task="${task} --wrap=\"${j}\""
#task="${task} /bin/bash"
echo $task | bash
echo "ABC : R[$(squeue -u $USER -h -t running -r -p abc | wc -l)], P[$(squeue -u $USER -h -t pending -r -p abc | wc -l)]"
