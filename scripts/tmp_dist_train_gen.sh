#!/usr/bin/env bash
#sh scripts/tmp_dist_train_gen.sh 0,1,2,3 4 --batch_size=8
export CUDA_VISIBLE_DEVICES=$1
export NUM_GPUS=$2
export OMP_NUM_THREADS=24
export EXPR_NAME=GenLaneNet
export NUM_EPOCHS=40
PY_ARGS=${@:3}

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT
python -m torch.distributed.launch --master_port=$PORT --nproc_per_node ${NUM_GPUS} train.py \
      --mod=${EXPR_NAME} \
      --nepochs=${NUM_EPOCHS} \
      ${PY_ARGS}