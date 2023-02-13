#!/usr/bin/env bash
#sh scripts/tmp_dist_train_gen.sh 0,1,2,3 4 --batch_size=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export NUM_GPUS=6
export OMP_NUM_THREADS=24
export EXPR_NAME=GenLaneNet
export NUM_EPOCHS=40

#while true
#do
#    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
#    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
#    if [ "${status}" != "0" ]; then
#        break;
#    fi
#done
#echo $PORT
#PORT=25750
# --master_port=$PORT
python -m torch.distributed.launch --nproc_per_node ${NUM_GPUS} main_genlanenet.py \
      --mod=${EXPR_NAME} \
      --nepochs=${NUM_EPOCHS} \
      --batch_size=2