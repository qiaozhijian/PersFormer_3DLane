#!/usr/bin/env bash
#./scripts/dist_eval_validation.sh 0 1 --batch_size=1
export CUDA_VISIBLE_DEVICES=$1
export NUM_GPUS=$2
export OMP_NUM_THREADS=24
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

export EXPR_NAME=PersFormer_validation
echo $EXPR_NAME
python -m torch.distributed.launch --master_port=$PORT --nproc_per_node ${NUM_GPUS} main_persformer.py \
      --mod=${EXPR_NAME} \
      --evaluate \
      --resume="data_splits/trained_models/model_best_PersFormer_on_OpenLaneV1.1.pth.tar" \
      ${PY_ARGS}
