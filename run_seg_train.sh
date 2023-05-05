GPUS=1
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-28523}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

python -m torch.distributed.run \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    segment/train.py \
    --data ./data/stone-seg-min.yaml \
    --weights /root/project/Modules/yolov5/pretrains/yolov5x-seg.pt \
    --cfg ./models/segment/yolov5x-seg.yaml \
    --hyp ./data/hyps/hyp.scratch-med.yaml \
    --batch-size 2 \
    --epochs 600 \
    --img 640 \
    --device 0 \
    --project runs/train-minseg \
    --cache