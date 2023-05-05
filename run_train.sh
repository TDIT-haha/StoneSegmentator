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
    train.py \
    --batch 16 \
    --imgsz 640 \
    --data ./data/stone.yaml \
    --weights ./pretrains/yolov5x.pt \
    --cfg ./models/yolov5x.yaml \
    --hyp ./data/hyps/hyp.scratch-med.yaml \
    --device 0 \
    --epochs 600


# python train.py \
#     --batch 16 \
#     --imgsz 640 \
#     --data ./data/stone.yaml \
#     --weights ./pretrains/yolov5x.pt \
#     --cfg ./models/yolov5x.yaml \
#     --hyp ./data/hyps/hyp.scratch-med.yaml \
#     --device 0 \
#     --epochs 300