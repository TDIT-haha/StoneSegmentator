python segment/predict.py \
--weights /root/project/Modules/yolov5/runs/train-minseg/V3/weights/best.pt \
--source /root/project/Datas/minStones/test/images \
--hide-labels --hide-conf --line-thickness 1 \
--conf-thres 0.2 \
--iou-thres 0.6 \
--save-txt \
--name V3/exp
# --iou-thres 0.5    --save-txt