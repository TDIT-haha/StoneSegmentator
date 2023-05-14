python segment/predict.py \
--weights /root/project/Modules/yolov5/runs/train-minseg/V3/weights/best.pt \
--source /root/project/Modules/yolov5/data/images/[1.8]11.jpg \
--hide-labels --hide-conf --line-thickness 1 \
--conf-thres 0.2 \
--name V3/exp
# --iou-thres 0.5    --save-txt