python detect.py \
--weights /root/project/Modules/yolov5/runs/train/exp4/weights/best.pt \
--source /root/project/Datas/Stones/val/images/[4689]61.jpg \
--hide-labels --hide-conf --line-thickness 1 --save-txt \
--conf-thres 0.5 \
--iou-thres 0.45