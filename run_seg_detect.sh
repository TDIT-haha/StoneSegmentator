python segment/predict.py \
--weights /root/project/Modules/yolov5/runs/train-minseg/exp7/weights/best.pt \
--source /root/project/Modules/yolov5/A_4542.JPG \
--hide-labels --hide-conf --line-thickness 1  --save-txt \
--conf-thres 0.5