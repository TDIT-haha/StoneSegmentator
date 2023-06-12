python smallImgSeg.py \
--weights "/root/project/Modules/yolov5/runs/train-minseg/V3/weights/best.onnx" \
--imagepath "/root/project/Modules/yolov5/data/images/[1.8]11.jpg" \
--savefolder "./draws" \
--conf-thres 0.3 \
--iou-thres 0.45 \
--scale-factor 1.0 \

