python bigImgSeg.py \
--weights "/root/project/Modules/yolov5/runs/train-minseg/V3/weights/best.onnx" \
--imagefolder "/root/project/Modules/yolov5/example/images" \
--savefolder "./draws" \
--conf-thres 0.3 \
--iou-thres 0.45 \
--scale-factor 1.0 \