python bigImgSeg_slide.py \
--weights "/root/project/Modules/yolov5/runs/train-minseg-0623/exp/exp/weights/best.onnx" \
--imagefolder "/root/project/Modules/yolov5/example/images" \
--savefolder "./draws" \
--conf-thres 0.1 \
--iou-thres 0.45 \
--scale-factor 1.0 \