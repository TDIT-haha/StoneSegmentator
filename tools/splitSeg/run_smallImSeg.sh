python smallImgSeg.py \
--weights "/root/project/Modules/yolov5/runs/train-minseg/V3/weights/best.onnx" \ 
--imagepath r"/root/project/Modules/yolov5/data/images/[1.8]11.jpg" \
--savefolder r"./draws" \
--conf-thres 0.3 \
--iou-thres 0.45 \
--scale-factor 1.0 \