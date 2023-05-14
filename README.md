# StoneSegmentator

## Data Generation Process

## Fine-tuning
### Run detection
```
# python scripts
python detect.py \
--weights /root/project/Modules/yolov5/runs/train/exp4/weights/best.pt \
--source /root/project/Datas/Stones/val/images/[4689]61.jpg \
--hide-labels --hide-conf --line-thickness 1 --save-txt \
--conf-thres 0.5 \
--iou-thres 0.45

# shell scripts
sh run_detect.sh
```
### Run Seg train
```
# run_detect.sh

```
### Export onnx
```
# run_export.sh
```
