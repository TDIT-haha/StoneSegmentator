# StoneSegmentator
Instance segmentation task for stones

## 项目流程
1. 数据获取以及处理
    1.1 将6000x4000的图像进行切割为512x512的图像images01<br>
    1.2 结合SAM对images01进行自动标注获得mask图，并通过mash图获得图像的分割标签labels01

2. 模型训练
    2.1 结合yolov5框架进行模型的训练，使用上面生成分割标签labels01对模型进行训练可获得分割模型pth，训练细节可查询操作指示，分别有训练，验证，可视化

3. 在项目中的使用 
    3.1 为了用于模型的部署，对分割模型pth转换为ONNX，基于ONNX框架进行模型的部署<br>
    3.2 对根据项目的需求对6000x4000图像结合模型进行处理，其逻辑基本流程: 大图像分割为小图像 -> 小图像的模型推理 -> 获得小图像的mask -> 小图像的mask拼接回大图像或进行mask统计 -> 获得石头的大小分布



## checkpoints
| Model   |   size    |  (box)mAP50  | (box)mAP50-95  |  (mask)mAP50  | (mask)mAP50-95   |
| ------- | ----------|--------------|----------------| ------------- |----------------- |
| v1.0    |  640      |   0.962      |    0.868       |     0.965     |      0.838       |

百度网盘链接：https://pan.baidu.com/s/1r64zoGveZ1zOpAICdX4IpQ  提取码：h5h1 

## speed
| Model   |  size | GPU/A4000/ONNX/BigImage     |
| ------- | ------|---------------------------  |
| v1.0    |  640  |     8.5s                    | 
               

## datasets
### 数据集
百度网盘链接：https://pan.baidu.com/s/1QF6ydVZBmLIcLQ6Xfv5IrA  提取码：wspt 

### 结合sam创建数据集
```
cd samseg/segment-anything-main
sh run.sh   #使用sam获得每个石头的masks
```
使用masks转换为json文件用于labelme工具进行查看或是修改
```
python draw_img_train.py  #生成训练集
python draw_img_val.py  #生成测试机
```

## train
```
sh run_seg_train.sh
```

## val
```
sh run_seg_val.sh
```

## predict
```
sh run_seg_detect.sh
```

## export
```
sh run_export.sh
```

## use onnx to run images
#### 如果运行大图
```
cd tools/splitSeg
python bigImgSeg.py \
--weights "/root/project/Modules/yolov5/runs/train-minseg/V3/weights/best.onnx" \ 
--imagefolder r"./images" \
--savefolder r"./draws" \
--conf-thres 0.3 \
--iou-thres 0.45 \
--scale-factor 1.0 \
```
#### 如果运行分割后的图
```
cd tools/splitSeg
python smallImgSeg.py \
--weights "/root/project/Modules/yolov5/runs/train-minseg/V3/weights/best.onnx" \ 
--imagepath r"/root/project/Modules/yolov5/data/images/[1.8]11.jpg" \
--savefolder r"./draws" \
--conf-thres 0.3 \
--iou-thres 0.45 \
--scale-factor 1.0 \
```

## tools
```
cd tools/splitSeg
python splitImg.py  #切割数据
```


## TODO
1、修改分支名字 ok  
2、上传onnx模型、torch模型 ok  
3、整合SAM代码，整理生成数据流程，整理运行脚本到markdown中  ok  
4、整合运行onnx模型的demo代码，整理运行脚本到markdown中 ok  
5、切分大图为小图的代码 ok
