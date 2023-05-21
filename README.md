# StoneSegmentator
Instance segmentation task for stones

## 环境安装
```
conda create -n env_name python=3.8
conda activate env_name
pip install -r requirements.txt
```

## 模型指标
| Model   |   size    |  (box)mAP50  | (box)mAP50-95  |  (mask)mAP50  | (mask)mAP50-95   |
| ------- | ----------|--------------|----------------| ------------- |----------------- |
| v1.0    |  640      |   0.962      |    0.868       |     0.965     |      0.838       |

百度网盘链接：https://pan.baidu.com/s/1r64zoGveZ1zOpAICdX4IpQ  提取码：h5h1 

## 速度
| Model   |  size | GPU/A4000/ONNX/BigImage     |
| ------- | ------|---------------------------  |
| v1.0    |  640  |     8.5s                    | 
               
## 项目流程
1. 数据获取以及处理<br>
    1.1 将6000x4000的图像进行切割为512x512的图像images01<br>
    1.2 结合SAM对images01进行自动标注获得mask图，并通过mash图获得图像的分割标签labels01<br>
    (可集合example/images文件夹内的图像，结合下面的指令进行试跑)

2. 模型训练<br>
    2.1 结合yolov5框架进行模型的训练，使用上面生成分割标签labels01对模型进行训练可获得分割模型pth，训练细节可查询操作指示，分别有训练，验证，可视化<br>

3. 在项目中的使用 <br>
    3.1 为了用于模型的部署，对分割模型pth转换为ONNX，基于ONNX框架进行模型的部署<br>
    3.2 对根据项目的需求对6000x4000图像结合模型进行处理，其逻辑基本流程: 大图像分割为小图像 -> 小图像的模型推理 -> 获得小图像的mask -> 小图像的mask拼接回大图像或进行mask统计 -> 获得石头的大小分布<br>


## 数据获取以及处理
#### 已有的数据集
百度网盘链接：https://pan.baidu.com/s/1QF6ydVZBmLIcLQ6Xfv5IrA  提取码：wspt 

#### 若想创建自己的数据集，需结合sam创建数据集
案例图像A_4542.JPG 百度网盘链接：https://pan.baidu.com/s/1R88gtAb_Cea3udIfsPetzg  提取码：ni4z <br>
将图像A_4542.JPG放置example/images，可以运行下面例子进行试跑<br>


将6000x4000的图像进行切割为512x512的图像集
```
cd tools/splitSeg
python splitImg.py  #切割数据,需自己修改图像地址和保存地址savepath01
```

SAM的vit_h.pt模型，百度网盘链接：https://pan.baidu.com/s/1qZIow1FZ5uQ28SVtRGbY2A 提取码：z86u
```
cd samseg/segment-anything-main
sh run.sh   #使用sam获得每个石头的masks
```

使用SAM的masks转换为json文件用于labelme工具进行查看或是修改，可以再labelme中自己查看
```
python sam_mask2json.py  #生成json文件，可用于labelme的可视化或进行修改
cd ./tools
python labelme2mask.py  #可视化json文件的图像进行保存
```

对json文件转换为yolov5可以训练的数据集
```
cd /root/project/Modules/yolov5/tools
python json2txt_seg.py  #生成可用于训练的数据集
```

## 模型训练
以下的训练可参考yolov5框架的训练修改下面的脚本参数
#### 训练
```
sh run_seg_train.sh
```

#### 验证
```
sh run_seg_val.sh
```

#### 可视化
```
sh run_seg_detect.sh
```

#### 模型转换
模型pth转onnx
```
sh run_export.sh
```

## 在项目中的使用
通过ONNX模型进行对大图像（6000x4000）的处理
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

## 工具
```
cd tools/splitSeg
python splitImg.py  #切割数据
python labelme2mask.py #labelme的可视化
```


## TODO
1、修改分支名字 ok  
2、上传onnx模型、torch模型 ok  
3、整合SAM代码，整理生成数据流程，整理运行脚本到markdown中  ok  
4、整合运行onnx模型的demo代码，整理运行脚本到markdown中 ok  
5、切分大图为小图的代码 ok
