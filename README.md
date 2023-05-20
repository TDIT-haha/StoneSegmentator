# StoneSegmentator
Instance segmentation task for stones

## checkpoints
百度网盘链接：https://pan.baidu.com/s/1r64zoGveZ1zOpAICdX4IpQ  提取码：h5h1 

# speed
| Model   |   GPU/A4000/ONNX/BigImage     |
| ------- | ---------------------------   |
| v1.0    |    8.5s                       | 
|         |                               |                   


## datasets
结合sam创建数据集
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
```
cd tools/splitSeg
python bigImgSeg.py
```



## TODO
1、修改分支名字 ok  
2、上传onnx模型、torch模型 ok  
3、整合SAM代码，整理生成数据流程，整理运行脚本到markdown中  ok  
4、整合运行onnx模型的demo代码，整理运行脚本到markdown中 ok  
5、切分大图为小图的代码   
