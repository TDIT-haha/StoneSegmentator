import os
from tqdm import tqdm
import numpy as np
import onnxruntime
import torch
import glob
from codes.tools import *
from codes.model import StoneSeg

if __name__ == "__main__":
    # 模型的初始化
    modelpath = "/root/project/Modules/yolov5/runs/train-minseg/V3/weights/best.onnx"
    model = StoneSeg(modelpath, className=["stone"], size=640, conf_thres=0.3, iou_thres=0.45)
    model.modelinit()
    model.modelwarmup()

    # 比例尺系数，默认为1.0
    scale_factor = 1.0 

    imagepath = r"/root/project/Modules/yolov5/data/images/[1.8]11.jpg"
    img = cv2.imread(imagepath)
    h,w = img.shape[:2]
    dets, masks = model.inter(img)
    new_mask = np.zeros((w, h, 3)).astype(np.uint8)
    totalAreas = []
    for mask_ in masks:
        # cv2.imwrite("mask.jpg", mask_)
        contours, hierarchy = cv2.findContours(mask_, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 
        areas = []
        for c in range(len(contours)):
            areas.append(cv2.contourArea(contours[0]))
        max_areas = np.max(areas)
        totalAreas.append(max_areas)
            
        color = np.random.randint(0,255,size=(3))
        new_mask[:,:,0][mask_==255]=color[0]
        new_mask[:,:,1][mask_==255]=color[1]
        new_mask[:,:,2][mask_==255]=color[2]
        
    cv2.imwrite("draw_{}".format(os.path.basename(imagepath)), new_mask)
    totalAreas_ = [num * scale_factor for num in totalAreas]
    totalAreas_.sort()
    print("共有块数：{}".format(len(totalAreas_)))
    print("面积分别有:{}".format(totalAreas_))
    