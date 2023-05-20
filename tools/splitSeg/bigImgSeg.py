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

    savefolder = r"./draws"
    if not os.path.exists(savefolder):
        os.mkdir(savefolder)
        
    set_split_size = (512, 512)
    imagefolder = r"./images"
    imagelist = glob.glob(os.path.join(imagefolder, "*.JPG"))
    
    print("load image .......")
    for imagepath in imagelist:
        # imagepath = r"./images/A_4542.JPG"
        basename_ = os.path.basename(imagepath)
        img = cv2.imread(imagepath)
        h,w = img.shape[:2]

        newimg = cv2.copyMakeBorder(img, 0, 96, 0, 144, cv2.BORDER_CONSTANT, value=(0,0,0))  # add border
        h,w = newimg.shape[:2]
        # print("{} {}".format(w, h))
        
        drawimg = np.zeros((w, h, 3)).astype(np.uint8)
        colsimg = []
        totalAreas = []
        for i in tqdm(range(12)):
            raws_img = []
            for j in tqdm(range(8)):
                cropimg = newimg[j*512:(j+1)*512, i*512:(i+1)*512]
                dets, masks = model.inter(cropimg)
                h_,w_ = cropimg.shape[:2]
                new_mask = np.zeros((w_, h_, 3)).astype(np.uint8)
                for mask_ in masks:
                    mask_[mask_<255]=0
                    contours, hierarchy = cv2.findContours(mask_, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) 
                    areas = []
                    if len(contours) == 0:
                        continue
                    for c in range(len(contours)):
                        if cv2.contourArea(contours[c]) == 0:
                            continue
                        areas.append(cv2.contourArea(contours[c]))
                    if len(areas) == 0:
                        continue
                    max_areas = np.max(areas)
                    # if int(max_areas) == 3:
                    #     cv2.imwrite("mask.jpg", mask_)
                    #     cv2.imwrite("cropimg.jpg", cropimg)
                    #     exit()
                    totalAreas.append(max_areas)
                    
                    color = np.random.randint(0,255,size=(3))
                    new_mask[:,:,0][mask_==255]=color[0]
                    new_mask[:,:,1][mask_==255]=color[1]
                    new_mask[:,:,2][mask_==255]=color[2]
                    
                raws_img.append(new_mask)

            tmp_img = np.concatenate(raws_img)         
            colsimg.append(tmp_img)
                    
        tmp_img = np.concatenate(colsimg, axis=1) 
        # cv2.imwrite(os.path.join(savefolder, basename_), tmp_img)
        cv2.imwrite("draw_{}".format(basename_), tmp_img)
        # cv2.imwrite("draw_{}".format(os.path.basename(imagepath)), drawimg)
        totalAreas_ = [num * scale_factor for num in totalAreas]
        totalAreas.sort()
        print("共有块数：{}".format(len(totalAreas)))
        print("面积分别有:{}".format(totalAreas))
        
        
        # exit(1)
        