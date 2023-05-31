import os
import glob
import numpy as np
import json
import cv2

from labelme import utils
from labelme.utils import image


if __name__ == '__main__':
    jsonfolder = r"/root/project/Modules/yolov5/samseg/segment-anything-main/datasets/jsons" #json文件地址
    imagefolder = r"/root/project/Modules/yolov5/samseg/segment-anything-main/datasets/images" #图像文件地址
    savefolder = r"/root/project/Modules/yolov5/samseg/segment-anything-main/datasets/masks" #图像生成的mask地址
    if not os.path.exists(savefolder):
        os.mkdir(savefolder)
    
    for pathi in os.listdir(jsonfolder):
        imgPath = os.path.join(imagefolder, pathi.replace(".json",".jpg"))
        basename_ = os.path.basename(imgPath)
        labelPath = os.path.join(jsonfolder, pathi)
        
        img = cv2.imread(imgPath)
        data = json.load(open(labelPath))  # 加载json文件

        newshapes = []
        for shapei in data['shapes']:
            if len(shapei["points"]) <=2:
                continue
            newshapes.append(shapei)
        
        lbl, lbl_names = utils.labelme_shapes_to_label(img.shape, newshapes)
        mask = []
        class_id = []
        for i in range(1, len(lbl_names)):  # 跳过第一个class（因为0默认为背景,跳过不取！）
            mask.append((lbl == i).astype(np.uint8))  # 举例：当解析出像素值为1，此时对应第一个mask 为0、1组成的（0为背景，1为对象）
            class_id.append(i)  # mask与class_id 对应记录保存

        mask = np.asarray(mask).squeeze().astype(np.uint8)
        w, h = mask.shape[1:]
        new_mask = np.zeros((w, h, 3)).astype(np.uint8)
        
        for mask_ in mask:
            color = np.random.randint(0,255,size=(3))
            new_mask[:,:,0][mask_==1]=color[0]
            new_mask[:,:,1][mask_==1]=color[1]
            new_mask[:,:,2][mask_==1]=color[2]
            
        alpha = 0.4
        beta = 0.8
        draw_im = cv2.addWeighted(new_mask, alpha, img, beta, 0)
        cv2.imwrite(os.path.join(savefolder, basename_), draw_im)
            
    

