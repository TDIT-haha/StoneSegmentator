import os
import numpy as np
import cv2
from tqdm import tqdm
from pycococreatortools import pycococreatortools
import json
from tools import *

def get_approx(img, contour, length_p=0.005):
    """获取逼近多边形
    :param img: 处理图片
    :param contour: 连通域
    :param length_p: 逼近长度百分比
    """
    img_adp = img.copy()
    # 逼近长度计算
    epsilon = length_p * cv2.arcLength(contour, True)
    # 获取逼近多边形
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return approx

# train
# imgfolder = r"/root/project/Datas/Stones/train/images"
# outputfolder = r"/root/project/Modules/segment-anything-main/train_outputs"
# savepath = r"/root/project/Datas/Stones/train/draws"
# datafolder_images = r"/root/project/Datas/Stones/train/images"
# datafolder_jsons = r"/root/project/Datas/Stones/train/jsons"
# # datafolder_labels = r"/root/project/Datas/Stones/train/labels"

# val
imgfolder = r"/root/project/Modules/segment-anything-main/tmps/images"
outputfolder = r"/root/project/Modules/segment-anything-main/test_haha_outputs"
savepath = r"/root/project/Modules/segment-anything-main/tmps/draws"
datafolder_images = r"/root/project/Modules/segment-anything-main/tmps/images"
datafolder_jsons = r"/root/project/Modules/segment-anything-main/tmps/jsons"


if not os.path.exists(datafolder_images):
    os.mkdir(datafolder_images)
if not os.path.exists(datafolder_jsons):
    os.mkdir(datafolder_jsons)
# if not os.path.exists(datafolder_labels):
#     os.mkdir(datafolder_labels)

alpha = 0.2 
beta = 1
for ii , basename_ in enumerate(tqdm(os.listdir(outputfolder))):
    # if ii <1498:
    #     continue
    imgname = "{}.jpg".format(basename_)
    imgpath = os.path.join(imgfolder, imgname)
    outputpath = os.path.join(outputfolder, basename_)
    im = cv2.imread(imgpath)
    imageHeight, imageWidth = im.shape[:2]
    im_copy = im.copy()

    
    cons = []
    annos = {
        "flags":{},
        "version": "4.5.10",
        "imagePath":imgname,
        "shapes":[],
        "imageHeight":imageHeight,
        "imageWidth":imageWidth,
        "imageData": None
    }
    
    for indx, pathi in enumerate(os.listdir(outputpath)):
        if pathi == "metadata.csv":
            continue
        
        tmp_im_org = cv2.imread(os.path.join(outputpath, pathi))
        src_h, src_w = tmp_im_org.shape[:2]
        # 进行一波腐蚀
        gray = cv2.cvtColor(tmp_im_org,cv2.COLOR_BGR2GRAY)
        ret,binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        dst = cv2.erode(binary,kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(4,4))
        dst = cv2.dilate(dst,kernel)
        dst = dst[...,np.newaxis]
        tmp_im_org = np.repeat(dst,3,axis=2)
        binary_mask = tmp_im_org.copy() 

        # 对mask进行选取颜色
        tmp_im = tmp_im_org.copy()
        color = np.random.randint(0,255,size=(3))
        tmp_im[:,:,0][tmp_im[:,:,0]==255] = color[0]
        tmp_im[:,:,1][tmp_im[:,:,1]==255] = color[1]
        tmp_im[:,:,2][tmp_im[:,:,2]==255] = color[2]
        draw_im = cv2.addWeighted(tmp_im, alpha, im, beta,0)
        
        # 对mask获取轮廓点
        thresh = cv2.Canny(tmp_im_org, 128, 256)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 
        areas = []
        for c in range(len(contours)):
            areas.append(cv2.contourArea(contours[c]))
        max_id = areas.index(max(areas))
        x, y, w, h = cv2.boundingRect(contours[max_id])
       
        
        # 对mask获取轮廓点
        polygons = process(binary_mask)
        
        if len(polygons)>=2:
            print("len(polygons):{}".format(len(polygons)))
            for polygoni in polygons:
                polygoni = np.array(polygoni).reshape(-1,2)
                print(polygoni.shape)
                if len(polygoni.tolist())==0:
                    continue
                shape_ = {
                "label":"stone_{}".format(indx),
                "points": polygoni.tolist(),
                "group_id":None,
                "shape_type":"polygon",
                "flags":{},
                "rect": [[int(x), int(y)], [int(x+w), int(y+h)]]
                }
                annos["shapes"].append(shape_)
                # cv2.imwrite("haha.jpg", binary_mask)


        else:
            polygons = np.array(polygons).reshape(-1,2)
            if len(polygons.tolist())==0:
                    continue
            shape_ = {
                "label":"stone_{}".format(indx),
                "points": polygons.tolist(),
                "group_id":None,
                "shape_type":"polygon",
                "flags":{},
                "rect": [[int(x), int(y)], [int(x+w), int(y+h)]]
            }
            annos["shapes"].append(shape_)

            
        im = draw_im
        dx = (x+w/2)/src_w
        dy = (y+h/2)/src_h
        dw = w/src_w
        dh = h/src_h
        cons.append("{} {} {} {} {}\n".format(0, dx, dy, dw, dh))
    
    # with open(os.path.join(datafolder_labels, imgname.replace("jpg","txt")), "w") as ff:
    #     for coni in cons:
    #         ff.write(coni)
            
    with open(os.path.join(datafolder_jsons, imgname.replace("jpg","json")), "w") as ff:
        json.dump(annos, ff, indent=2)
    
    cv2.imwrite(os.path.join(savepath, imgname), im)
    # cv2.imwrite(os.path.join(datafolder_images, imgname), im_copy)

    # exit()


