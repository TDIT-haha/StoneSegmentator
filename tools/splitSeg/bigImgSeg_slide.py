import os
from tqdm import tqdm
import numpy as np
import onnxruntime
import torch
import glob
from codes.tools import *
from codes.model import StoneSeg
import argparse
import torchvision

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default="/root/project/Modules/yolov5/runs/train-minseg/V3/weights/best.onnx", help='onnx model path')
    parser.add_argument('--imagefolder', type=str, default=r"./images", help='imagefolder')
    parser.add_argument('--savefolder', type=str, default=r"./draws", help='savefolder')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=640, help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--scale-factor', type=float, default=1.0, help='NMS IoU threshold')
    opt = parser.parse_known_args()[0]
    
    # 模型的初始化
    modelpath = opt.weights
    model = StoneSeg(modelpath, className=["stone"], size=640, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres)
    model.modelinit()
    model.modelwarmup()
    # 比例尺系数，默认为1.0
    scale_factor = opt.scale_factor

    savefolder = opt.savefolder
    if not os.path.exists(savefolder):
        os.mkdir(savefolder)
    
    # 切割图像的大小
    set_split_size = (512, 512)
    slidew, slideh = 256, 256
    splitw, splith = set_split_size
    
    imagefolder = opt.imagefolder
    imagelist = glob.glob(os.path.join(imagefolder, "*.JPG"))
    
    print("load image .......")
    for index_, imagepath in enumerate(imagelist):
        print(f"--------------------index_: {index_}------------------------")
        # imagepath = r"./images/A_4542.JPG"
        basename_ = os.path.basename(imagepath)
        img = cv2.imread(imagepath)
        imageHight, imageWidth = img.shape[:2]
        cols = int(np.ceil((imageWidth-splitw)/slidew))+1
        rows = int(np.ceil((imageHight-splith)/slideh))+1
        
        drawimg = np.zeros((imageWidth, imageHight, 3)).astype(np.uint8)
        cols_img = []
        totalAreas = []
        allDets = []  
        allMasks = []  
        allTopPoints = []  
         
        for row_ in tqdm(range(rows)):
            rows_img = []
            for col_ in tqdm(range(cols)):
                if col_ == cols-1 and row_ != rows-1:
                    print("coming in 1")
                    otherw = (col_+1)*splitw - imageWidth
                    otherh = 0
                    cropimg = img[row_*slideh:row_*(splith-slideh)+splith, imageWidth-splitw:imageWidth]
                    toppoint_x = imageWidth-splitw
                    toppoint_y = row_*slideh
                
                elif row_ == rows-1 and col_ != cols-1:
                    print("coming in 2")
                    otherw = 0
                    otherh = (row_+1)*splith - imageHight
                    cropimg = img[imageHight-splith:imageHight, col_*slidew:col_*(splitw-slidew)+splitw]
                    toppoint_x = col_*slidew
                    toppoint_y = imageHight-splith
                
                elif row_ == rows-1 and col_ == cols-1:
                    print("coming in 3")
                    otherh = (row_+1)*splith - imageHight
                    otherw = (col_+1)*splitw - imageWidth
                    cropimg = img[imageHight-splith:imageHight, imageWidth-splitw:imageWidth]
                    toppoint_x = imageWidth-splitw
                    toppoint_y = imageHight-splith
                else:
                    print("coming in 4")
                    cropimg = img[row_*slideh:row_*(splith-slideh)+splith, col_*slidew:col_*(splitw-slidew)+splitw]
                    otherw = 0
                    otherh = 0
                    toppoint_x = col_*slidew
                    toppoint_y = row_*slideh
                
                dets, masks = model.inter(cropimg)
                masks = np.array(masks)
                dets = np.array(dets)
                
                
                dets[:,0:4] = dets[:,0:4]+np.array([toppoint_x, toppoint_y,toppoint_x, toppoint_y])
                # masks = masks+np.array([toppoint_x, toppoint_y])
                
                allDets.extend(dets)
                allMasks.extend(masks)
                allTopPoints.extend(np.array([toppoint_x, toppoint_y]*len(masks)).reshape(-1,2))
                
        allDets = np.array(allDets)
        allMasks = np.array(allMasks)
        allTopPoints = np.array(allTopPoints)
        
        boxes, scores = allDets[:,0:4], allDets[:,5]
        boxes = torch.from_numpy(boxes)
        scores = torch.from_numpy(scores)
        i = torchvision.ops.nms(boxes, scores, 0.3)  # NMS
        allDets = allDets[i]
        allMasks = allMasks[i]
        allTopPoints = allTopPoints[i]
        
        # newimg = img[0:512, 0:imageWidth]
        newimg = img
        newimg_mask = newimg.copy()*0


        for j, [mask_, toppoint_, bbox_] in enumerate(zip(allMasks,allTopPoints, allDets)):
            # 逻辑处理与计算
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
            totalAreas.append(max_areas)
            
            
            x_, y_ = toppoint_
            newimg_mask_crop = newimg_mask[y_:y_+512,x_:x_+512]
            color = np.random.randint(125,255,size=(3))
            newimg_mask_crop[:,:,0][mask_==1]=color[0]
            newimg_mask_crop[:,:,1][mask_==1]=color[1]
            newimg_mask_crop[:,:,2][mask_==1]=color[2]
            
            x1,y1,x2,y2 = bbox_[:4]
            # if j==413:
            #     color = np.random.randint(125,255,size=(3))
            #     newimg_mask_crop[:,:,0][mask_==1]=255
            #     newimg_mask_crop[:,:,1][mask_==1]=255
            #     newimg_mask_crop[:,:,2][mask_==1]=255
            #     cv2.rectangle(newimg_mask, (int(x1),int(y1)), (int(x2),int(y2)), (0,0,255), 1, 1)
            #     break
                
            # else:
            cv2.rectangle(newimg_mask, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 1, 1)
            cv2.putText(newimg_mask, f"({j})", (int((x1+x2)/2),int((y1+y2)/2)),0,1,(0,0,255),thickness=1,lineType=cv2.LINE_AA)
            

        
        alpha = 0.4
        beta = 0.8
        transparent_im = cv2.addWeighted(newimg_mask, alpha, newimg, beta, 0)
        
        cv2.imwrite("tmp.jpg", newimg)     
        cv2.imwrite("tmp_mask.jpg", newimg_mask)     
        cv2.imwrite("tmp_transparent_im.jpg", transparent_im)     
        totalAreas_ = [num * scale_factor for num in totalAreas]
        totalAreas.sort()
        print("共有块数：{}".format(len(totalAreas)))
        print("面积分别有:{}".format(totalAreas))
        exit()
       
                
                
        

        