import os
import cv2
import numpy as np
import shutil

labelfolder = r"/root/project/Modules/yolov5/runs/detect/exp/labels"
imagefolder = r"/root/project/Modules/yolov5/runs/detect/exp"
cropfolder = r"/root/project/Modules/yolov5/runs/detect/exp/crops"

if not os.path.exists(cropfolder):
    os.mkdir(cropfolder)


for pathi in os.listdir(labelfolder):
    print(pathi)
    with open(os.path.join(labelfolder, pathi), "r") as ff:
        datas = ff.readlines()
        
    h , w = 512, 512
    imagepath = r"/root/project/Datas/Stones/val/images/[4552]7.jpg"
    basename_ = os.path.basename(imagepath)
    im = cv2.imread(imagepath)
    for index_, datai in enumerate(datas):
        datai = datai[:-1].split(" ")[1:]
        datai = np.array(datai, dtype=float)*512
        cx,cy,dw,dh = datai
        
        x1 = cx-dw/2
        y1 = cy-dh/2
        x2 = cx+dw/2
        y2 = cy+dh/2
        
        imCrop = im[int(y1):int(y2), int(x1):int(x2)]
        newpath = "{}_{}*{}*{}*{}.jpg".format(basename_.split(".")[0], int(x1), int(x2), int(y1), int(y2))
        cv2.imwrite(os.path.join(cropfolder,newpath), imCrop)
        # cv2.rectangle(im, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 1, 1)
        
        
        
        
     
    
        cv2.imwrite("tmp_.jpg", im)
        # exit(1)
    exit()









