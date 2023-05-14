import os
import cv2
import json
import numpy as np
from tqdm import tqdm

labelfolder = r"/root/project/Modules/yolov5/runs/predict-seg/exp2/labels"
imagefolder = r"/root/project/Modules/yolov5/runs/predict-seg/exp2"
jsonfolder = r"/root/project/Modules/yolov5/runs/predict-seg/exp2/jsons"

if not os.path.exists(jsonfolder):
    os.mkdir(jsonfolder)

for pathi in tqdm(os.listdir(labelfolder)):
    txtname = pathi
    imagename = pathi.replace('txt',"jpg")
    imgpath = os.path.join(imagefolder, imagename)
    txtpath = os.path.join(labelfolder, txtname)


    im = cv2.imread(imgpath)
    h_, w_ = im.shape[:2]
    with open(txtpath, "r") as ff:
        datas = ff.readlines()

    annos = {
        "flags":{},
        "version": "4.5.10",
        "imagePath":imagename,
        "shapes":[],
        "imageHeight":h_,
        "imageWidth":w_,
        "imageData": None
    }
    
    for index_, datai in enumerate(datas):
        datai = np.array(datai[:-1].split(" "), dtype=float)
        name_ = datai[0].copy()
        seg = datai[1:].copy()
        seg = seg.reshape(-1,2)*np.array([w_,h_])

        for point_ in seg:
            cv2.circle(im, point_.astype(int), 2, (0,255,0), -1)
            
        shape_ = {
            "label":"stone_{}".format(index_),
            "points": seg.tolist(),
            "group_id":None,
            "shape_type":"polygon",
            "flags":{},
        }
        annos["shapes"].append(shape_)
        
    with open(os.path.join(jsonfolder, imagename.replace("jpg","json")), "w") as ff:
            json.dump(annos, ff, indent=2)
    
    

