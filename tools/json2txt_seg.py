import os
import numpy as np
import json
import cv2
from tqdm import tqdm

rootfolder = r"/root/project/Modules/yolov5/example/datasets"
imgfolder = os.path.join(rootfolder, "images")
jsonfolder = os.path.join(rootfolder, "jsons")
save_label = os.path.join(rootfolder, "labels")

if not os.path.exists(save_label):
    os.mkdir(save_label)

for pathi in tqdm(os.listdir(imgfolder)):
    with open(os.path.join(jsonfolder, pathi.replace("jpg","json")), "r") as ff:
        datas = json.load(ff)
    basename_ = pathi.split(".jpg")[0]
    
    shapes = datas["shapes"]
    imageHeight = datas["imageHeight"]
    imageWidth = datas["imageWidth"]

    cons = []
    for index_ in range(len(shapes)):
        shapei = shapes[index_]
        seg = np.array(shapei["points"]).reshape(-1,2)
        dseg = (seg)/np.array([imageWidth,imageHeight])
        str_seg ="0 "
        for segi in dseg:
            str_seg += "{:.4f} {:.4f} ".format(segi[0], segi[1]) 
        cons.append(str_seg)
    with open(os.path.join(save_label, "{}.txt".format(basename_)), "w") as ff:
        for consi in cons:
            ff.write("{}\n".format(consi))



