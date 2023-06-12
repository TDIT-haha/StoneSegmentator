import os
import json
import numpy as np

jsonfolder = r"/root/project/Datas/minStones/V2/seg/jsons"
txtfolder = r"/root/project/Datas/minStones/V2/detect/labels"


for pathi in os.listdir(jsonfolder):
    with open(os.path.join(jsonfolder, pathi), "r") as ff:
        datas = json.load(ff)
        
    shapes = datas["shapes"]
    imgname = datas["imagePath"]
    imageHeight = datas["imageHeight"]
    imageWidth = datas["imageWidth"]
    cons = []
    for shapei in shapes:
        points = np.array(shapei["points"]).reshape(-1,2)
        x1 = np.min(points[:,0:1])
        x2 = np.max(points[:,0:1])
        y1 = np.min(points[:,1:2])
        y2 = np.max(points[:,1:2])
        # print(points[:,1:2])
        # print("{} {} {} {} {}\n".format(0, x1, x2, y1, y2))
        # exit()
        
        dx = ((x1+x2)/2)/imageWidth
        dy = ((y1+y2)/2)/imageHeight
        dw = ((x2-x1))/imageWidth
        dh = ((y2-y1))/imageHeight
        cons.append("{} {} {} {} {}\n".format(0, dx, dy, dw, dh))
        
    with open(os.path.join(txtfolder , imgname.replace("jpg","txt")), "w") as ff:
        for coni in cons:
            ff.write(coni)