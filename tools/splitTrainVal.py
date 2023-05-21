import os
import shutil
import numpy as np

trainRate = 0.8
valRate = 1 - trainRate

rootfolder = r"/root/project/Modules/yolov5/example/saveSplit"
savefolder = r"/root/project/Modules/yolov5/example/datasets"
trainfoler = os.path.join(savefolder, "train")
valfoler = os.path.join(savefolder, "val")
if not os.path.exists(trainfoler):
    os.mkdir(trainfoler)
if not os.path.exists(valfoler):
    os.mkdir(valfoler)
if not os.path.exists(os.path.join(trainfoler, "images")):
    os.mkdir(os.path.join(trainfoler, "images"))
if not os.path.exists(os.path.join(valfoler, "images")):
    os.mkdir(os.path.join(valfoler, "images"))

folders = os.listdir(rootfolder)
lens = len(folders)
np.random.shuffle(folders)

trainNUm = int(lens*trainRate)
valNUm = lens-trainNUm

for i, foldername in enumerate(folders):
    if i < trainNUm:
        for pathname in os.listdir(os.path.join(rootfolder, foldername)):
            shutil.copy(os.path.join(rootfolder, foldername, pathname), os.path.join(trainfoler, "images", pathname))

    else:
        for pathname in os.listdir(os.path.join(rootfolder, foldername)):
            shutil.copy(os.path.join(rootfolder, foldername, pathname), os.path.join(valfoler, "images", pathname))

