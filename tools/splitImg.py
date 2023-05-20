import os
import numpy as np
import shutil
import glob
from PIL import Image
import cv2

# 需要切割图像的路径
rootfolder = r"/root/project/Modules/yolov5/tools/splitSeg/images"

# 图像切割后的路径
savefolder = r"/root/project/Modules/yolov5/tools/saveSplit"

# 切割图像的大小
splitSize = (512,512)
splitw, splith = splitSize

imagePaths = glob.glob(os.path.join(rootfolder, "*.jpg"))
imagePaths.extend(glob.glob(os.path.join(rootfolder, "*.png")))
imagePaths.extend(glob.glob(os.path.join(rootfolder, "*.JPG")))

# np.random.shuffle(imagePaths)
if not os.path.exists(savefolder):
    os.mkdir(savefolder)
    
for i, imagepath in enumerate(imagePaths):
    basename_ = os.path.basename(imagepath)
    
    folder_ = os.path.join(savefolder, basename_.split(".")[0])
    if not os.path.exists(folder_):
        os.mkdir(folder_)

    img = cv2.imread(imagepath)
    imageHight, imageWidth = img.shape[:2]
    
    cols = int(np.ceil(imageWidth/splitw))
    rows = int(np.ceil(imageHight/splith))
    
    index_ = 0
    drawimg = np.zeros((imageWidth, imageHight, 3)).astype(np.uint8) # tmp
    cols_img = []
    for row_ in range(rows):
        raws_img = []
        for col_ in range(cols):
            if col_ == cols-1:
                cropimg = img[row_*splith:(row_+1)*splith, imageWidth-512:imageWidth]
                otherw = (col_+1)*splitw - imageWidth
                cropimg = cropimg[0:splith, otherw:splitw]
            # elif row_ == rows-1:
            #     cropimg = img[row_*splith:(row_+1)*splith, col_*splitw:(col_+1)*splitw]
                # otherw = (col_+1)*splitw - imageWidth
                # cropimg = cropimg[0:splith, otherw:splitw]
            else:
                cropimg = img[row_*splith:(row_+1)*splith, col_*splitw:(col_+1)*splitw]  
            raws_img.append(cropimg)
            cv2.imwrite(os.path.join(folder_, "{}_{}.jpg".format(basename_.split(".")[0], index_)), cropimg)
            index_+=1
        
        # 对行的图像进行拼接
        # tmp_img = np.concatenate(raws_img, axis=1)  
        # cols_img.append(tmp_img)
        
    # 对列的图像进行拼接，获得原图
    # tmp_img = np.concatenate(cols_img)
    # cv2.imwrite("tmp.jpg", tmp_img)






