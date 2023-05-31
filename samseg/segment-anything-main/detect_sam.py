import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm

def show_mask(mask, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    return mask_image

labelfolder = r"/root/project/Modules/yolov5/runs/detect/exp/labels"
imagefolder = r"/root/project/Datas/minStones/test/images"
savefolder = r"/root/project/Modules/yolov5/samseg/segment-anything-main/train_ouput"

if not os.path.exists(savefolder):
    os.mkdir(savefolder)

# 初始化模型权重
print("loading model ....")
sam_checkpoint = "/root/autodl-tmp/segeverything_pretrains/vit_h.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

for pathi in tqdm(os.listdir(labelfolder)):
    with open(os.path.join(labelfolder, pathi), "r") as ff:
        datas = ff.readlines()
        
    im = cv2.imread(os.path.join(imagefolder, pathi.replace("txt","jpg")))
    foldername = pathi.split(".")[0]
    
    image = cv2.cvtColor(im.copy(), cv2.COLOR_BGR2RGB)
    #喂入图像
    print("loading image ....")
    predictor.set_image(image)
    
    h_ , w_ = im.shape[:2]
    bboxs = []
    for datai in datas:
        datai = np.array(datai[:-1].split(" "), dtype=np.float64)
        classnum = datai[0]
        cx = datai[1]*w_
        cy = datai[2]*h_
        dw = datai[3]*w_
        dh = datai[4]*h_
        
        x1,y1,x2,y2 = cx-dw/2, cy-dh/2, cx+dw/2, cy+dh/2
        bboxs.append([x1,y1,x2,y2])
  
    input_boxes = torch.tensor(bboxs, device=predictor.device)
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
    # #预测
    print("loading predict ...")
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    
    # #选择框
    # input_box = np.array(bboxs)
    # #预测
    # print("loading predict ...")
    # masks, _, _ = predictor.predict(
    #     point_coords=None,
    #     point_labels=None,
    #     box=input_box[None, :],
    #     multimask_output=False,
    # )
    
    print(len(masks))
    # saveim = show_mask(masks[0].detach().cpu().numpy(), True)
    # print(np.unique(saveim))
    for index_, mask_ in enumerate(masks):
        saveim = mask_.detach().cpu().numpy()
        h, w = saveim.shape[-2:]
        saveim = saveim.reshape(h, w, 1)
        
        if not os.path.exists(os.path.join(savefolder, foldername)):
            os.mkdir(os.path.join(savefolder, foldername))
            
        cv2.imwrite(os.path.join(savefolder, foldername, "{}.png".format(index_)), saveim*255)
    













