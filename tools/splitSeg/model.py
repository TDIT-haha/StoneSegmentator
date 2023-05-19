import os
from tqdm import tqdm
import numpy as np
import onnxruntime
import torch
from tools import *

class StoneSeg:
    def __init__(self, modelpath=None, className=None, size=None, conf_thres=0.5, iou_thres=0.5): #必须要有一个self参数，
        self.modelpath = modelpath
        self.className = className
        self.img_size = size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def modelinit(self):
        self.session = onnxruntime.InferenceSession(self.modelpath, None)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, im0):
        im, ratio, (dw, dh) = letterbox(im0, self.img_size, stride=32, auto=False)  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        return im

    def postprocess(self, pred, proto):
        conf_thres = self.conf_thres
        iou_thres = self.iou_thres
        classes = None
        agnostic_nms = False
        max_det = 1000

        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)
        proto = torch.from_numpy(proto)
        
        results_det = []
        results_seg = []
        
        for i, det in enumerate(pred):
            if len(det):
                masks = process_mask(proto[i], det[:, 6:], det[:, :4], [self.img_size, self.img_size], upsample=True)  # HWC
                det[:, :4] = scale_boxes([self.img_size, self.img_size], det[:, :4], self.im.shape[:2]).round()
                masks = masks.detach().cpu().numpy().astype(np.uint8)
                masks[masks==1]=255
                
            for onemask in masks:
                newmaski = scale_image((640, 640), onemask, (512, 512))
                newmaski = newmaski[:,:,0]
                results_seg.append(newmaski)
                    
            for *xyxy, conf, cls in reversed(det[:, :6]):
                x1,y1,x2,y2 = xyxy
                results_det.append([x1,y1,x2,y2])


        return results_det, results_seg

    def model_inter(self, input_data):
        input_data = np.expand_dims(input_data, 0).astype(np.float32)
        input_data = input_data/255.0
        pred, proto = self.session.run([], {self.input_name: input_data})
        return pred, proto 

    def inter(self, image):
        self.im = image.copy()
        blob = self.preprocess(image)
        pred, proto = self.model_inter(blob)
        dets, masks = self.postprocess(pred, proto)
        return dets, masks



if __name__ == "__main__":
    # 模型的初始化
    modelpath = "/root/project/Modules/yolov5/runs/train-minseg/V3/weights/best.onnx"
    model = StoneSeg(modelpath, className=["stone"], size=640, conf_thres=0.3, iou_thres=0.45)
    model.modelinit()

    one_images_detect = False
    split_images_detect = True

    # # one images
    if one_images_detect:
        imagepath = r"/root/project/Modules/yolov5/data/images/[1.8]11.jpg"
        img = cv2.imread(imagepath)
        h,w = img.shape[:2]
        dets, masks = model.inter(img)
        new_mask = np.zeros((w, h, 3)).astype(np.uint8)
        totalAreas = []
        for mask_ in masks:
            # cv2.imwrite("mask.jpg", mask_)
            contours, hierarchy = cv2.findContours(mask_, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 
            areas = []
            for c in range(len(contours)):
                areas.append(cv2.contourArea(contours[0]))
            max_areas = np.max(areas)
            totalAreas.append(max_areas)
                
            color = np.random.randint(0,255,size=(3))
            new_mask[:,:,0][mask_==255]=color[0]
            new_mask[:,:,1][mask_==255]=color[1]
            new_mask[:,:,2][mask_==255]=color[2]
            
        cv2.imwrite("draw_{}".format(os.path.basename(imagepath)), new_mask)
        totalAreas.sort()
        print("共有块数：{}".format(len(totalAreas)))
        print("面积分别有:{}".format(totalAreas))
    
    
    # split images
    if split_images_detect:
        savefolder = r"./draws"
        if not os.path.exists(savefolder):
            os.mkdir(savefolder)
            
        set_split_size = (512, 512)
        imagepath = r"./A_4542.JPG"
        img = cv2.imread(imagepath)
        h,w = img.shape[:2]

        newimg = cv2.copyMakeBorder(img, 0, 96, 0, 144, cv2.BORDER_CONSTANT, value=(0,0,0))  # add border
        h,w = newimg.shape[:2]
        print("{} {}".format(w, h))
        
        drawimg = np.zeros((w, h, 3)).astype(np.uint8)
        colsimg = []
        totalAreas = []
        for i in tqdm(range(12)):
            raws_img = []
            for j in tqdm(range(8)):
                cropimg = newimg[j*512:(j+1)*512, i*512:(i+1)*512]
                dets, masks = model.inter(cropimg)
                
                h_,w_ = cropimg.shape[:2]
                new_mask = np.zeros((w_, h_, 3)).astype(np.uint8)
                for mask_ in masks:
                    mask_[mask_<255]=0
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
                    # if int(max_areas) == 3:
                    #     cv2.imwrite("mask.jpg", mask_)
                    #     cv2.imwrite("cropimg.jpg", cropimg)
                    #     exit()
                    totalAreas.append(max_areas)
                    
                    color = np.random.randint(0,255,size=(3))
                    new_mask[:,:,0][mask_==255]=color[0]
                    new_mask[:,:,1][mask_==255]=color[1]
                    new_mask[:,:,2][mask_==255]=color[2]
                    
                raws_img.append(new_mask)

            tmp_img = np.concatenate(raws_img)         
            colsimg.append(tmp_img)
                    
        tmp_img = np.concatenate(colsimg, axis=1) 
        cv2.imwrite("tmp.jpg", tmp_img)
        # cv2.imwrite("draw_{}".format(os.path.basename(imagepath)), drawimg)
        totalAreas.sort()
        print("共有块数：{}".format(len(totalAreas)))
        print("面积分别有:{}".format(totalAreas))
        
