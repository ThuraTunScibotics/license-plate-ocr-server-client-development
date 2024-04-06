import argparse
import os
import time
import numpy as np
from typing import Tuple
import cv2
import torch
from licenseplate_ocr.ocr_engine.yolox.data.data_augment import ValTransform
from licenseplate_ocr.ocr_engine.yolox.data.datasets import COCO_CLASSES
from licenseplate_ocr.ocr_engine.yolox.data.datasets.voc_classes import VOC_CLASSES
from licenseplate_ocr.ocr_engine.yolox.exp import get_exp
from licenseplate_ocr.ocr_engine.yolox.utils import fuse_model, get_model_info, postprocess, vis_img
from licenseplate_ocr.utils.logger import LicensePlateLogger

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=VOC_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):

        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        self.__logger = LicensePlateLogger().get_instance(gen_log=True)
        
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
            print("Image Shape: ", type(img))
        else:
            img_info["file_name"] = None
        
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        print("Test Size: ", self.test_size)

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        # print("Torch Image: ", img)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            # print("Torch Image: ", img.shape)
            # print("MODEL: ", self.model)
            outputs = self.model(img)
            # print("Outputs: ", outputs)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            self.__logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info
    

    def visual(self, output, img_info, cls_conf=0.35) -> Tuple[np.ndarray, list, list]:
        """
        Return:
            vis_res : image result with detected bounding box
            rois: list of cropped images of detected bbox
            coords: list of coordinates of bounding box
        """
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res, rois, coords = vis_img(img, bboxes, scores, cls, cls_conf, self.cls_names)
        
        return vis_res, rois, coords

    # ch = cv2.waitKey(0)
    # if ch == 27 or ch == ord("q") or ch == ord("Q"):
    #     break


if __name__ == "__main__":
    predictor = Predictor()