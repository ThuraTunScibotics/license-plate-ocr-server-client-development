import torch
import cv2
import numpy as np
import os, time
from licenseplate_ocr.ocr_engine.yolox_predictor import Predictor
from loguru import logger
from typing import Tuple
from paddleocr import PaddleOCR
from licenseplate_ocr.ocr_engine.yolox.exp import get_exp
from licenseplate_ocr.ocr_engine.yolox.utils.put_license_text import put_lincense_txt
from licenseplate_ocr.ocr_engine.yolox.data.datasets.voc_classes import VOC_CLASSES
from licenseplate_ocr.ocr_engine.super_resolution import InitializeModel, spr_resolution
from licenseplate_ocr.utils.config import SUPER_RESOLUTION, YOLOX, PADDLE_OCR
from licenseplate_ocr.utils.logger import LicensePlateLogger

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

class OCR_process:
    def __init__(self,
                 yoloxcfg = YOLOX,
                 paddlecfg = PADDLE_OCR,
                 sprcfg = SUPER_RESOLUTION):
        self.__yoloxcfg = yoloxcfg
        self.__paddlecfg = paddlecfg
        self.__sprcfg = sprcfg
        self.__exp = get_exp(self.__yoloxcfg["exp_file"], None)
        self.__logger = LicensePlateLogger().get_instance(gen_log=True)
        self.predictor = self.detect_license_plate()
        self.__current_time = time.localtime()

    def detect_license_plate(self):
        file_name = os.path.join(self.__exp.output_dir, self.__exp.exp_name)
        os.makedirs(file_name, exist_ok=True)

        device = "gpu" if self.__yoloxcfg["tensor_rt"] else "cpu"
        fp16 = self.__yoloxcfg["fp16"]
        legacy = self.__yoloxcfg["legacy"]
        self.__exp.test_conf = self.__yoloxcfg["conf"]
        self.__exp.nmsthres = self.__yoloxcfg["nms"]
        self.__exp.test_size = self.__yoloxcfg["img_size"]

        model = self.__exp.get_model()

        if device == "gpu":
            model.cuda()
            if fp16:
                model.half()
        model.eval()

        weight_file = self.__yoloxcfg["weight"]
        # print("Weight File Path: ", weight_file)
        ckpt = torch.load(weight_file, map_location="cpu")
        # print("Loaded ckpt")
        model.load_state_dict(ckpt["model"])
        # print("Loaded Model Done: ", model)
        self.__logger.info("loaded checkpoint done.")

        if self.__yoloxcfg["tensor_rt"]:
            trt_file = os.path.join(file_name, "model_trt.pth")
            assert os.path.exists(
            trt_file
            ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
            model.head.decode_in_inference = False
            decoder = model.head.decode_outputs
            self.__logger.info("Using TensorRT to inference")
        else:
            trt_file = None
            decoder = None
        
        predictor = Predictor(
            model, self.__exp, VOC_CLASSES, trt_file, decoder,
            device, fp16, legacy
        )
        self.__logger.info("Initialized predictor object.")

        return predictor

    def ocr_demo(self, image: cv2.Mat) -> Tuple[np.ndarray, list, list, list]:

        # Super Resolution Model & PaddleOCR Initialization
        spr_model_pth = self.__sprcfg["spr_model"]
        spr_r = InitializeModel(sr_model_pth=spr_model_pth, scale=4)
        self.__logger.info(f"Start initializing model from the path -> {spr_model_pth}")

        use_angle = self.__paddlecfg["use_angle"]
        language = self.__paddlecfg["language"]
        ocr = PaddleOCR(use_angle_cls=use_angle, lang=language)
        self.__logger.info("Initialize PaddleOCR.")

        outputs, img_info = self.predictor.inference(image)

        print("Outputs::: ", outputs)
        print("Img Path::: ", image)

        if outputs[0] == None:
            print("Output is None!!!")
            self.__logger.warning("Couldn't detect License Plate...!")
            result_image = image.copy()
            print("Result Image Shape: ", result_image.shape)
            result_roi_imgs = []
            coords = []
            plate_numbers = ["Couldn't detect license plate from this image!"]

        else:

            self.__logger.info("License Plate is detected.")
            result_image, result_roi_imgs, coords = self.predictor.visual(outputs[0], img_info, self.predictor.confthre)

            plate_numbers = []

            for i, (roi, cord) in enumerate(zip(result_roi_imgs, coords)):

                # passed cropped roi_image to super resolution
                resoluted_img = spr_resolution(roi, sr=spr_r)

                # pass cropped resoluted image to paddle ocr
                result = ocr.ocr(resoluted_img, cls=True)
                result = result[0]
                
                if result is None:
                    txts = "Couldn't Recognize"
                    self.__logger.warning("Couldn't recognize the character...!")
                else:
                    txts = [line[1][0] for line in result]

                text = ' '.join(txts)
                print("Recognized Text: ", text)

                # Put License Text on the top of bbox
                put_lincense_txt(result_image, cord, text)

                # result_images.append(result_image)
                plate_numbers.append(text)
        print("Result Image Shape: ", result_image.shape)
        return result_image, result_roi_imgs, coords, plate_numbers

if __name__ == "__main__":
    ocr_pred = OCR_process()
    path = "/home/thuratun/thura_tun_personal/license_plate_web_app/src/licenseplate_ocr/inference_images/N249.jpeg"
    # path = "/home/thuratun/thura_tun_personal/license_plate_web_app/src/licenseplate_ocr/ocr_server/static/upload/N249.jpg"
    result_images, roi_lists, coords, plate_numbers = ocr_pred.ocr_demo(path)

    print("Result Image Type: ", type(result_images))
    print("ROI list len: ", len(roi_lists))
    print("Plate Number len: ", plate_numbers)
    print("Coordinate len: ", len(coords))

    cv2.imshow("Result Image", result_images)
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()