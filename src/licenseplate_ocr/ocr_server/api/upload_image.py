from flask import Blueprint
from licenseplate_ocr.ocr_engine.ocr_recognizer import OCR_process
from flask import render_template, request, abort, jsonify
from licenseplate_ocr.utils.config import SUPER_RESOLUTION, YOLOX, PADDLE_OCR
from licenseplate_ocr.utils.cvImgToBase64 import cvImgToBase64
import cv2
import base64
from PIL import Image
from io import BytesIO
import os
import sys
import numpy as np
from licenseplate_ocr.utils.logger import LicensePlateLogger
from typing import Tuple

sys.path.append(os.path.dirname("../ocr_engine/exps/example/yolox_voc/yolox_voc_s.py"))

BASE_PATH = os.getcwd()
RESULT_PATH = os.path.join(BASE_PATH, 'static')

class UploadImageAPI(Blueprint):
    def __init__(self, **kwargs) -> None:
        super().__init__("upload_process", __name__, **kwargs)
        self.add_url_rule("/upload_image", view_func=self.upload_image, methods=['POST'])
        # self.__logger = LicensePlateLogger().get_instance(gen_log=True)

    def upload_image(self):
        try:
            base64_image = request.json.get("base64_image")
            image_name = request.json.get("filename")
            id = request.json.get("ID")

            if not base64_image and image_name=="" and id=="":
                return "File Not Found!", 404

            # print("Base64: ", base64_image)
            print("filename: ", image_name)

            server_path = os.path.join(RESULT_PATH, id)
            if not os.path.exists(server_path):
                os.makedirs(server_path, exist_ok=True)

            upload_path = os.path.join(server_path, 'upload')
            if not os.path.exists(upload_path):
                os.makedirs(upload_path, exist_ok=True)
            
            decoded = base64.b64decode(base64_image)
            uploaded_img = np.array(Image.open(BytesIO(decoded)))
            # upload_img = np.fromstring(decoded, dtype=np.uint8)
            # img = cv2.imdecode(upload_img, flags=cv2.IMREAD_GRAYSCALE)
            image = cv2.cvtColor(uploaded_img, cv2.COLOR_BGR2RGB)
            print(image.shape)
            
            cv2.imwrite(f"{upload_path}/{image_name}", image)

            result_image, roi_list, coords, plate_numbers = self.__ocrProcessJob(image)
            print("ResImg Shape: ", result_image.shape)
            
            print("ROI List in API: ", roi_list)

            if len(roi_list) != 0:

                pred_path = os.path.join(server_path, 'pred')
                if not os.path.exists(pred_path):
                    os.makedirs(pred_path, exist_ok=True)
                cv2.imwrite(f"{pred_path}/{image_name}", result_image)

                # Convert BGR before sending as base64
                bgr_res_img = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
                base64String = cvImgToBase64(bgr_res_img)

                roi_path = os.path.join(server_path, 'roi')
                if not os.path.exists(roi_path):
                    os.makedirs(roi_path, exist_ok=True)
                    
                img_name = image_name.split('.')[0]
                extension = image_name.split('.')[1]
                roi_base64_list = []
                for idx, roi in enumerate(roi_list):
                    roi_img_name = f"{img_name}_{idx}.{extension}"
                    bgr_roi_img = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
                    roi_b64 = cvImgToBase64(bgr_roi_img)
                    roi_base64_list.append(roi_b64)
                    cv2.imwrite(f"{roi_path}/{roi_img_name}", roi)

            # TO DO: return base64 for non detected image
            else:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(result_image, plate_numbers[0], (30, 60), font, 0.7, (0, 0, 255), 2)
                # Convert BGR before sending as base64
                bgr_res_img = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
                base64String = cvImgToBase64(bgr_res_img)
                roi_base64_list = []

            return jsonify({'result_base64': base64String, 'roi_base64_list': roi_base64_list, 'plate_numbers_list': plate_numbers})
        
        except:
            abort(500)

    def __ocrProcessJob(self, server_uploaded: str) -> Tuple[np.ndarray, list, list, list]:
        ocrPred = OCR_process(YOLOX, PADDLE_OCR, SUPER_RESOLUTION)
        print(server_uploaded)
        result_image, result_roi_imgs, coords, plate_numbers = ocrPred.ocr_demo(server_uploaded)
        print("Result Image::: ", result_image)
        print("Result ROI Images::: ", result_roi_imgs)
        print("coords of infer::: ", coords)
        print("Plate Numbers::: ", plate_numbers)
        return result_image, result_roi_imgs, coords, plate_numbers