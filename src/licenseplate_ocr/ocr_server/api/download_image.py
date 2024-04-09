from flask import Blueprint
from licenseplate_ocr.ocr_engine.ocr_recognizer import OCR_process
from licenseplate_ocr.utils.cvImgToBase64 import cvImgToBase64
from flask import render_template, request, abort, send_file, jsonify
import cv2
import os, sys
import numpy as np
from licenseplate_ocr.utils.logger import LicensePlateLogger
from typing import Tuple

BASE_PATH = "./static/"
# ROI_PATH = os.path.join(BASE_PATH, 'static/roi_result/')
# PRED_PATH = os.path.join(BASE_PATH, 'static/pred_result/')
# OCR_JSON_PATH = os.path.join(BASE_PATH, 'static/pred_result/')


class DownloadImageAPI(Blueprint):
    def __init__(self, **kwargs) -> None:
        super().__init__("download_image", __name__, **kwargs)
        self.add_url_rule("/download_roi_image", view_func=self.download_roi_image, methods=["GET"])
        self.add_url_rule("/download_pred_image", view_func=self.download_pred_image, methods=["GET"])
    
    def download_roi_image(self):
        
        # filename = resqt_json["image_name"]
        ID = request.args.get("ID")
        filename = request.args.get("filename")

        if ID is None or filename=="":
            return 400
        
        roi_path = os.path.join(BASE_PATH, f"{ID}/roi")
        files = os.listdir(roi_path)
        
        image_files = [file for file in files if file.lower().endswith((".png", ".jpeg", ".jpg"))]
        
        try:
            roi_base64_list = []
            filenames_list = []
            if not os.path.exists(roi_path):
                print("Path Not Exist!")
                abort(404, "Server path not found!")
            for image in image_files:
                img_path = os.path.join(roi_path, image)
                bgr_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_RGB2BGR)
                base64_img = cvImgToBase64(bgr_img)
                filenames_list.append(image)
                roi_base64_list.append(base64_img)
            
            return jsonify({'img_name_list': filenames_list, 'roi_base64_list': roi_base64_list})
        
        except:
            abort(500)

    def download_pred_image(self):

        ID = request.args.get("ID")
        filename = request.args.get("filename")
        print("FIlename:: ", filename)

        if ID is None or filename=="":
            return 400

        pred_path = os.path.join(BASE_PATH, F"{ID}/pred/{filename}")
        print("Pred Path: ", pred_path)

        try:
            if not os.path.exists(pred_path):
                abort(404, "Server path not found!")
            bgr_img = cv2.cvtColor(cv2.imread(pred_path), cv2.COLOR_RGB2BGR)
            base64_img = cvImgToBase64(bgr_img)

            return jsonify({'pred_base64_img': base64_img, 'filename': filename})
        
        except:
            abort(500)





          