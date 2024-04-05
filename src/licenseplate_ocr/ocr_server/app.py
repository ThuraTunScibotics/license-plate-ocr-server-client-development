from licenseplate_ocr.ocr_engine.ocr_recognizer import OCR_process
from flask import Flask, render_template, request
from licenseplate_ocr.utils.config import SUPER_RESOLUTION, YOLOX, PADDLE_OCR, SERVER
from licenseplate_ocr.utils.logger import LicensePlateLogger
from licenseplate_ocr.ocr_server.api import (GeneralAPI, UploadImageAPI, DownloadImageAPI)
from typing import Tuple
import cv2
import numpy as np
import os
import base64
import sys
import warnings

# webserver gateway interface
# app = Flask(__name__)

# sys.path.append(os.path.dirname("../ocr_engine/exps/example/yolox_voc/yolox_voc_s.py"))
# YOLOX_EXP = {"exp_file": "ocr_engine/exps/example/yolox_voc/yolox_voc_s.py"}

# BASE_PATH = os.getcwd()
# UPLOAD_PATH = os.path.join(BASE_PATH, 'static/upload/')
# ROI_PATH = os.path.join(BASE_PATH, 'static/roi_result/')
# PRED_PATH = os.path.join(BASE_PATH, 'static/pred_result/')
# OCR_JSON_PATH = os.path.join(BASE_PATH, 'static/pred_result/')

warnings.filterwarnings(action="ignore")

class FlaskApp:
    def __init__(self) -> None:
        self.__app = Flask(__name__)
        self.__logger = LicensePlateLogger().get_instance(gen_log=True)

        self.__register_apis()

    def __register_apis(self) -> None:
        try:
            general_license = GeneralAPI        ()
            upload_img_license = UploadImageAPI ()
            download_img_license = DownloadImageAPI()

            self.__app.register_blueprint(general_license)
            self.__app.register_blueprint(upload_img_license)
            self.__app.register_blueprint(download_img_license)
        except:
            self.__logger.error("API Registeration Failed.")
            sys.exit()

    def get_app(self) -> Flask:
        return self.__app
    
    def run_api(self) -> Flask:
        self.__logger.info("Start running API....")
        return self.__app.run(
            debug=SERVER["debug"],
            port=SERVER["port"],
            host=SERVER["host"],
        )

if __name__ == "__main__":
    app = FlaskApp()
    app.run_api()