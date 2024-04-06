import os
from typing import Dict
import yaml

def __get_ocr_conf(part: str) -> Dict:
    ocr_cfg = os.path.join(os.path.dirname(__file__), "../configs/ocr.yaml")
    with open(ocr_cfg) as cfg_f:
        cfg_f: Dict = yaml.load(cfg_f, yaml.SafeLoader)
    return cfg_f[part]

def __get_server_conf(part: str) -> Dict:
    server_cfg = os.path.join(os.path.dirname(__file__), "../configs/server.yaml")
    with open(server_cfg) as cfg_f:
        cfg_f: Dict = yaml.load(cfg_f, yaml.SafeLoader)
    return cfg_f[part]

# fmt: off
YOLOX = __get_ocr_conf("yolox")
SUPER_RESOLUTION = __get_ocr_conf("super_resolution")
PADDLE_OCR = __get_ocr_conf("paddle_ocr")
LOGFILE = __get_ocr_conf("logger")
SERVER = __get_server_conf("server")
# fmt: on