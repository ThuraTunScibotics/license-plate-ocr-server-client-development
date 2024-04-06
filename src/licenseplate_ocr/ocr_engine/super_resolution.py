import cv2
from cv2 import dnn_superres
import os


def InitializeModel(sr_model_pth, scale):
    sr = dnn_superres.DnnSuperResImpl_create()
    # sr = dnn_superres.
    sr.readModel(sr_model_pth)
    model_name = sr_model_pth.split(os.path.sep)[-1].split("_")[0].lower()
    sr.setModel(model_name, scale)
    return sr


def spr_resolution(image, sr):
    img = sr.upsample(image)
    print(img.shape)
    #result = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img