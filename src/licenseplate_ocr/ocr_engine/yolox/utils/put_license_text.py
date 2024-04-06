#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import cv2
import numpy as np
from typing import Tuple

def put_lincense_txt(image: np.ndarray, coord0: list, lincense_text: str) -> None:
    
    # print( coord0 )
    # print("Coord Type: ", type(coord0))

    x0, y0 = coord0[0], coord0[1]

    #image = img.copy()

    lt = 1 or round(0.0008* (image.shape[0] + image.shape[1]) / 2) 
    ft = max(lt - 1, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5

    # Create white rectangle with the same size with image
    text_size = cv2.getTextSize(lincense_text, font, font_scale, 2)[0]
    textx = x0 - 15
    texty = y0 - 15


    rect_x = textx - ft
    rect_y = texty - text_size[1] - ft
    rect_width = text_size[0] + ft
    rect_height = text_size[1] + ft * 2

    cv2.rectangle(image, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (255, 255, 255), -1)
        
    text_color = [0, 0, 255]
    text = '{}'.format(lincense_text)

    cv2.putText(image, text, (textx, texty), font, font_scale, text_color, ft, lineType=cv2.LINE_AA)