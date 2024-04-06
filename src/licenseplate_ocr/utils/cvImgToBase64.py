from PIL import Image
from io import BytesIO
import base64

def cvImgToBase64(cv_img):
        res_img = Image.fromarray(cv_img.astype('uint8'))
        rawBytes = BytesIO()
        res_img.save(rawBytes, 'PNG')
        rawBytes.seek(0)
        result_base64 = base64.b64encode(rawBytes.read())
        base64String = 'data:image/jpeg;base64,' + str(result_base64).split('\'')[1]
        return base64String