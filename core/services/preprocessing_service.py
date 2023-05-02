import io
import numpy as np
import cv2
import PIL.Image as Image

class PreprocessingService:

    def rezize_relative(self, img: np.array, scale: float) -> np.array:
        height, width, _ = img.shape
        new_height = int(height * scale)
        new_width = int(width * scale)
        new_size = (new_width, new_height)

        return cv2.resize(img, new_size)

    def bytes_to_cv2_image(self, img_bytes: bytes):
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB') 

        np_image = np.array(image) 
        img_cv2 = np_image[:, :, ::-1].copy() 
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

        return img_cv2

    def to_graysacle(self, img: np.array) -> np.array:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return gray
