from cv2 import Mat
import numpy as np
import cv2


class PreprocessingService:
    def bytes_to_cv2_image(self, img_bytes: bytes):
        array_img = np.fromstring(img_bytes, np.uint8)  # type: ignore
        cv2_img = cv2.imdecode(array_img, cv2.IMREAD_COLOR)

        return cv2_img
