import numpy as np
import cv2


class PreprocessingService:

    def rezize_relative(self, img: np.array, scale: float) -> np.array:
        height, width, _ = img.shape
        new_height = int(height * scale)
        new_width = int(width * scale)
        new_size = (new_width, new_height)

        return cv2.resize(img, new_size)

    def bytes_to_cv2_image(self, img_bytes: bytes):
        array_img = np.fromstring(img_bytes, np.uint8)  # type: ignore
        cv2_img = cv2.imdecode(array_img, cv2.IMREAD_COLOR)

        return cv2_img

    def to_graysacle(self, img: np.array) -> np.array:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return gray
