import io
from typing import List
import cv2

import numpy as np
import PIL.Image as Image

from core.config.net_config import NetConfig
from core.lisam_net.inference_result import InferenceResult
from core.lisam_net.lisam_net import LisamNet
from core.services.preprocessing_service import PreprocessingService


class LisamNetService:
    def __init__(self) -> None:
        self.lisam_net = LisamNet(
            NetConfig.names_path,
            NetConfig.key_point_classifier_model_path,
        )

        self.preprocessing_service = PreprocessingService()

    def inference_img_bytes(self, img_bytes: bytes) -> List[InferenceResult]:
        filepath = './test.jpg'
        image = Image.open(io.BytesIO(img_bytes))

        image.save(filepath)

        img_cv2 = cv2.imread(filepath)
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        img_cv2 = self.preprocessing_service.rezize_relative(img_cv2, 0.3)

        inference_result = self.lisam_net.run_inference(img_cv2)

        return inference_result

    def inference_img(self, img: np.ndarray) -> List[InferenceResult]:
        inference_result = self.lisam_net.run_inference(img)

        return inference_result

