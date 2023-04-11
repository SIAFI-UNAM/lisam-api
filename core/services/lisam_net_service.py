from typing import List

import numpy as np
from core.config.net_config import NetConfig
from core.lisam_net.inference_result import InferenceResult
from core.lisam_net.lisam_net import LisamNet
from core.services.preprocessing_service import PreprocessingService


class LisamNetService:
    def __init__(self) -> None:
        self.lisam_net = LisamNet(
            NetConfig.cfg_path,
            NetConfig.weights_path,
            NetConfig.names_path
        )

        self.preprocessing_service = PreprocessingService()

    def inference_img_bytes(self, img_bytes: bytes) -> List[InferenceResult]:
        img_cv2 = self.preprocessing_service.bytes_to_cv2_image(img_bytes)
        img_cv2 = self.preprocessing_service.to_graysacle(img_cv2)
        inference_result = self.lisam_net.run_inference(img_cv2)

        return inference_result

    def inference_img(self, img: np.ndarray) -> List[InferenceResult]:
        inference_result = self.lisam_net.run_inference(img)

        return inference_result

