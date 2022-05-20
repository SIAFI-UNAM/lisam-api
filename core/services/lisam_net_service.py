from typing import List
from core.lisam_net.inference_result import InferenceResult
from core.lisam_net.lisam_net import LisamNet
from core.services.preprocessing_service import PreprocessingService


class LisamNetService:
    def __init__(self) -> None:
        self.lisam_net = LisamNet()
        self.preprocessing_service = PreprocessingService()

    def inference_img_bytes(self, img: bytes) -> List[InferenceResult]:
        img_cv2 = self.preprocessing_service.bytes_to_cv2_image(img)
        inference_result = self.lisam_net.run_inference(img_cv2)

        return inference_result
