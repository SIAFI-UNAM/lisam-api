from typing import List
import cv2
import numpy as np

from core.constants.const import X_INPUT_SIZE, Y_INPUT_SIZE
from core.lisam_net.inference_result import InferenceResult


class LisamNet:
    def __init__(
        self,
        CFG_PATH: str,
        WEIGHTS_PATH: str,
        NAMES_PATH: str
    ) -> None:
        self.net: cv2.dnn_DetectionModel = cv2.dnn_DetectionModel(
            CFG_PATH, WEIGHTS_PATH)
        self.set_network_settings()

        self.load_names(NAMES_PATH)

    def set_network_settings(self) -> None:
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        self.net.setInputSize(X_INPUT_SIZE, Y_INPUT_SIZE)
        self.net.setInputScale(1.0 / 255)
        self.net.setInputSwapRB(True)

    def load_names(self, NAMES_PATH: str) -> None:
        with open(NAMES_PATH, 'rt') as f:
            self.names = f.read().rstrip('\n').split('\n')

    def run_inference(self, frame: np.ndarray) -> List[InferenceResult]:
        classes, confidences, boxes = self.net.detect(
            frame,
            confThreshold=0.1,
            nmsThreshold=0.4
        )

        results = []

        isEmptyInferences = len(classes) == 0
        if isEmptyInferences:
            return results

        for classId, confidence, box in zip(classes.flatten(),
                                            confidences.flatten(),
                                            boxes):
            results.append(InferenceResult(
                self.names[classId], confidence, box))

        return results
