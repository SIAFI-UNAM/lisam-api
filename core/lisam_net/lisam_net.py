import copy
import csv
import itertools
from typing import List
import numpy as np

import mediapipe as mp

from core.constants.const import X_INPUT_SIZE, Y_INPUT_SIZE
from core.lisam_net.inference_result import InferenceResult
from core.lisam_net.key_point_classifier.key_point_classifier import KeyPointClassifier


class LisamNet:
    def __init__(
        self,
        labels_path: str,
        key_point_model_path: str,
    ) -> None:

        self.load_keypoint_classifier(key_point_model_path)
        self.load_hand_detention_model()

        self.load_names(labels_path)

    def load_names(self, NAMES_PATH: str) -> None:
        with open(NAMES_PATH,
              encoding='utf-8-sig') as f:
            self.keypoint_classifier_labels = csv.reader(f)
            self.keypoint_classifier_labels = [
                row[0] for row in self.keypoint_classifier_labels
            ]

    def run_inference(self, frame: np.ndarray) -> List[InferenceResult]:
        results = self.hands_detector.process(frame)

        if results.multi_hand_landmarks is None:
            return []
        
        results: List[InferenceResult] = []
        
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
            landmark_list = calc_landmark_list(frame, hand_landmarks)
            # Conversion to relative coordinates / normalized coordinates
            pre_processed_landmark_list = pre_process_landmark(
                landmark_list)
            
            hand_sign_id = self.keyponit_classifier.classify_landmark(pre_processed_landmark_list)
        
            results.append(
                InferenceResult(self.keypoint_classifier_labels[hand_sign_id])
            )

        return results

    def load_hand_detention_model(self) -> None:
        mp_hands = mp.solutions.hands
        self.hands_detector = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

    def load_keypoint_classifier(self,
                                 model_path: str,
                                 ) -> None:
        self.keyponit_classifier = KeyPointClassifier(model_path)


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list