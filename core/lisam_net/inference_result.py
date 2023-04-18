from typing import Any, Dict
import numpy as np


class InferenceBox:
    def __init__(self, box: np.ndarray) -> None:
        self.xMin = box[0]
        self.xMax = box[1]
        self.yMin = box[2]
        self.yMax = box[3]


class InferenceResult:
    def __init__(self,
                 label: str,
    ) -> None:

        self.label = label

    def to_dict(self) -> Dict[str, Any]:
        return {
            'label': self.label,
        }
