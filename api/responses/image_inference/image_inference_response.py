from typing import Any, Dict, List
from api.responses.api_response import ApiResponse
from core.lisam_net.inference_result import InferenceResult


class ImageInferenceResponse(ApiResponse):
    def __init__(
            self, inference_results: List[InferenceResult],
            status_code: int) -> None:
        super().__init__(status_code)
        self.inference_results = inference_results

    def to_dict_response(self) -> Dict[str, Any]:
        inferenceResults = [
            inference_result.to_dict()
            for inference_result in self.inference_results
        ]

        return {
            'inferenceResults': inferenceResults
        }
