from flask.wrappers import Request
from http import HTTPStatus
from api.response_handler import response_handler
from api.responses.api_response import ApiResponse

from core.services.lisam_net_service import LisamNetService

from api.responses.message_error_response import MessageErrorResponse
from api.responses.image_inference.image_inference_response import \
    ImageInferenceResponse


class ImageInferenceController:
    def __init__(self) -> None:
        self.lisam_service = LisamNetService()

    @response_handler
    def handle_inference_image(self, r: Request) -> ApiResponse:
        if 'image' not in r.files:
            return MessageErrorResponse(
                'Requires uploading an image',
                HTTPStatus.BAD_REQUEST)

        bytes_img = r.files['image'].read()
        is_image_empty = len(bytes_img) == 0
        if is_image_empty:
            return MessageErrorResponse(
                'Image has no content',
                HTTPStatus.BAD_REQUEST)

        inference_results = self.lisam_service.inference_img_bytes(bytes_img)

        return ImageInferenceResponse(inference_results, HTTPStatus.OK)
