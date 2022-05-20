from flask import jsonify
from flask.wrappers import Request, Response
from api.responses.message_error_response import MessageErrorResponse

from core.services.lisam_net_service import LisamNetService

from api.responses.image_inference.image_inference_response import \
    ImageInferenceResponse


class ImageInferenceController:
    def __init__(self) -> None:
        self.lisam_service = LisamNetService()

    def handle_inference_image(self, r: Request) -> Response:
        if 'image' not in r.files:
            response = MessageErrorResponse('Requires uploading an image')
            return jsonify(response.to_dict_response())

        bytes_img = r.files['image'].read()
        inference_results = self.lisam_service.inference_img_bytes(bytes_img)

        response = ImageInferenceResponse(inference_results)
        return jsonify(response.to_dict_response())
