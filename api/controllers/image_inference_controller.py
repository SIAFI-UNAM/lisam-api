from typing import Tuple
from flask import jsonify
from flask.wrappers import Request, Response
from http import HTTPStatus

from core.services.lisam_net_service import LisamNetService

from api.responses.message_error_response import MessageErrorResponse
from api.responses.image_inference.image_inference_response import \
    ImageInferenceResponse


class ImageInferenceController:
    def __init__(self) -> None:
        self.lisam_service = LisamNetService()

    def handle_inference_image(self, r: Request) -> Tuple[Response, int]:
        if 'image' not in r.files:
            response = MessageErrorResponse('Requires uploading an image')
            return jsonify(response.to_dict_response()), HTTPStatus.BAD_REQUEST

        bytes_img = r.files['image'].read()
        is_image_empty = len(bytes_img) == 0
        if is_image_empty:
            response = MessageErrorResponse('Image has no content')
            return jsonify(response.to_dict_response()), HTTPStatus.BAD_REQUEST

        inference_results = self.lisam_service.inference_img_bytes(bytes_img)

        response = ImageInferenceResponse(inference_results)
        return jsonify(response.to_dict_response()), HTTPStatus.OK
