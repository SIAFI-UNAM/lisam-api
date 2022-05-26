from typing import Callable, Tuple
from flask import jsonify

from flask.wrappers import Response
from api.responses.api_response import ApiResponse


def response_handler(f: Callable[..., ApiResponse]):
    def to_json_status_code(*args) -> Tuple[Response, int]:
        response = f(*args)
        return jsonify(response.to_dict_response()), response.status_code

    return to_json_status_code
