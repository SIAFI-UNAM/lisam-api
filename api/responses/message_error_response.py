from typing import Any, Dict
from api.responses.api_response import ApiResponse


class MessageErrorResponse(ApiResponse):
    def __init__(self, error_message: str) -> None:
        super().__init__()

        self.error_message = error_message

    def to_dict_response(self) -> Dict[str, Any]:
        return {
            'error': self.error_message
        }
