from typing import Any, Dict


class ApiResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code

    def to_dict_response(self) -> Dict[str, Any]:
        return {}
