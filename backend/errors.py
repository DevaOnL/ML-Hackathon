"""Public error descriptions deliberately exclude input values and filesystem paths."""
from __future__ import annotations


class ServiceError(RuntimeError):
    def __init__(self, code: str, message: str, status_code: int = 503):
        super().__init__(message)
        self.code = code
        self.public_message = message
        self.status_code = status_code

    def public(self) -> dict[str, str]:
        return {"code": self.code, "message": self.public_message}
