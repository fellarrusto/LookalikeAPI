from pydantic import BaseModel
from typing import List


# Responses
class ResponseDTO(BaseModel):
    outcome: bool
    code: str
    description: str

class EmbeddingDTO(BaseModel):
    embedding: List[float]

class ErrorCodes:
    INTERNAL_SERVER_ERROR: str = "internal-server-error"
