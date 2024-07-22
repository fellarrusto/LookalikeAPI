from functools import wraps
from app.dto.api_models import EmbeddingDTO, ErrorCodes, ResponseDTO
from fastapi import HTTPException, FastAPI, UploadFile, File
from app.modules.lookalike import Lookalike
from io import BytesIO
from PIL import Image
import numpy as np

app = FastAPI()

app.openapi_version = "3.0.1"

model = Lookalike()

def error_handle(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            response_dto = ResponseDTO(outcome=False, code=ErrorCodes.INTERNAL_SERVER_ERROR, description=str(e))
            raise HTTPException(status_code=500, detail=response_dto.dict())
    return wrapper

@app.post("/embed", responses = {200: {"model": EmbeddingDTO}, 500: {"model": ResponseDTO}})
@error_handle
async def embed(file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read()))
    embedding = model.embed(image)
    return EmbeddingDTO(embedding=embedding.tolist())
