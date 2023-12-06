from typing import List
from io import BytesIO

from fastapi import APIRouter, status, UploadFile, File, Form

from .process import proc as processing

routes = APIRouter(prefix="/knowledge-base", tags=["knowledge-base"])

### Payloads
from pydantic import BaseModel, Field

class PayloadUploadFiles(BaseModel):
    tenant_id: str = Field(example="Telkom")
    files: List[UploadFile] = Field(example=["file1.pdf", "file2.pdf"])

@routes.post(
        path="/upload-file",
)
async def upload_file(files: List[UploadFile], tenant_id: str = Form()):
    for f in files:
        content = await f.read()
        pdf = BytesIO(content)
        if processing(pdf, tenant_id):
            result = {
                "status": status.HTTP_201_CREATED,
                "filename": f.filename
            }
            return result
