# src/routes/router.py
from typing import Annotated

from fastapi import APIRouter, Depends, UploadFile, status

from src.di import get_vector_service
from src.schemas.req_in import QueryRequest, TextRequest
from src.services.vector_services import VectorService

router = APIRouter(
    prefix="/api/v1",
    tags=["api"],
)


@router.get("/")
async def hello():
    return {"message": "Hello from the API!"}


@router.post("/process/text", status_code=status.HTTP_200_OK)
async def process_text(
    paragraph: TextRequest,
    vector_service: Annotated[VectorService, Depends(get_vector_service)],
):
    text = paragraph.text
    await vector_service.store(text)
    return {"message": "Text processed and stored successfully"}


@router.post("/process/doc", status_code=status.HTTP_200_OK)
async def process_doc(
    doc: UploadFile,
    vector_service: Annotated[VectorService, Depends(get_vector_service)],
):
    await vector_service.store(doc)
    return {"message": "Document processed and stored successfully"}


@router.post("/process/docs", status_code=status.HTTP_200_OK)
async def process_docs(
    docs: list[UploadFile],
    vector_service: Annotated[VectorService, Depends(get_vector_service)],
):
    for doc in docs:
        await vector_service.store(doc)
    return {"message": "Documents processed and stored successfully"}


@router.post("/query", status_code=status.HTTP_200_OK)
async def query_text(
    query: QueryRequest,
    vector_service: Annotated[VectorService, Depends(get_vector_service)],
):
    query_text = query.query
    results = vector_service.search(query_text)
    nearest_texts = [point.payload["text"] for point in results.points]  # type: ignore
    return {"results": nearest_texts}
