# src/routes/router.py
from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile, status
from fastapi.responses import JSONResponse

from src.di import get_usecase
from src.schemas.req_in import QueryRequest, TextRequest
from src.usecase import UseCase

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
    use_case: Annotated[UseCase, Depends(get_usecase)],
):
    use_case.process_texts([paragraph.text])


@router.post("/process/docs", status_code=status.HTTP_200_OK)
async def process_docs(
    docs: Annotated[list[UploadFile], File(...)],
    use_case: Annotated[UseCase, Depends(get_usecase)],
):
    print(docs)

    for doc in docs:
        print(doc)
        print(doc.filename)
        print(doc.content_type)

    await use_case.process_documents(docs)


@router.post("/query", status_code=status.HTTP_200_OK, response_model=dict)
async def query_text(
    query: QueryRequest,
    use_case: Annotated[UseCase, Depends(get_usecase)],
):
    return JSONResponse(content={"response": use_case.query(query.query)})
