from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from PIL import Image
import io

from utils import pdf_to_images, preprocess_image, extract_table_cells, parse_row

app = FastAPI(title="Lab-Report OCR API", version="0.2.0")


@app.get("/")
async def root():
    return {"message": "Lab-report OCR API is running"}


class LabTest(BaseModel):
    test_name: str
    test_value: str
    bio_reference_range: str
    test_unit: str
    lab_test_out_of_range: bool


class ResponseModel(BaseModel):
    is_success: bool
    data: List[LabTest]


@app.post("/get-lab-tests", response_model=ResponseModel)
async def get_lab_tests(file: UploadFile = File(...)):
    # 1) Validate upload type
    if not (file.content_type.startswith("image/") or file.content_type == "application/pdf"):
        raise HTTPException(400, "Unsupported file type: upload PNG/JPG or PDF")

    contents = await file.read()

    # 2) Load into PIL images
    if file.content_type == "application/pdf":
        pil_images = pdf_to_images(contents)
    else:
        pil_images = [Image.open(io.BytesIO(contents))]

    # 3) OCR & parse each image
    tests = []
    for pil in pil_images:
        bw = preprocess_image(pil)
        table = extract_table_cells(bw)
        for row in table:
            parsed = parse_row(row)
            if parsed:
                tests.append(parsed)

    # 4) Return structured JSON
    return JSONResponse(status_code=200, content={"is_success": True, "data": tests})
