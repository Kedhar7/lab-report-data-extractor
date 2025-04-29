--- main.py
+++ main.py
@@ -1,7 +1,8 @@
-from fastapi import FastAPI, File, UploadFile, HTTPException
-from fastapi.responses import JSONResponse
+from fastapi import FastAPI, File, UploadFile, HTTPException
+from fastapi.responses import JSONResponse, RedirectResponse
 from pydantic import BaseModel
 from typing import List
 from PIL import Image
+import io
 
 from utils import pdf_to_images, preprocess_image, extract_table_cells, parse_row
 
@@
-app = FastAPI()
+app = FastAPI()
+
+# Redirect “/” → Swagger UI at /docs
+@app.get("/", include_in_schema=False)
+async def redirect_to_docs():
+    return RedirectResponse(url="/docs")
 
 @app.get("/")
 async def root():
@@ -12,14 +15,6 @@
     return {"message": "Lab-report OCR API is running"}
 
-class LabTest(BaseModel):
-    test_name: str
-    test_value: str
-    bio_reference_range: str
-    test_unit: str
-    lab_test_out_of_range: bool
-
-class ResponseModel(BaseModel):
-    is_success: bool
-    data: List[LabTest]
+class LabTest(BaseModel):
+    test_name: str
+    test_value: str
+    bio_reference_range: str
+    test_unit: str
+    lab_test_out_of_range: bool
+
+class ResponseModel(BaseModel):
+    is_success: bool
+    data: List[LabTest]
 
 @app.post("/get-lab-tests", response_model=ResponseModel)
 async def get_lab_tests(file: UploadFile = File(...)):
