import io
import re
import logging
from pdf2image import convert_from_bytes
from PIL import Image
import numpy as np
import cv2
import pytesseract

# ——— Logging setup —————————————————————————————
logging.basicConfig(level=logging.DEBUG)

def pdf_to_images(file_bytes: bytes):
    """
    Convert PDF bytes into a list of high-res PIL images.
    """
    return convert_from_bytes(file_bytes, dpi=300)

def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    """
    Grayscale + Otsu threshold to binary for clean OCR.
    """
    gray = np.array(pil_img.convert("L"))
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw

def extract_table_cells(img: np.ndarray):
    """
    1) OCR → find header words “Test”, “Result”, “Unit”, “Range”
    2) Infer column splits from their x-centers
    3) Assign every word into (row,line) & column bucket
    4) Emit list of rows, each a list of cell strings.
    """
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    # Find header words
    header_idxs = [
        i for i, w in enumerate(data['text'])
        if w and w.lower() in ("test", "result", "unit", "range")
    ]
    if not header_idxs:
        logging.debug("No table header detected; returning []")
        return []

    # Compute splits
    xs = sorted(data['left'][i] + data['width'][i] // 2 for i in header_idxs)
    num_cols = len(xs)
    splits = [(xs[i] + xs[i+1]) // 2 for i in range(num_cols-1)]
    logging.debug(f"Header x-centers → {xs}")
    logging.debug(f"Column splits → {splits}")

    # Cluster words by (page, line) & column
    rows = {}
    for i, txt in enumerate(data['text']):
        if not txt.strip():
            continue
        page, line = data['page_num'][i], data['line_num'][i]
        x_center = data['left'][i] + data['width'][i] // 2
        col = min(sum(1 for b in splits if x_center > b), num_cols-1)
        key = (page, line)
        rows.setdefault(key, []).append((col, txt))

    # Build each row’s cells
    table = []
    for key in sorted(rows.keys()):
        cells = [""] * num_cols
        for col, word in rows[key]:
            cells[col] = (cells[col] + " " + word).strip()
        table.append(cells)

    return table

def parse_row(cells):
    """
    Given [name, raw_val, unit, raw_range], returns dict or None.
    Extracts the first two floats from raw_range, ignores trailing text.
    """
    name, raw_val, unit, raw_range = cells

    # Must have a name and a range string
    if not name.strip() or not raw_range.strip():
        logging.debug(f"Skipping incomplete row: {cells}")
        return None

    # Extract numeric test value
    m_val = re.search(r"(\d+\.?\d*)", raw_val)
    if not m_val:
        logging.debug(f"Skipping row without numeric value: {cells}")
        return None
    val = float(m_val.group(1))

    # Extract first two numbers from raw_range
    nums = re.findall(r"\d+\.?\d*", raw_range)
    if len(nums) < 2:
        logging.debug(f"Skipping row with invalid range '{raw_range}': {cells}")
        return None
    lo, hi = float(nums[0]), float(nums[1])

    return {
        "test_name": name.strip(),
        "test_value": str(val),
        "test_unit": unit.strip(),
        "bio_reference_range": f"{lo:.1f}-{hi:.1f}",
        "lab_test_out_of_range": not (lo <= val <= hi)
    }
