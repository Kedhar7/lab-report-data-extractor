# utils.py

import re, logging
from typing import List, Optional
import numpy as np
import cv2
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# If some reports have a 5th “Remark” column, list it here
HEADER_LABELS = ["test", "result", "unit", "range", "remark"]

def pdf_to_images(file_bytes: bytes) -> List[Image.Image]:
    return convert_from_bytes(file_bytes, dpi=300)

def preprocess_image(pil: Image.Image) -> np.ndarray:
    gray = np.array(pil.convert("L"))
    _, bw = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw

def extract_table_cells(img: np.ndarray) -> List[List[str]]:
    """
    Returns a list of rows, each a list of N cell‐strings.
    Attempts four strategies for column splits:
      1) Header OCR → exact x‐centers
      2) KMeans on all word‐centers
      3) Grid‐line detection via morphology
      4) Equal‐width fallback
    If *all* that fails, falls back to regex scanning in parse_fallback().
    """
    # 1) OCR → get data dict
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    n = len(data['text'])
    # group word‐indices by line
    lines = {}
    for i, txt in enumerate(data['text']):
        if not txt.strip(): continue
        ln = data['line_num'][i]
        lines.setdefault(ln, []).append(i)

    # 2) Look for a “header” line containing at least 4 of our labels
    header_ln = None
    for ln, idxs in lines.items():
        words = [data['text'][i].lower() for i in idxs]
        found = sum(1 for lbl in HEADER_LABELS if any(lbl in w for w in words))
        if found >= 4:
            header_ln = ln
            break

    splits = []
    if header_ln is not None:
        xs = []
        for lbl in HEADER_LABELS:
            for i in lines[header_ln]:
                if lbl in data['text'][i].lower():
                    xc = data['left'][i] + data['width'][i]//2
                    xs.append(xc)
                    break
        xs = sorted(xs)
        splits = [(xs[i]+xs[i+1])//2 for i in range(len(xs)-1)]
        logger.debug(f"[header] splits → {splits}")

    # 3) KMeans fallback
    if not splits:
        xcenters = [
            data['left'][i] + data['width'][i]//2
            for i in range(n) if data['text'][i].strip()
        ]
        if len(xcenters) >= 4:
            try:
                k = min(5, max(2, len(xcenters)//15))
                km = KMeans(n_clusters=k, random_state=0).fit(
                    np.array(xcenters).reshape(-1,1)
                )
                centers = sorted(int(c[0]) for c in km.cluster_centers_)
                splits = [(centers[i]+centers[i+1])//2 for i in range(len(centers)-1)]
                logger.debug(f"[kmeans] splits → {splits}")
            except Exception as e:
                logger.debug(f"KMeans failed: {e}")

    # 4) Grid‐line detection fallback
    def detect_grid_splits(bin_img):
        inv = 255 - bin_img
        # a tall, thin kernel to pick vertical lines
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (1, bin_img.shape[0]//40))
        vert = cv2.erode(inv, kern, iterations=1)
        vert = cv2.dilate(vert, kern, iterations=2)
        cnts,_ = cv2.findContours(vert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        xs = []
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            if h > bin_img.shape[0]//2:
                xs.append(x + w//2)
        return sorted(xs)

    if not splits:
        vxs = detect_grid_splits(img)
        if len(vxs) >= 2:
            # take the mid‐points
            splits = [(vxs[i] + vxs[i+1])//2 for i in range(len(vxs)-1)]
            logger.debug(f"[grid ] splits → {splits}")

    # 5) Equal‐width fallback
    if not splits:
        w = img.shape[1]
        splits = [w//4, w//2, 3*w//4]
        logger.debug(f"[equal] splits → {splits}")

    # Now bucket every word into (page,line) + column
    num_cols = len(splits)+1
    rows = {}
    for i, txt in enumerate(data['text']):
        t = txt.strip()
        if not t: continue
        pg, ln = data['page_num'][i], data['line_num'][i]
        xc = data['left'][i] + data['width'][i]//2
        col = min(sum(1 for s in splits if xc > s), num_cols-1)
        key = (pg, ln)
        rows.setdefault(key, []).append((col, t))

    table = []
    for key in sorted(rows):
        cells = [""]*num_cols
        for col, w in rows[key]:
            cells[col] = (cells[col] + " " + w).strip()
        table.append(cells)

    # If still nothing, let parse_fallback do row‐by‐row regex
    if not table or all(len(r)<4 for r in table):
        return parse_fallback(img)

    return table

def parse_fallback(img: np.ndarray) -> List[List[str]]:
    """
    Last‐resort: OCR entire page as text, split lines by whitespace,
    look for lines like: Name  Value  Unit  Low-High
    """
    txt = pytesseract.image_to_string(img)
    out = []
    for line in txt.splitlines():
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        # look for the “low-high” at the end
        m = re.search(r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)$", line)
        if not m:
            continue
        rng = m.group(0)
        # assume the token before rng is the unit, and before that the value
        unit = parts[-2]
        val  = parts[-3]
        name = " ".join(parts[:-3])
        out.append([name, val, unit, rng])
    logger.debug(f"[fallback regex] extracted {len(out)} rows")
    return out

def parse_row(cells: List[str]) -> Optional[dict]:
    # assume at least [name, raw_val, unit, raw_range, ...]
    if len(cells) < 4:
        return None
    name, raw_val, unit, raw_range = [cells[i].strip() for i in range(4)]
    if not name or not raw_range:
        return None

    # numeric test_value
    m = re.search(r"(\d+\.?\d*)", raw_val)
    if not m:
        return None
    val = float(m.group(1))

    # numeric range
    nums = re.findall(r"\d+\.?\d*", raw_range)
    if len(nums) < 2:
        return None
    lo, hi = float(nums[0]), float(nums[1])

    return {
        "test_name": name,
        "test_value": str(val),
        "test_unit": unit,
        "bio_reference_range": f"{lo:.1f}-{hi:.1f}",
        "lab_test_out_of_range": not (lo <= val <= hi)
    }
