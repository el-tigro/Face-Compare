import uvicorn

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

import insightface
from insightface.app import FaceAnalysis

from pydantic import BaseModel
from load_data import load_hashedFilename, convert_data
from face_compare import extract_face_compare_features

import logging
import warnings

warnings.filterwarnings("ignore")


for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# CONFIG_PATH = "config.yaml"


class LoanIdRequest(BaseModel):
    company_name: str
    loan_id: int


app = FastAPI()

# Loading the Face Detection and Comparison Model
model = FaceAnalysis()
model.prepare(ctx_id=0, det_size=(640, 640))


@app.post("/face_compare")
def get_face_compare(
    request: LoanIdRequest
):
    """Pipeline for retrieving photo IDs from MySQL and calculating face similarity and recognition metrics"""
    logger.info(
        f"Starting request for loan_id: {request.loan_id}, "
        f"company: {request.company_name} in MySQL database"
    )
    df = load_hashedFilename(request.company_name, request.loan_id)

    dict_face_compare_features = extract_face_compare_features(request.company_name, df, model)
    logger.info(f"Created feature dictionary, photos count: {len(dict_face_compare_features)}")

    dict_face_compare_features_convert = convert_data(dict_face_compare_features)
    logger.info(f"Converted data types for JSON encoder")

    return JSONResponse(content=jsonable_encoder(dict_face_compare_features_convert))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8599)
