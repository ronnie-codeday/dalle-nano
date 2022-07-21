from typing import List, Dict, Union

from fastapi import FastAPI, status
from pydantic import BaseModel

class Prediction(BaseModel):
    instances: List[Dict] = []

app = FastAPI()

@app.post("/prediction")
async def getPred(pred: Prediction):
    return {"predictions": pred.instances}

@app.get("/health", status_code=200)
async def health_check():
    return {"Everything": "OK!"}
