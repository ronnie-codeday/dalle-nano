from typing import List, Dict, Union

from fastapi import FastAPI, status, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from transformers import pipeline, set_seed


class Prediction(BaseModel):
    instances: List[Dict] = []

app = FastAPI()

templates = Jinja2Templates(directory='app/templates')

app.mount('/static/', StaticFiles(directory='app/static'), name='static')

@app.get('/', response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse('home.html',
            {'request': request})

@app.post("/prediction")
async def getPred(pred: Prediction):

    from .model_service.dalle import generate_predictions
    # here you can massage `generate_predictions`
    # output to make compatible with response
    
    generate_predictions("old mcdonald")


    return {"predictions": pred.instances}

@app.post("/submit")
async def Predict(request:Request, prompt: str = Form(...)):
    generator = pipeline('text-generation', model='gpt2')
    set_seed(103)
    res = generator(prompt, max_length=50, num_return_sequences=1)
    return res[0]["generated_text"]

@app.get("/health", status_code=200)
async def health_check():
    return {"Everything": "OK!"}
