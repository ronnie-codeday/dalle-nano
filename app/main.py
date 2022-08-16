from typing import List, Dict, Union

from fastapi import FastAPI, status, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import wandb

from transformers import pipeline, set_seed

from .d_mini import d_mini
import base64


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
    prompt = pred.instances[0]['prompt']
    d_mini.generate_image(
        is_mega=False,
        text=prompt,
        seed=-1,
        grid_size=1,
        top_k=256,
        image_path='generated',
        models_root='pretrained',
        fp16=False,
    )

    with open('/static/images/generated.png', mode='rb') as file:
        img = file.read()
    img = base64.encodebytes(img).decode('utf-8')
    return {"predictions": img} 

@app.post("/submit")
async def Predict(request:Request, prompt: str = Form(...)):
    d_mini.generate_image(
            is_mega=False,
            text=prompt,
            seed=-1,
            grid_size=2,
            top_k=256,
            image_path='generated',
            models_root='pretrained',
            fp16=False,
        )

@app.get("/health", status_code=200)
async def health_check():
    return {"Everything": "OK!"}
