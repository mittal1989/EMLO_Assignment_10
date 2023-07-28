import requests
import torch
import io

import numpy as np
from torch.nn import functional as F
from typing import Annotated
from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image
# from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel
import torchvision.transforms as T
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load model
model = torch.jit.load("./Models/model.script.vit.pt")
model = model.eval()

# transforms
transforms = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])

@app.get("/infer")
async def infer(image: Annotated[bytes, File()]):
    img: Image.Image = Image.open(io.BytesIO(image))
    # img: Image.Image = Image.open(image)
    img = img.convert("L")
    img = img.resize((28, 28))
    print(img.size)

    img_t = transforms(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(img_t)
    preds = F.softmax(logits, dim=1).squeeze(0).tolist()

    return {str(i): preds[i] for i in range(10)}

@app.get("/health")
async def health():
    return {"message": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)