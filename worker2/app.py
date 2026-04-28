from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2LMHeadModel
import torch

app = FastAPI()

class InputData(BaseModel):
    hidden_states: list


model = None
layers = None
device = torch.device("cpu")


@app.on_event("startup")
def load_model():
    global model, layers

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.to(device)
    model.eval()

    # Remaining transformer layers
    layers = model.transformer.h[6:]


@app.post("/process")
def process(data: InputData):

    x = torch.tensor(data.hidden_states).to(device)

    for layer in layers:
        x = layer(x)[0]

    logits = model.lm_head(x)

    return {
        "logits": logits.detach().cpu().tolist()
    }
