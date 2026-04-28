from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2Model
import torch

app = FastAPI()

class InputData(BaseModel):
    input_ids: list


model = None
layers = None
device = torch.device("cpu")


@app.on_event("startup")
def load_model():
    global model, layers

    model = GPT2Model.from_pretrained("gpt2")
    model.to(device)
    model.eval()

    # First 6 transformer layers
    layers = model.h[:6]


@app.post("/process")
def process(data: InputData):

    input_ids = torch.tensor(data.input_ids).to(device)

    # embeddings
    x = model.wte(input_ids)

    # first half layers
    for layer in layers:
        x = layer(x)[0]

    return {
        "hidden_states": x.detach().cpu().tolist()
    }
