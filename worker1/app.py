from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2Model
import torch

app = FastAPI()

# ---- Request Schema ----
class InputData(BaseModel):
    input_ids: list

# ---- Global model variables ----
model = None
layers = None
device = torch.device("cpu")

# ---- Load model at startup (IMPORTANT for deployment) ----
@app.on_event("startup")
def load_model():
    global model, layers

    model = GPT2Model.from_pretrained("gpt2")
    model.to(device)
    model.eval()

    # Take first 6 transformer layers (simulating worker partition)
    layers = model.h[:6]


# ---- Inference endpoint ----
@app.post("/process")
def process(data: InputData):
    input_ids = torch.tensor(data.input_ids).to(device)

    # Token embedding
    x = model.wte(input_ids)

    # Pass through assigned layers
    for layer in layers:
        x = layer(x)[0]

    return {
        "hidden_states": x.detach().cpu().tolist()
    }
