from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2Model
from contextlib import asynccontextmanager
import torch

model = None
layers = None
device = torch.device("cpu")


def load_model():
    global model, layers
    model = GPT2Model.from_pretrained("gpt2")
    model.to(device)
    model.eval()
    NUM_LAYERS = len(model.h)  # 12 for GPT-2
    layers = model.h[:NUM_LAYERS // 2]  # First 6


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(lifespan=lifespan)


class InputData(BaseModel):
    input_ids: list


@app.post("/process")
def process(data: InputData):
    try:
        input_ids = torch.tensor(data.input_ids).unsqueeze(0).to(device)

        # Word + Position Embeddings
        position_ids = torch.arange(input_ids.shape[-1], dtype=torch.long, device=device)
        x = model.wte(input_ids) + model.wpe(position_ids)

        # First half transformer layers
        for layer in layers:
            x = layer(x)[0]

        return {"hidden_states": x.detach().cpu().tolist()}

    except Exception as e:
        return {"error": str(e)}
