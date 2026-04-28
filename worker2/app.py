from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2LMHeadModel
from contextlib import asynccontextmanager
import torch

model = None
layers = None
device = torch.device("cpu")


def load_model():
    global model, layers
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.to(device)
    model.eval()
    NUM_LAYERS = len(model.transformer.h)
    layers = model.transformer.h[NUM_LAYERS // 2:]  # Last 6


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(lifespan=lifespan)


class InputData(BaseModel):
    hidden_states: list


@app.post("/process")
def process(data: InputData):
    try:
        x = torch.tensor(data.hidden_states).to(device)

        for layer in layers:
            x = layer(x)[0]

        # Final Layer Norm
        x = model.transformer.ln_f(x)
        logits = model.lm_head(x)

        return {"logits": logits.detach().cpu().tolist()}

    except Exception as e:
        return {"error": str(e)}
