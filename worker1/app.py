#worker1/app.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2Model
from contextlib import asynccontextmanager
import torch
import os
import time

model = None
layers = None
device = torch.device("cpu")
WORKER_PID = os.getpid()


def load_model():
    global model, layers
    model = GPT2Model.from_pretrained("gpt2")
    model.to(device)
    model.eval()
    NUM_LAYERS = len(model.h)
    layers = model.h[:NUM_LAYERS // 2]
    print(f"\n🔧 [W1-PID:{WORKER_PID}] Model loaded\n")


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
        start_time = time.time()
        print(f"[W1-PID:{WORKER_PID}] ▶️  START: input_len={len(data.input_ids)}")

        input_ids = torch.tensor(data.input_ids).unsqueeze(0).to(device)
        position_ids = torch.arange(input_ids.shape[-1], dtype=torch.long, device=device).unsqueeze(0)

        x = model.wte(input_ids) + model.wpe(position_ids)
        for layer in layers:
            x = layer(x)[0]

        elapsed = time.time() - start_time
        print(f"[W1-PID:{WORKER_PID}] ✅ DONE: {elapsed:.3f}s")

        return {
            "hidden_states": x.detach().cpu().tolist(),
            "worker_pid": WORKER_PID   # ✅ Return PID to master
        }

    except Exception as e:
        print(f"[W1-PID:{WORKER_PID}] ❌ ERROR: {str(e)}")
        return {"error": str(e)}


@app.get("/health")
async def health():
    return {"status": "ok", "worker_pid": WORKER_PID}
