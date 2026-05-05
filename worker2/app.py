#worker2/app.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2LMHeadModel
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
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.to(device)
    model.eval()
    NUM_LAYERS = len(model.transformer.h)
    layers = model.transformer.h[NUM_LAYERS // 2:]
    print(f"\n🔧 [W2-PID:{WORKER_PID}] Model loaded\n")


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
        start_time = time.time()
        print(f"[W2-PID:{WORKER_PID}] ▶️  START: hidden_states_seq={len(data.hidden_states)}")

        x = torch.tensor(data.hidden_states).to(device)
        for layer in layers:
            x = layer(x)[0]

        x = model.transformer.ln_f(x)
        logits = model.lm_head(x)

        elapsed = time.time() - start_time
        print(f"[W2-PID:{WORKER_PID}] ✅ DONE: {elapsed:.3f}s")

        return {
            "logits": logits.detach().cpu().tolist(),
            "worker_pid": WORKER_PID   # ✅ Return PID to master
        }

    except Exception as e:
        print(f"[W2-PID:{WORKER_PID}] ❌ ERROR: {str(e)}")
        return {"error": str(e)}


@app.get("/health")
async def health():
    return {"status": "ok", "worker_pid": WORKER_PID}
