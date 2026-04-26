from fastapi import FastAPI
from transformers import GPT2Model
import torch

app = FastAPI()

model = GPT2Model.from_pretrained("gpt2")
layers = model.h[:6]

@app.post("/process")
def process(input_ids: list):
    input_ids = torch.tensor(input_ids)
    x = model.wte(input_ids)
    
    for layer in layers:
        x = layer(x)[0]
    
    return {"hidden_states": x.tolist()}
