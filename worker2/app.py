from fastapi import FastAPI
from transformers import GPT2LMHeadModel
import torch

app = FastAPI()

model = GPT2LMHeadModel.from_pretrained("gpt2")
layers = model.transformer.h[6:]

@app.post("/process")
def process(hidden_states: list):
    x = torch.tensor(hidden_states)
    
    for layer in layers:
        x = layer(x)[0]
    
    logits = model.lm_head(x)
    return {"logits": logits.tolist()}
