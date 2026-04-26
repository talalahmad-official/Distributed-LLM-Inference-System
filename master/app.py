from fastapi import FastAPI
import requests
from transformers import GPT2Tokenizer

app = FastAPI()

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

WORKER1_URL = "https://worker1.onrender.com/process"
WORKER2_URL = "https://worker2.onrender.com/process"

@app.get("/generate")
def generate(prompt: str):
    tokens = tokenizer.encode(prompt)
    
    # Worker 1
    res1 = requests.post(WORKER1_URL, json=tokens).json()
    
    # Worker 2
    res2 = requests.post(WORKER2_URL, json=res1["hidden_states"]).json()
    
    return {"logits": res2["logits"]}
