#master/app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from transformers import GPT2Tokenizer
import torch
import torch.nn.functional as F
import asyncio
from typing import Dict, Optional
from datetime import datetime
import uuid

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

WORKER1_URL = "http://localhost:8001/process"
WORKER2_URL = "http://localhost:8002/process"

queries_db: Dict[str, dict] = {}
query_queue = asyncio.Queue()


class QueryState:
    def __init__(self, query_id, prompt, max_tokens):
        self.query_id = query_id
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.tokens = tokenizer.encode(prompt)
        self.status = "queued"
        self.current_iteration = 0
        self.result = None
        self.error = None
        self.created_at = datetime.now().isoformat()
        self.hidden_states = None
        self.current_stage = "queued"
        self.last_updated = datetime.now()
        self.worker1_count = 0
        self.worker2_count = 0
        self.worker1_pid = None   # ✅ Store W1 PID
        self.worker2_pid = None   # ✅ Store W2 PID


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 10


@app.post("/generate")
async def generate(request: GenerateRequest):
    query_id = str(uuid.uuid4())[:8]
    queries_db[query_id] = QueryState(
        query_id=query_id,
        prompt=request.prompt,
        max_tokens=request.max_tokens
    )
    await query_queue.put(query_id)
    return {
        "query_id": query_id,
        "status": "queued",
        "message": "Query queued for processing"
    }


@app.get("/status/{query_id}")
async def get_status(query_id: str):
    if query_id not in queries_db:
        return {"error": "Query not found"}
    q = queries_db[query_id]
    return {
        "query_id": query_id,
        "status": q.status,
        "prompt": q.prompt,
        "result": q.result,
        "error": q.error,
        "progress": f"{q.current_iteration}/{q.max_tokens}"
    }


@app.get("/all-queries")
async def get_all_queries():
    return {
        "total": len(queries_db),
        "queries": {
            qid: {
                "status": q.status,
                "prompt": q.prompt,
                "result": q.result,
                "error": q.error,
                "progress": f"{q.current_iteration}/{q.max_tokens}",
                "current_stage": q.current_stage,
                "worker1_pid": q.worker1_pid,   # ✅ Expose PIDs to frontend
                "worker2_pid": q.worker2_pid,
            }
            for qid, q in queries_db.items()
        }
    }


@app.get("/health")
async def health():
    return {"status": "ok", "queued_queries": query_queue.qsize()}


async def call_worker1(query_id: str, query: QueryState):
    try:
        response = await asyncio.to_thread(
            lambda: requests.post(
                WORKER1_URL,
                json={"input_ids": query.tokens},
                timeout=30
            ).json()
        )

        if "error" in response:
            raise Exception(f"Worker1 error: {response['error']}")
        if "hidden_states" not in response:
            raise Exception(f"Worker1 response invalid: {list(response.keys())}")

        # ✅ Log W1 PID in master terminal
        w1_pid = response.get("worker_pid", "unknown")
        print(f"[MASTER] [W1-PID:{w1_pid}] {query_id[:4]} hidden_states received")

        return response["hidden_states"], w1_pid

    except Exception as e:
        print(f"[MASTER] [W1] ERROR for {query_id}: {str(e)}")
        raise


async def call_worker2(query_id: str, query: QueryState):
    try:
        response = await asyncio.to_thread(
            lambda: requests.post(
                WORKER2_URL,
                json={"hidden_states": query.hidden_states},
                timeout=30
            ).json()
        )

        if "error" in response:
            raise Exception(f"Worker2 error: {response['error']}")
        if "logits" not in response:
            raise Exception(f"Worker2 response invalid: {list(response.keys())}")

        # ✅ Log W2 PID in master terminal
        w2_pid = response.get("worker_pid", "unknown")
        print(f"[MASTER] [W2-PID:{w2_pid}] {query_id[:4]} logits received")

        return response["logits"], w2_pid

    except Exception as e:
        print(f"[MASTER] [W2] ERROR for {query_id}: {str(e)}")
        raise


async def process_queries_pipelined():
    pending_worker1: Dict[str, QueryState] = {}
    pending_worker2: Dict[str, QueryState] = {}

    while True:
        try:
            while not query_queue.empty():
                query_id = query_queue.get_nowait()
                query = queries_db[query_id]
                query.status = "queued"
                query.current_stage = "queued"
                pending_worker1[query_id] = query
                print(f"[MASTER] [Scheduler] {query_id[:4]} QUEUED → ready for Worker1")
        except asyncio.QueueEmpty:
            pass

        # WORKER 1 STAGE
        worker1_tasks = []
        for query_id in list(pending_worker1.keys()):
            query = pending_worker1[query_id]
            if query.current_iteration < query.max_tokens:
                query.status = "worker1"
                query.current_stage = "worker1"
                query.last_updated = datetime.now()
                query.worker1_count += 1

                task = asyncio.create_task(call_worker1(query_id, query))
                worker1_tasks.append((query_id, task))
                print(f"[MASTER] [W1] {query_id[:4]} started (iteration {query.current_iteration + 1}/{query.max_tokens})")

        if worker1_tasks:
            for query_id, task in worker1_tasks:
                try:
                    hidden_states, w1_pid = await asyncio.wait_for(task, timeout=30)
                    query = queries_db[query_id]
                    query.hidden_states = hidden_states
                    query.worker1_pid = w1_pid          # ✅ Save PID
                    query.status = "returned"
                    query.current_stage = "returned"
                    query.last_updated = datetime.now()

                    del pending_worker1[query_id]
                    pending_worker2[query_id] = query
                    print(f"[MASTER] [W1-PID:{w1_pid}] {query_id[:4]} completed → sending to Worker2")

                except Exception as e:
                    query = queries_db[query_id]
                    query.error = str(e)
                    query.status = "error"
                    query.current_stage = "error"
                    if query_id in pending_worker1:
                        del pending_worker1[query_id]
                    print(f"[MASTER] [W1] {query_id[:4]} ERROR: {e}")

        # WORKER 2 STAGE
        worker2_tasks = []
        for query_id in list(pending_worker2.keys()):
            query = pending_worker2[query_id]
            query.status = "worker2"
            query.current_stage = "worker2"
            query.last_updated = datetime.now()
            query.worker2_count += 1

            task = asyncio.create_task(call_worker2(query_id, query))
            worker2_tasks.append((query_id, task))
            print(f"[MASTER] [W2] {query_id[:4]} started (iteration {query.current_iteration + 1}/{query.max_tokens})")

        if worker2_tasks:
            for query_id, task in worker2_tasks:
                try:
                    logits, w2_pid = await asyncio.wait_for(task, timeout=30)
                    query = queries_db[query_id]
                    query.worker2_pid = w2_pid          # ✅ Save PID

                    tokens = query.tokens
                    logits_tensor = torch.tensor(logits)
                    logits_last = logits_tensor[0, -1]

                    for token_id in set(tokens):
                        if logits_last[token_id] > 0:
                            logits_last[token_id] /= 2.5
                        else:
                            logits_last[token_id] *= 2.5

                    temperature = 0.9
                    logits_last = logits_last / temperature
                    values, indices = torch.topk(logits_last, 50)
                    probs = F.softmax(values, dim=-1)
                    next_token = indices[torch.multinomial(probs, 1)].item()

                    tokens.append(next_token)
                    query.tokens = tokens
                    query.current_iteration += 1
                    query.last_updated = datetime.now()

                    if query.current_iteration >= query.max_tokens:
                        query.result = tokenizer.decode(tokens)
                        query.status = "completed"
                        query.current_stage = "completed"
                        if query_id in pending_worker2:
                            del pending_worker2[query_id]
                        print(f"[MASTER] [W2-PID:{w2_pid}] {query_id[:4]} ✅ COMPLETED")
                    else:
                        if query_id in pending_worker2:
                            del pending_worker2[query_id]
                        query.status = "worker1"
                        query.current_stage = "worker1"
                        pending_worker1[query_id] = query
                        print(f"[MASTER] [W2-PID:{w2_pid}] {query_id[:4]} → back to Worker1 (iteration {query.current_iteration + 1}/{query.max_tokens})")

                except Exception as e:
                    query = queries_db[query_id]
                    query.error = str(e)
                    query.status = "error"
                    query.current_stage = "error"
                    if query_id in pending_worker2:
                        del pending_worker2[query_id]
                    print(f"[MASTER] [W2] {query_id[:4]} ERROR: {e}")

        await asyncio.sleep(0.05)


@app.on_event("startup")
async def startup_event():
    print("[MASTER] Starting pipelined query scheduler...")
    asyncio.create_task(process_queries_pipelined())
    print("[MASTER] Scheduler started!")
