# main.py
from fastapi import FastAPI
from pydantic import BaseModel

from movie_recommend import main_chain

app = FastAPI(title="Neo4j Movie GraphRAG API")

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/ask", response_model=AskResponse)
def ask_movie(req: AskRequest):
    answer = main_chain(req.question)
    return {"answer": answer}
