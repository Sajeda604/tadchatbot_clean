import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from connect_memory_with_llm import build_rag_chain


load_dotenv()


app = FastAPI(title="Software Services Bot API")


try:
    rag_chain = build_rag_chain()
except Exception as e:
    print(f"Error initializing RAG chain: {e}")
    rag_chain = None


class QuestionRequest(BaseModel):
    question: str


@app.post("/chat")
async def chat_endpoint(request: QuestionRequest):
    if not rag_chain:
        raise HTTPException(status_code=500, detail="Bot engine not initialized.")

    try:
        response = rag_chain.invoke({"input": request.question})

        answer = None
        if isinstance(response, dict):
            answer = (
                response.get("answer")
                or response.get("output_text")
                or response.get("text")
            )
        else:
            answer = str(response)

        if not answer:
            answer = "عذراً، لم أستطع العثور على إجابة."

        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
