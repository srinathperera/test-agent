"""Simple LangChain agent backed by OpenRouter (free models), served at POST /chat on port 8000.

Request/response follow the standard OpenAI-style chat completion shape.
"""

import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()
logger = logging.getLogger("agent")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# "openrouter/free" routes to an available free model; set a specific ":free" model id to pin one.
MODEL_NAME = os.getenv("OPENROUTER_MODEL", "openrouter/free")
API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is not set. Add it to your environment or a .env file.")


@tool
def get_current_time() -> str:
    """Return the current UTC date and time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=API_KEY,
    base_url=OPENROUTER_BASE_URL,
    temperature=0.2,
    default_headers={"X-Title": "LangChain Agent"},  # optional; shown in your OpenRouter dashboard
)
agent = create_agent(
    model=llm,
    tools=[get_current_time],
    system_prompt="You are a helpful assistant. Answer clearly and concisely.",
)


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., max_length=32_000)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1, max_length=100)
    model: str | None = None  # accepted for client compatibility; server uses OPENROUTER_MODEL


app = FastAPI(title="OpenRouter LangChain Agent")


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        result = await agent.ainvoke({"messages": [m.model_dump() for m in req.messages]})
    except Exception as exc:  # don't leak internals to clients; log them server-side
        logger.exception("Model call failed")
        raise HTTPException(status_code=502, detail="Upstream model error") from exc

    answer = result["messages"][-1].text
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
