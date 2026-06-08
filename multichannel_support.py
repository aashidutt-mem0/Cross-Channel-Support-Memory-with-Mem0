"""
Multi-Channel Support Agent with Shared Mem0 Memory
----------------------------------------------------
Three channel endpoints - POST /call, POST /email, POST /chat -
all reading and writing to the same customer email in Mem0.

Install:
    python -m pip install -r requirements.txt

Run:
    Fill in .env, then:
    uvicorn multichannel_support:app --reload

Try the walkthrough:
    python3 test_walkthrough.py
"""

import os
import re
from typing import Optional

from fastapi import FastAPI, HTTPException
from mem0 import MemoryClient
from openai import AzureOpenAI, OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

load_dotenv()

mem0 = MemoryClient(api_key=os.environ["MEM0_API_KEY"])

OPENAI_PROVIDER = os.getenv("OPENAI_PROVIDER", "azure" if os.getenv("AZURE_OPENAI_ENDPOINT") else "openai").lower()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", OPENAI_MODEL)
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

if OPENAI_PROVIDER == "azure":
    llm = AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=AZURE_OPENAI_API_VERSION,
    )
    CHAT_MODEL = AZURE_OPENAI_DEPLOYMENT
else:
    llm = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    CHAT_MODEL = OPENAI_MODEL

# Agent IDs - one per channel so you can tune system prompts independently
# while still sharing the same user_id memory store
PHONE_AGENT_ID = "phone_agent"
EMAIL_AGENT_ID = "email_agent"
CHAT_AGENT_ID  = "chat_agent"

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class CallRequest(BaseModel):
    customer_email: Optional[str] = None
    user_id: Optional[str] = None
    transcript: str          # voice transcript passed as plain text
    run_id: Optional[str] = None  # optional session ID for this call

class EmailRequest(BaseModel):
    customer_email: Optional[str] = None
    user_id: Optional[str] = None
    subject: str
    body: str
    run_id: Optional[str] = None

class ChatRequest(BaseModel):
    customer_email: Optional[str] = None
    user_id: Optional[str] = None
    message: str
    run_id: Optional[str] = None  # chat session ID - pass the same one per session

class MemoryResponse(BaseModel):
    status: str
    memories_stored: int
    message: str

class ChatResponse(BaseModel):
    reply: str
    context_used: list[str]   # memories injected into this turn

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def get_memory_user_id(customer_email: Optional[str] = None, user_id: Optional[str] = None) -> str:
    """
    Mem0 calls this field user_id. In this demo, the customer's email is the
    stable identity that all channels share.
    """
    identity = customer_email or user_id
    if not identity or not identity.strip():
        raise HTTPException(status_code=400, detail="customer_email is required")
    return identity.strip().lower()


def extract_email(text: str) -> Optional[str]:
    match = re.search(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", text)
    return match.group(0).lower() if match else None


def resolve_customer_email(
    *,
    provided_email: Optional[str] = None,
    user_id: Optional[str] = None,
    text: str = "",
) -> Optional[str]:
    if provided_email and provided_email.strip():
        return provided_email.strip().lower()
    if user_id and user_id.strip():
        return user_id.strip().lower()
    return extract_email(text)


def require_customer_email(
    *,
    provided_email: Optional[str] = None,
    user_id: Optional[str] = None,
    text: str = "",
) -> str:
    customer_email = resolve_customer_email(
        provided_email=provided_email,
        user_id=user_id,
        text=text,
    )
    if not customer_email:
        raise HTTPException(status_code=400, detail="customer email not found in message")
    return customer_email


def store_memories(
    messages: list[dict],
    user_id: str,
    agent_id: str,
    run_id: Optional[str],
    channel: str,
) -> int:
    """
    Write messages to Mem0 under the shared user_id.
    - user_id  : cross-channel scope - facts here survive into every channel
    - agent_id : channel-agent scope - lets you inspect per-channel adds
    - run_id   : session scope       - optional, useful for session cleanup
    - metadata : channel tag         - queryable for analytics
    """
    kwargs = dict(
        user_id=user_id,
        agent_id=agent_id,
        metadata={"channel": channel},
    )
    if run_id:
        kwargs["run_id"] = run_id

    result = mem0.add(messages, **kwargs)
    # v1.1 API returns {"results": [...]} or {"id": ..., ...}
    results = result.get("results", [])
    return len(results)


def retrieve_memories(user_id: str, query: str, limit: int = 6) -> list[str]:
    """
    Retrieve cross-channel memories scoped to this user.
    search() requires entity params inside filters={} not as top-level kwargs.
    """
    result = mem0.search(
        query,
        filters={"user_id": user_id},
        top_k=limit,
    )
    results = result.get("results", [])
    return [r["memory"] for r in results if "memory" in r]


def build_support_prompt(channel: str, memories: list[str]) -> str:
    """Assemble a system prompt with injected cross-channel memory context."""
    memory_block = (
        "\n".join(f"- {m}" for m in memories)
        if memories
        else "No prior history for this customer."
    )
    return f"""You are a helpful B2C customer support agent responding via {channel}.

Customer history across all channels:
{memory_block}

Guidelines:
- If you recognise the customer from prior interactions, acknowledge it naturally.
- In an ongoing chat, greet the customer only once. Do not start every reply with repeated salutations.
- Never ask a customer to repeat information already in their history.
- Do not ask for a ticket number.
- Be concise. Resolve or clearly escalate. Do not pad responses.
- If the issue is new, ask only the one most useful clarifying question."""


def build_email_reply_prompt(memories: list[str]) -> str:
    memory_block = (
        "\n".join(f"- {m}" for m in memories)
        if memories
        else "No prior history for this customer."
    )
    return f"""You are a concise B2C customer support agent writing an email reply.

Customer history across all channels:
{memory_block}

Write a professional reply that acknowledges known context, avoids asking for repeated information, gives the next clear step, and does not ask for a ticket number."""


def generate_support_reply(system_prompt: str, user_message: str) -> str:
    """Generate the support response with OpenAI or Azure OpenAI."""
    response = llm.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content or ""

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Multi-Channel Support Agent",
    description="Phone, email, and chat — one shared Mem0 memory store per user.",
    version="1.0.0",
)


@app.get("/")
def root():
    return {"status": "ok", "channels": ["/call", "/email", "/chat"]}


# ---------------------------------------------------------------------------
# POST /call  — ingest a voice transcript
# ---------------------------------------------------------------------------

@app.post("/call", response_model=MemoryResponse)
def handle_call(req: CallRequest):
    """
    Receives a voice call transcript (as plain text) and stores extracted
    facts under the user's cross-channel memory.

    In production, connect your telephony provider (Twilio, Vonage, Pipecat)
    to POST the transcript here after the call ends.
    """
    if not req.transcript.strip():
        raise HTTPException(status_code=400, detail="transcript cannot be empty")

    messages = [{"role": "user", "content": req.transcript}]
    customer_email = require_customer_email(
        provided_email=req.customer_email,
        user_id=req.user_id,
        text=req.transcript,
    )

    stored = store_memories(
        messages=messages,
        user_id=customer_email,
        agent_id=PHONE_AGENT_ID,
        run_id=req.run_id,
        channel="phone",
    )

    return MemoryResponse(
        status="ok",
        memories_stored=stored,
        message=f"Call transcript processed for {customer_email}.",
    )


# ---------------------------------------------------------------------------
# POST /email  — ingest an inbound support email
# ---------------------------------------------------------------------------

@app.post("/email", response_model=MemoryResponse)
def handle_email(req: EmailRequest):
    """
    Receives an inbound support email (subject + body) and stores extracted
    facts under the user's cross-channel memory.

    In production, connect via SendGrid Inbound Parse, Postmark webhooks,
    or any SMTP-to-HTTP bridge to POST here on new mail.
    """
    if not req.body.strip():
        raise HTTPException(status_code=400, detail="email body cannot be empty")

    # Combine subject and body so Mem0 extracts intent from both
    combined = f"Subject: {req.subject}\n\n{req.body}"
    messages = [{"role": "user", "content": combined}]
    customer_email = require_customer_email(
        provided_email=req.customer_email,
        user_id=req.user_id,
        text=combined,
    )

    stored = store_memories(
        messages=messages,
        user_id=customer_email,
        agent_id=EMAIL_AGENT_ID,
        run_id=req.run_id,
        channel="email",
    )

    return MemoryResponse(
        status="ok",
        memories_stored=stored,
        message=f"Email processed for {customer_email}.",
    )


# ---------------------------------------------------------------------------
# POST /chat  — real-time chat turn with memory-augmented response
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
def handle_chat(req: ChatRequest):
    """
    Real-time chat turn.
    1. Retrieve cross-channel memories for this user.
    2. Build a memory-augmented system prompt.
    3. Call OpenAI or Azure OpenAI for a response.
    4. Store new facts from this turn (user message + assistant reply).
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    customer_email = resolve_customer_email(
        provided_email=req.customer_email,
        user_id=req.user_id,
        text=req.message,
    )
    if not customer_email:
        reply = generate_support_reply(
            system_prompt=(
                "You are a concise B2C customer support chatbot. "
                "You cannot access account history until the customer provides their email address. "
                "Ask for their account email in one short sentence."
            ),
            user_message=req.message,
        )
        return ChatResponse(reply=reply, context_used=[])

    # Step 1: pull cross-channel context
    memories = retrieve_memories(
        user_id=customer_email,
        query=req.message,
    )

    # Step 2: build system prompt with injected memory
    system_prompt = build_support_prompt(channel="chat", memories=memories)

    # Step 3: generate response
    reply = generate_support_reply(
        system_prompt=system_prompt,
        user_message=req.message,
    )

    # Step 4: store this turn so future channels can see it
    turn_messages = [
        {"role": "user",      "content": req.message},
        {"role": "assistant", "content": reply},
    ]
    store_memories(
        messages=turn_messages,
        user_id=customer_email,
        agent_id=CHAT_AGENT_ID,
        run_id=req.run_id,
        channel="chat",
    )

    return ChatResponse(reply=reply, context_used=memories)
