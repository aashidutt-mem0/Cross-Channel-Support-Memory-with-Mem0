"""
test_walkthrough.py
-------------------
Simulates Sarah's 3-channel support journey over 5 days.
Run after starting the FastAPI server:

    uvicorn multichannel_support:app --reload

Then:

    python3 test_walkthrough.py
"""

import httpx
import json
import time

BASE_URL = "http://localhost:8000"
CUSTOMER_EMAIL = "sarah@example.com"

def pretty(label: str, data: dict):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(json.dumps(data, indent=2))

def check_server():
    try:
        r = httpx.get(f"{BASE_URL}/", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

# ---------------------------------------------------------------------------
# Day 1 — Sarah calls about a failed payment
# ---------------------------------------------------------------------------

def day1_call():
    print("\n>>> DAY 1: Sarah calls support about a failed payment")
    transcript = """
    Sarah: Hello, I'm Sarah.
    Company: Hello, Sarah. How can I help you today?
    Sarah: I'm calling because my payment failed again. This is the second time
    this has happened. I'm on the annual plan, and my card ending in 4821 was
    declined for the $199 charge. I need the account active for work.
    Company: Can I get your email?
    Sarah: Sure, it's sarah@example.com.
    """
    r = httpx.post(f"{BASE_URL}/call", json={
        "transcript": transcript.strip(),
        "run_id": "call_session_001",
    }, timeout=30)
    pretty("POST /call response", r.json())

# ---------------------------------------------------------------------------
# Day 3 — Sarah emails asking for a receipt
# ---------------------------------------------------------------------------

def day3_email():
    print("\n>>> DAY 3: Sarah emails asking for a receipt")
    r = httpx.post(f"{BASE_URL}/email", json={
        "subject": "Receipt for May charge",
        "body": """
Hello Support Team,

I am following up on my recent call about the failed payment on my account.
The $199 charge appears to have gone through on May 14, but I have not received
a receipt yet.

Could you please send the receipt to sarah@example.com? I need it for my expense
report by the end of the week.

Best,
Sarah
        """.strip(),
        "run_id": "email_session_001",
    }, timeout=30)
    pretty("POST /email response", r.json())

# ---------------------------------------------------------------------------
# Day 5 — Sarah opens the chatbot: "is my account okay now?"
# ---------------------------------------------------------------------------

def day5_chat_with_memory():
    print("\n>>> DAY 5: Sarah opens the chatbot — WITH shared memory")
    print("    (Agent has context from the call on Day 1 and email on Day 3)")
    r = httpx.post(f"{BASE_URL}/chat", json={
        "message": "Hi, is my account okay now? My email is sarah@example.com.",
        "run_id": "chat_session_001",
    }, timeout=45)
    data = r.json()
    print(f"\n{'='*60}")
    print("  POST /chat — Agent reply WITH cross-channel memory")
    print(f"{'='*60}")
    print(f"\nMemories injected ({len(data['context_used'])} total):")
    for m in data["context_used"]:
        print(f"  - {m}")
    print(f"\nAgent reply:\n{data['reply']}")

# ---------------------------------------------------------------------------
# Comparison: what the agent says WITHOUT any memory
# ---------------------------------------------------------------------------

def day5_chat_without_memory():
    print("\n>>> COMPARISON: Same question from a NEW user (no memory)")
    r = httpx.post(f"{BASE_URL}/chat", json={
        "customer_email": "new.customer@example.com",
        "message": "Hi, is my account okay now? I had some payment issues earlier.",
        "run_id": "chat_session_anon",
    }, timeout=45)
    data = r.json()
    print(f"\n{'='*60}")
    print("  POST /chat — Agent reply WITHOUT any memory")
    print(f"{'='*60}")
    print(f"\nMemories injected: none")
    print(f"\nAgent reply:\n{data['reply']}")

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not check_server():
        print("\nERROR: Server not running.")
        print("Start it with:  uvicorn multichannel_support:app --reload")
        exit(1)

    print("\nMulti-Channel Memory Walkthrough — Sarah's 5-day support journey")
    print("Customer email:", CUSTOMER_EMAIL)

    day1_call()
    print("\n[Simulating 2-day gap before email...]\n")
    time.sleep(1)

    day3_email()
    print("\n[Simulating 2-day gap before chat...]\n")
    time.sleep(1)

    day5_chat_with_memory()
    day5_chat_without_memory()

    print("\n\nDone. Notice the difference in the two chat replies above.")
    print("The memory-aware agent opens with context. The blank agent asks Sarah to repeat herself.")
