# Cross-Channel Support Memory with Mem0

This demo shows a support agent remembering a customer across call, email, and chat using Mem0.

A customer first shares context in one channel. Later, when they come back through another channel and provide the same account email, the agent retrieves the right memory instead of asking them to repeat the issue.

## What It Demonstrates

- A call transcript is saved to Mem0.
- A business email reply can use the same customer memory.
- A chat agent asks for the customer's email, retrieves existing memory, and responds with context.
- The customer's email acts as the shared memory key across channels.

## Files

- `streamlit_app.py` - Streamlit UI for the demo.
- `multichannel_support.py` - FastAPI backend and shared memory helpers.
- `test_walkthrough.py` - Optional command-line walkthrough for the FastAPI API.
- `requirements.txt` - Python dependencies.
- `.env` - Local API keys and model settings.

## Setup

Create and activate a local virtual environment:

```bash
cd /Users/aashiwork/Desktop/Multichannel_Mem0
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Fill in `.env`:

```bash
MEM0_API_KEY=your_mem0_key
OPENAI_PROVIDER=azure
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_DEPLOYMENT=your-gpt-4o-or-gpt-5-deployment
AZURE_OPENAI_API_VERSION=2024-10-21
```

For regular OpenAI instead of Azure OpenAI:

```bash
OPENAI_PROVIDER=openai
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4o
```

## Run The Streamlit Demo

```bash
source .venv/bin/activate
streamlit run streamlit_app.py
```

## Demo Flow

1. Open the **Call** tab.
2. Click `Save Call`.
3. Open the **Chat** tab.
4. The chatbot asks for the account email.
5. Enter:

```text
sarah@example.com
```

6. The chatbot retrieves Sarah's saved call memory and acknowledges the known issue.

Optional:

1. Open the **Email** tab.
2. Click `Send`.
3. The business reply uses Sarah's existing cross-channel memory.

## Run The FastAPI Backend

The Streamlit app does not require the FastAPI server. Use this only if you want to test the API endpoints directly.

```bash
source .venv/bin/activate
uvicorn multichannel_support:app --reload
```

In a second terminal:

```bash
cd /Users/aashiwork/Desktop/Multichannel_Mem0
source .venv/bin/activate
python test_walkthrough.py
```

## API Endpoints

- `POST /call` - saves a voice transcript.
- `POST /email` - saves an inbound email.
- `POST /chat` - retrieves memory and generates a chat response.

Each endpoint can infer the customer email from the message text. Internally, that email is passed to Mem0 as `user_id`, which keeps memory shared across all channels.

