from html import escape

import streamlit as st
from fastapi import HTTPException

from multichannel_support import (
    CHAT_AGENT_ID,
    EMAIL_AGENT_ID,
    CallRequest,
    build_support_prompt,
    EmailRequest,
    build_email_reply_prompt,
    extract_email,
    generate_support_reply,
    handle_call,
    handle_email,
    handle_chat,
    retrieve_memories,
    store_memories,
    ChatRequest,
)


st.set_page_config(
    page_title="Mem0 Support Memory Across Call, Email, and Chat",
    page_icon=None,
    layout="centered",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #111827;
        --muted: #6b7280;
        --line: #d9dee8;
        --surface: #ffffff;
        --soft: #f6f8fb;
        --accent: #0f766e;
        --accent-dark: #0b4f4a;
    }
    .stApp {
        background:
            linear-gradient(180deg, #f7f9fc 0%, #ffffff 42%);
        color: var(--ink);
    }
    .block-container {
        max-width: 900px;
        padding-top: 3rem;
        padding-bottom: 4rem;
    }
    [data-testid="stSidebar"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"] {
        display: none;
    }
    .app-title {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 1.75rem;
        padding-bottom: 1.15rem;
        border-bottom: 1px solid rgba(17, 24, 39, 0.08);
    }
    .app-title h1 {
        font-size: 2.05rem;
        line-height: 1.1;
        font-weight: 650;
        letter-spacing: 0;
        margin: 0;
        color: var(--ink);
    }
    .mem0-mark {
        border: 1px solid rgba(15, 118, 110, 0.25);
        color: var(--accent-dark);
        background: rgba(15, 118, 110, 0.07);
        border-radius: 999px;
        padding: 0.35rem 0.7rem;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0;
        white-space: nowrap;
    }
    [data-testid="stTabs"] {
        margin-top: 0.25rem;
    }
    [data-testid="stTabs"] button {
        color: var(--muted);
        font-weight: 600;
        letter-spacing: 0;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--ink);
    }
    [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        background-color: var(--accent);
    }
    label {
        color: #374151 !important;
        font-weight: 600 !important;
    }
    input,
    textarea {
        border-radius: 6px !important;
        border-color: var(--line) !important;
        background: var(--surface) !important;
    }
    input:focus,
    textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 1px rgba(15, 118, 110, 0.12) !important;
    }
    div[data-testid="stForm"] {
        border: 0;
        padding: 0;
    }
    div[data-testid="stFormSubmitButton"] button {
        border-radius: 6px;
        border: 1px solid var(--ink);
        background: var(--ink);
        color: white;
        font-weight: 650;
        letter-spacing: 0;
        padding: 0.5rem 1rem;
        transition: all 120ms ease;
    }
    div[data-testid="stFormSubmitButton"] button:hover {
        border-color: var(--accent-dark);
        background: var(--accent-dark);
        color: white;
        transform: translateY(-1px);
    }
    div[data-testid="stSuccess"] {
        border-radius: 6px;
        border-color: rgba(15, 118, 110, 0.2);
        background: rgba(15, 118, 110, 0.08);
    }
    .mail {
        border: 1px solid var(--line);
        border-radius: 6px;
        padding: 1.1rem 1.2rem;
        margin-top: 1.1rem;
        background: var(--surface);
        box-shadow: 0 12px 35px rgba(17, 24, 39, 0.08);
        white-space: pre-wrap;
        line-height: 1.55;
    }
    .mail strong {
        color: #374151;
    }
    .chat-row {
        margin: 0.7rem 0;
        display: flex;
    }
    .chat-user {
        justify-content: flex-end;
    }
    .chat-assistant {
        justify-content: flex-start;
    }
    .bubble {
        max-width: 76%;
        border: 1px solid var(--line);
        border-radius: 6px;
        padding: 0.72rem 0.85rem;
        white-space: pre-wrap;
        background: var(--surface);
        line-height: 1.48;
        box-shadow: 0 7px 24px rgba(17, 24, 39, 0.06);
    }
    .chat-user .bubble {
        background: var(--ink);
        color: white;
        border-color: var(--ink);
        box-shadow: 0 10px 28px rgba(17, 24, 39, 0.16);
    }
    .chat-assistant .bubble {
        background: #fbfcfe;
    }
    @media (max-width: 640px) {
        .block-container {
            padding-top: 1.8rem;
        }
        .app-title {
            align-items: flex-start;
            flex-direction: column;
        }
        .app-title h1 {
            font-size: 1.55rem;
        }
        .bubble {
            max-width: 88%;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def show_error(error: Exception) -> None:
    if isinstance(error, HTTPException):
        st.error(error.detail)
    else:
        st.error(str(error))


def render_mail(sender: str, recipient: str, subject: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="mail">
        <strong>From:</strong> {escape(sender)}<br>
        <strong>To:</strong> {escape(recipient)}<br>
        <strong>Subject:</strong> {escape(subject)}<br><br>
        {escape(body)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_conversation(messages: list[dict]) -> None:
    for message in messages:
        role_class = "chat-user" if message["role"] == "user" else "chat-assistant"
        st.markdown(
            f"""
            <div class="chat-row {role_class}">
              <div class="bubble">{escape(message["content"])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if "reply_email" not in st.session_state:
    st.session_state.reply_email = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "Hi, Please share your email before getting started."}
    ]
if "chat_customer_email" not in st.session_state:
    st.session_state.chat_customer_email = None
if "pending_chat_messages" not in st.session_state:
    st.session_state.pending_chat_messages = []


st.markdown(
    """
    <div class="app-title">
      <h1>Cross-Channel Support Memory with Mem0</h1>
      <div class="mem0-mark">MEM0</div>
    </div>
    """,
    unsafe_allow_html=True,
)

call_tab, email_tab, chat_tab = st.tabs(["Call", "Email", "Chat"])

with call_tab:
    call_messages = [
        {"role": "user", "content": "Hello, I'm Sarah."},
        {"role": "assistant", "content": "Hello, Sarah. How can I help you today?"},
        {
            "role": "user",
            "content": (
                "I'm calling because my payment failed again. This is the second time "
                "this has happened. I'm on the annual plan, and my card ending in 4821 "
                "was declined for the $199 charge. I need the account active for work."
            ),
        },
        {"role": "assistant", "content": "Can I get your email?"},
        {"role": "user", "content": "Sure, it's sarah@example.com."},
    ]
    render_conversation(call_messages)

    with st.form("call_form", clear_on_submit=False):
        submitted = st.form_submit_button("Save Call")

    if submitted:
        try:
            transcript = "\n".join(
                f"{'Sarah' if message['role'] == 'user' else 'Company'}: {message['content']}"
                for message in call_messages
            )
            handle_call(CallRequest(transcript=transcript, run_id="streamlit_call"))
            st.success("Call saved")
        except Exception as error:
            show_error(error)

with email_tab:
    with st.form("email_form", clear_on_submit=False):
        from_email = st.text_input("From", value="sarah@example.com")
        to_email = st.text_input("To", value="support@example.com")
        subject = st.text_input("Subject", value="Receipt request for May 14 payment")
        body = st.text_area(
            "Message",
            value=(
                "Hello Support Team,\n\n"
                "I am following up on my recent call about the failed payment on my account. "
                "The $199 charge appears to have gone through on May 14, but I have not "
                "received a receipt yet.\n\n"
                "Could you please send the receipt to this email address? I need it for my "
                "expense report by the end of the week.\n\n"
                "Best,\n"
                "Sarah"
            ),
            height=230,
        )
        submitted = st.form_submit_button("Send")

    if submitted:
        try:
            inbound_body = f"From: {from_email}\n\n{body}"
            handle_email(
                EmailRequest(
                    customer_email=from_email,
                    subject=subject,
                    body=inbound_body,
                    run_id="streamlit_email",
                )
            )
            memories = retrieve_memories(
                user_id=from_email.lower(),
                query=f"{subject}\n{body}",
                limit=6,
            )
            reply_body = generate_support_reply(
                system_prompt=build_email_reply_prompt(memories),
                user_message=f"Subject: {subject}\n\n{body}",
            )
            store_memories(
                messages=[{"role": "assistant", "content": reply_body}],
                user_id=from_email.lower(),
                agent_id=EMAIL_AGENT_ID,
                run_id="streamlit_email",
                channel="email",
            )
            st.session_state.reply_email = {
                "sender": to_email,
                "recipient": from_email,
                "subject": f"Re: {subject}",
                "body": reply_body,
            }
        except Exception as error:
            show_error(error)

    if st.session_state.reply_email:
        render_mail(**st.session_state.reply_email)

with chat_tab:
    render_conversation(st.session_state.chat_messages)

    with st.form("chat_form", clear_on_submit=True):
        message = st.text_input("Message", value="")
        submitted = st.form_submit_button("Send")

    if submitted and message.strip():
        st.session_state.chat_messages.append({"role": "user", "content": message})
        found_email = extract_email(message)
        if found_email:
            st.session_state.chat_customer_email = found_email

        if not st.session_state.chat_customer_email:
            try:
                response = handle_chat(ChatRequest(message=message, run_id="streamlit_chat"))
                st.session_state.chat_messages.append({"role": "assistant", "content": response.reply})
                st.session_state.pending_chat_messages.extend(
                    [
                        {"role": "user", "content": message},
                        {"role": "assistant", "content": response.reply},
                    ]
                )
            except Exception as error:
                show_error(error)
        else:
            try:
                if st.session_state.pending_chat_messages:
                    pending_user_text = "\n".join(
                        item["content"]
                        for item in st.session_state.pending_chat_messages
                        if item["role"] == "user"
                    )
                    store_memories(
                        messages=st.session_state.pending_chat_messages,
                        user_id=st.session_state.chat_customer_email,
                        agent_id=CHAT_AGENT_ID,
                        run_id="streamlit_chat",
                        channel="chat",
                    )
                    st.session_state.pending_chat_messages = []
                else:
                    pending_user_text = ""

                email_only_turn = bool(found_email) and not pending_user_text.strip()
                query_text = (
                    f"{pending_user_text}\n{message}".strip()
                    if not email_only_turn
                    else "recent support history payment failed annual plan card declined account active receipt issue"
                )
                memories = retrieve_memories(
                    user_id=st.session_state.chat_customer_email,
                    query=query_text,
                    limit=6,
                )
                if email_only_turn:
                    reply = generate_support_reply(
                        system_prompt=(
                            build_support_prompt(channel="chat", memories=memories)
                            + "\n- The customer has just provided their email. If prior memory is available, briefly acknowledge the remembered issue and ask whether they are contacting you about the same issue or need help with something else."
                            + "\n- Do not propose payment actions yet."
                            + "\n- Greet the customer only once."
                        ),
                        user_message=(
                            "The customer provided their email address. Use the retrieved memory, if any, "
                            "to acknowledge their known support history and ask what they need next."
                        ),
                    )
                else:
                    reply = generate_support_reply(
                        system_prompt=(
                            build_support_prompt(channel="chat", memories=memories)
                            + "\n- In an ongoing chat, greet the customer only once. Do not start every reply with repeated salutations like 'Hi Sarah'."
                        ),
                        user_message=query_text,
                    )
                st.session_state.chat_messages.append({"role": "assistant", "content": reply})
                store_memories(
                    messages=[
                        {"role": "user", "content": message},
                        {"role": "assistant", "content": reply},
                    ],
                    user_id=st.session_state.chat_customer_email,
                    agent_id=CHAT_AGENT_ID,
                    run_id="streamlit_chat",
                    channel="chat",
                )
            except Exception as error:
                show_error(error)

        st.rerun()
