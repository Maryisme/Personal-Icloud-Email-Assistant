# gradio_app.py
import gradio as gr
from notebook_port import build_conversational_chain

# Build once at startup. Uses your Chroma store + Ollama settings from notebook_port.py
def _new_chain():
    return build_conversational_chain()

# Keep the chain in app state so its internal memory tracks the conversation
CHAIN = _new_chain()

def respond(user_msg, chat_history):
    """
    Called on each submit. We let the LangChain ConversationalRetrievalChain
    handle history internally; Gradio history is only for UI display.
    """
    try:
        result = CHAIN.invoke({"question": user_msg})
        answer = result.get("answer") or ""
    except Exception as e:
        # Fail loudly so you fix config fast
        answer = f"[ERROR] {type(e).__name__}: {e}"
    return answer

def reset_session():
    """
    Rebuild the chain to clear its internal memory AND clear the UI.
    """
    global CHAIN
    CHAIN = _new_chain()
    return []

with gr.Blocks(title="Email RAG Chat") as demo:
    gr.Markdown("### Email RAG Chat")
    chat = gr.Chatbot(height=450)
    msg = gr.Textbox(placeholder="Ask something grounded in your emails…", autofocus=True)
    send = gr.Button("Send", variant="primary")
    reset = gr.Button("Reset session")

    # On submit via Enter or Send
    def _user_submit(m, h):
        h = h + [[m, None]]
        return "", h

    def _bot_reply(h):
        user_latest = h[-1][0]
        bot = respond(user_latest, h)
        h[-1][1] = bot
        return h

    msg.submit(_user_submit, [msg, chat], [msg, chat]).then(_bot_reply, inputs=chat, outputs=chat)
    send.click(_user_submit, [msg, chat], [msg, chat]).then(_bot_reply, inputs=chat, outputs=chat)

    # Full memory reset (chain + UI)
    reset.click(fn=reset_session, outputs=chat)

if __name__ == "__main__":
    # Expose on localhost:7860. Change host/port as needed.
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)


