import streamlit as st
from PIL import Image
from agent import process_user_message

st.set_page_config(page_title="Vision Agent Chatbot", layout="wide")
st.title("🧠 Vision-Language Chatbot with RAG")
st.markdown("Upload an image and chat with the AI agent. It can caption images, answer questions, compare captions, and retrieve facts from Wikipedia.")

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

# --- SIDEBAR IMAGE UPLOAD ---
st.sidebar.header("📷 Upload an Image")
uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.session_state.uploaded_image = image
    st.sidebar.image(image, caption="Uploaded Image", use_container_width=True)

# --- MAIN CHAT UI ---
st.markdown("---")
if st.session_state.uploaded_image:
    for entry in st.session_state.messages:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])

    # User message input
    user_input = st.chat_input("Type your message here...")

    if user_input:
        # Show user message immediately
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Agent is thinking..."):
                result = process_user_message(st.session_state.uploaded_image, user_input)
                tool = result.get("tool")

                reply = f"**Tool Used:** `{tool}`\n"
                if "caption" in result:
                    reply += f"**Caption:** {result['caption']}\n"
                if "wiki_info" in result:
                    reply += f"**Wikipedia Info:** {result['wiki_info']}\n"
                if "llm_response" in result:
                    reply += f"**Response:** {result['llm_response']}\n"
                if "answer" in result:
                    reply += f"**Answer:** {result['answer']}\n"
                if "match_score" in result:
                    score = result['match_score'] * 100
                    reply += f"**Match Score:** {score:.2f}%\n"
                if "blip_caption" in result:
                    reply += f"**BLIP Caption:** {result['blip_caption']}\n"
                if "error" in result:
                    reply += f"⚠️ {result['error']}"

                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
else:
    st.warning("Please upload an image to begin.")