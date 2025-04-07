# Vision-Language Chatbot with BLIP, RAG + Tool-Using Agent

## Problem Statement & Overview

Modern image captioning and visual QA systems often lack real-world knowledge or context-awareness. While they may describe what they "see" in an image, they can’t explain it or answer open-ended questions about it in depth - especially when such knowledge isn’t directly visible.

This project addresses that gap by building a **streamlit-based multimodal chatbot** that understands images using **BLIP** (Bootstrapped Language-Image Pretraining), then intelligently selects tools to:
- Generate captions
- Answer visual questions
- Compare captions to verify user input
- Look up factual knowledge from Wikipedia
- Perform RAG (retrieval-augmented generation) to provide detailed explanations

The chatbot acts like an **AI Agent**, using a lightweight LLM (hosted on [Groq](https://groq.com/)) to decide which specialized tool should be invoked to respond best to user input.

### Goals:
- Build an intelligent image chatbot that feels interactive and smart
- Seamlessly combine **image understanding**, **external factual retrieval**, and **LLM-based reasoning**
- Explore real-world tool-use reasoning by an LLM Agent


## Methodology

This project integrates vision-language reasoning using a combination of the following:

---

### BLIP Transformers for Image Understanding Tasks

- **Image Captioning** with `Salesforce/blip-image-captioning-base`
- **Visual Question Answering (VQA)** with `Salesforce/blip-vqa-base`
- **Image-Text Matching** with `Salesforce/blip-itm-base-coco`

---

### RAG (Retrieval-Augmented Generation) Pipeline

- Uses the **image caption** to fetch Wikipedia context using the `wikipedia` Python library
- Combines **visual + factual data** to answer deep questions with an LLM

---

### AI Agent for Tool Selection

- Uses a Groq-hosted LLaMA3-8B-Instruct model to analyze user input
- Dynamically selects the appropriate tool from:
  - `caption`, `vqa`, `compare_caption`, `get_info`, `rag_answer`
- Enables an adaptive, modular response pipeline

---

### Streamlit Chat Interface

- Simple, user-friendly chat UI for uploading images and interacting with the AI assistant
- Maintains **stateful conversation**
- Displays all intermediate outputs (captions, Wikipedia info, answers, etc.)

---

### Techniques from the Course Applied

- **Multimodal Learning**: Combining text and vision input for richer AI reasoning
- **Transformer Architectures**: BLIP and LLaMA3 are both state-of-the-art transformer-based models
- **Agent-based Reasoning**: The agent acts as an orchestrator using LLM-driven decision-making
- **RAG for Context Injection**: Extending the model’s knowledge using external retrieval (Wikipedia)

  ## Implementation & Code

This project is structured with modular Python files and is fully runnable through a **Streamlit** interface.

---

### Runnable Demo (Live Chatbot)

You can run the full chatbot via:

```bash
streamlit run app.py

GROQ_API_KEY=your_groq_api_key_here


