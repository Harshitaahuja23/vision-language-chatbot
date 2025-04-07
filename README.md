# Vision-Language Chatbot with BLIP, RAG + Tool-Using Agent

## Problem Statement & Overview

Modern image captioning and visual QA systems often lack real-world knowledge or context-awareness. While they may describe what they "see" in an image, they can’t explain it or answer open-ended questions about it in depth — especially when such knowledge isn’t directly visible.

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



