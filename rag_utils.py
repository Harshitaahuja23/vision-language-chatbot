import wikipedia
import os
from dotenv import load_dotenv
import openai

load_dotenv()

openai.api_key = os.getenv("GROQ_API_KEY")
openai.api_base = "https://api.groq.com/openai/v1"

def search_wikipedia(query, sentences=5):
    """Search Wikipedia and return a summary based on the query"""
    try:
        result = wikipedia.summary(query, sentences=sentences)
        return result
    except wikipedia.DisambiguationError as e:
        return f"⚠️ Disambiguation Error: Try one of these - {e.options[:5]}"
    except wikipedia.PageError:
        return "⚠️ No Wikipedia page found for this term."
    except Exception as e:
        return f"⚠️ Wikipedia search failed: {str(e)}"

def generate_response(caption, wiki_text, question=None, max_tokens=200):

    prompt = f"""You are a knowledgeable and helpful AI assistant.

    Use the image caption and factual context below to generate a meaningful and informative response. 
    If a user question is provided, answer it directly using both the visual and factual information.

    Image Caption:
    {caption}

    Wikipedia Context:
    {wiki_text}
    """

    if question:
        prompt += f"User Question: {question}\nAnswer:"
    else:
        prompt += "Describe the image in more detail using the context.\nAnswer:"

    response = openai.ChatCompletion.create(
        model="llama3-8b-8192",  
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=max_tokens
    )

    return response["choices"][0]["message"]["content"].strip()
