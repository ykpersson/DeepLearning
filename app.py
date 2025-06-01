import streamlit as st
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
import joblib
from transformers import AutoModel
from main import Chatbot

import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


# Ladda embeddings in till streamplit webappen
@st.cache_resource
def load_embeddings():

# Ladda embeddings
    loaded_embeddings = joblib.load("embedded.pkl")
    return loaded_embeddings

embeddings = load_embeddings()
model = AutoModel.from_pretrained("AI-Nordics/bert-large-swedish-cased")
Chatbot = Chatbot()

# Streamlit UI
st.title("Skattejuridisk Chatbot")
user_question = st.text_input("Ställ din fråga om inkomstskattelagen:")

if user_question:

    relevant_text = "Hämtad relevant lagtext här..."

    # Skicka till Gemini
    response = genai.client("gemini-2.0-pro").generate_content(
        model="gemini-2.0-pro",
        contents=[f"Fråga: {user_question}\n\nLagutdrag: {relevant_text}"],
        config=genai.types.GenerateContentConfig(
            max_output_tokens=300,
            temperature=0.7,
            system_instruction="Ge ett juridiskt korrekt och tydligt svar baserat på lagtexten."
        )
    )
    # Visa svar
    st.write(f"📝 Gemini svarar:\n{response.text}")