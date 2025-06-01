import os
import google.genai as genai
from google.genai import types
from dotenv import load_dotenv

class Chatbot:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GEMINI_API_KEY")
    
        self.client = genai.Client(api_key=self.api_key)
    
    def ask(self, user_question, relevant_text=""):
        """Genererar svar med Gemini baserat på användarfrågan och relevant text."""
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[f"Fråga: {user_question}\n\nLagutdrag: {relevant_text}"],
            config=types.GenerateContentConfig(
                max_output_tokens=300,
                temperature=0.7,
                system_instruction="Ge ett juridiskt korrekt och tydligt svar baserat på lagtexten."
            )
        )
        return response.text

