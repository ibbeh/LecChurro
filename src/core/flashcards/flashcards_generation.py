# src/core/flashcards/flashcards_generation.py

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with the API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_flashcards(transcription_text):
    prompt = f"""
    Based on the following lecture transcription, generate a set of flashcards.

    Each flashcard should have:
    - A question or term on the front.
    - The answer or definition on the back.

    Lecture Transcription:
    "{transcription_text}"

    Provide the flashcards in the format:

    Front: [Question/Term]
    Back: [Answer/Definition]
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI assistant that creates flashcards to help students learn."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.7,
    )
    flashcards = response.choices[0].message.content.strip()
    return flashcards
