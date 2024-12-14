# src/core/flashcards/flashcards_generation.py
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with the API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_flashcards(transcription_text):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_file_path = os.path.join(current_dir, 'flashcards_prompt.txt')
    with open(prompt_file_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    prompt = prompt_template.format(transcription_text=transcription_text)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI assistant that creates flashcards to help students learn."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.7,
    )
    flashcards_raw = response.choices[0].message.content.strip()

    return flashcards_raw
