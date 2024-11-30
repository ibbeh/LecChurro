# src/core/quiz_generation/quiz_generation_transcription.py

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with the API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_quiz(transcription_text):
    prompt = f"""
    Based on the following lecture transcription, generate a set of quiz questions including:

    - Multiple-choice questions (provide 4 options each)
    - True or False questions
    - Matching questions

    Ensure the questions cover key concepts and vary in difficulty.

    Lecture Transcription:
    "{transcription_text}"

    Provide the questions followed by the answers.
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert teacher skilled in producing detailed and correct student assessments."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.7,
    )
    quiz = response.choices[0].message.content.strip()
    return quiz
