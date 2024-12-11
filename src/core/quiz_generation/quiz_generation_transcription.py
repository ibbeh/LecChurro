# src/core/quiz_generation/quiz_generation_transcription.py

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with the API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_quiz(transcription_text):
    # Construct the relative path for the quiz prompt file
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of this script
    prompt_file_path = os.path.join(current_dir, 'quiz_generation_prompt_multiple_choice.txt')  # Combine directory with the file name

    # Read the prompt from the file
    with open(prompt_file_path, 'r') as file:
        prompt_template = file.read()

    # Debugging: Print the template content
    print(f"Prompt Template: {prompt_template}")

    # Format the prompt with the transcription text
    try:
        prompt = prompt_template.format(transcription_text=transcription_text)
        print(f"Formatted Prompt: {prompt}")  # Debugging
    except KeyError as e:
        print(f"Formatting error: {e}")
        raise

    # Send the formatted prompt to the OpenAI API
    try:
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
    except Exception as e:
        print(f"Error in OpenAI API call: {e}")
        raise