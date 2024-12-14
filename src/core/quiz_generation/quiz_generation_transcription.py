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
    prompt_file_path = os.path.join(current_dir, 'quiz_generation_json.txt')  # Combine directory with the file name

    # Read the prompt from the file
    with open(prompt_file_path, 'r', encoding='utf-8') as file:
        prompt_template = file.read()

    # Replace the placeholder with the actual transcription
    prompt = prompt_template.replace("TRANSCRIPTION_HERE", transcription_text)

    # Send the formatted prompt to the OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert teacher skilled in producing detailed and correct student assessments."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=14000,
        temperature=0.7,
    )
    quiz = response.choices[0].message.content.strip()
    return quiz
