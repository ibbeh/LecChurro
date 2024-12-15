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
    prompt_file_path = os.path.join(current_dir, 'prompts/flashcards_prompt.txt')
    
    with open(prompt_file_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    
    # Replace the placeholder with the actual transcription
    prompt = prompt_template.replace("TRANSCRIPTION_HERE", transcription_text)
    
    # Send the prompt to OpenAI's API
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI assistant that creates flashcards to help students learn."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=14000,
        temperature=0.7,
    )
    
    flashcards_raw = response.choices[0].message.content.strip()
    return flashcards_raw

def format_flashcards_markdown(flashcards_text):
    # Parse the flashcards into Front and Back pairs
    cards = flashcards_text.strip().split('\n')
    flashcards_list = []
    current_front = None
    current_back = None

    for line in cards:
        line = line.strip()
        if line.lower().startswith("front:"):
            current_front = line[len("front:"):].strip()
        elif line.lower().startswith("back:"):
            current_back = line[len("back:"):].strip()
            if current_front and current_back:
                flashcards_list.append((current_front, current_back))
                current_front = None
                current_back = None

    if not flashcards_list:
        return "No flashcards generated."

    # Create Markdown with <details> and <summary> for interactive flashcards
    md = "## Flashcards\n\n"
    for i, (front, back) in enumerate(flashcards_list, start=1):
        md += f"""
<details style="background-color: #f0f8ff; padding: 10px; border-radius: 5px;">
<summary><span style="font-size: 20px; font-weight: bold; cursor: pointer;">Flashcard {i}:</span> <span style="font-size: 18px;">{front}</span></summary>

<p style="font-size: 16px; margin-top: 10px;"><b>Answer:</b> {back}</p>

</details>

<br>
"""
    return md
