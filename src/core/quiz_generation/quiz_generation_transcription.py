import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with the API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_file(file_path):
    """Loads and returns the content of a text file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def generate_quiz(prompt_file, summary_file, output_file, model="gpt-3.5-turbo"):
    # Load the prompt template and summary text
    prompt_template = load_file(prompt_file)
    summary = load_file(summary_file)

    # Combine prompt and summary content
    prompt = f"{prompt_template}\n\n{summary}"

    # Send the prompt to the GPT API for quiz generation
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an AI assistant that creates quiz questions based on provided content."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.7
    )

    # Extract the quiz content from the response
    quiz_content = completion.choices[0].message.content.strip()

    # Save the quiz to the output file
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(quiz_content)
    print(f"Quiz saved to {output_file}")

if __name__ == "__main__":
    import sys

    # Ensure the correct number of arguments are provided
    if len(sys.argv) != 4:
        print("Usage: python generate_quiz.py <prompt_file> <summary_file> <output_file>")
        sys.exit(1)

    # File paths from command-line arguments
    prompt_file = sys.argv[1]
    summary_file = sys.argv[2]
    output_file = sys.argv[3]

    # Run the quiz generation
    generate_quiz(prompt_file, summary_file, output_file)