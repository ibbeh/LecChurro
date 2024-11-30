import os
#from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
#load_dotenv()

# Initialize OpenAI client with the API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_file(file_path):
    """Loads and returns the content of a text file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def summarize_lecture(prompt_file, transcription_file, output_file, model="gpt-4-turbo"):
    # Load the prompt template and transcription text
    prompt_template = load_file(prompt_file)
    transcription = load_file(transcription_file)

    # Concatenate the prompt template and transcription
    prompt = f"{prompt_template}\n\n{transcription}"

    # Send the prompt to the GPT API for summarization
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an AI assistant that helps summarize lecture notes."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.5
    )

    # Extract the summary from the response
    summary = completion.choices[0].message.content.strip()


    # Save the summary to the output file
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(summary)
    print(f"Summary saved to {output_file}")

if __name__ == "__main__":
    import sys

    # Ensure the correct number of arguments are provided
    if len(sys.argv) != 4:
        print("Usage: python summarize_lecture.py <prompt_file> <transcription_file> <output_file>")
        sys.exit(1)

    # File paths from command-line arguments
    prompt_file = sys.argv[1]
    transcription_file = sys.argv[2]
    output_file = sys.argv[3]

    # Run the summarization
    summarize_lecture(prompt_file, transcription_file, output_file)
