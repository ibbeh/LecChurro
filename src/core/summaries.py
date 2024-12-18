# src/core/summaries.py
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from the .env file to access API keys or other configurations
load_dotenv()

# Initialize OpenAI client with the API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def summarize_text(transcription, segments):
    """
    Summarizes the entire lecture transcription.
    Args:
    - transcription (str): The complete transcription of the lecture.
    - segments (list of dict): A list of segments, each containing 'start', 'end', and 'text'.
    Returns:
    - summary (str): A consolidated summary of the entire lecture.
    """
    try:
        # Format timestamps for major segments as a reference for summarization
        major_segments = []
        for segment in segments:
            start = segment["start"]
            end = segment["end"]
            text = segment["text"]
            # Include segments with significant content length as major segments
            if len(text) > 50:  # Assuming a major segment is defined by having more significant content
                major_segments.append(f"{start:.2f}s - {end:.2f}s: {text[:100]}...")  # Truncate to avoid long texts

        # Create a reference string of timestamps for the summarization prompt
        timestamps_reference = "\n".join(major_segments)

        # Construct the path to the summarization prompt file
        current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of this script
        prompt_file_path = os.path.join(current_dir, 'prompts/summarization_prompt.txt')  # Combine directory with the file name

        # Check if the prompt file exists
        if not os.path.exists(prompt_file_path):
            raise FileNotFoundError(f"Prompt file not found at {prompt_file_path}")

        # Read the summarization prompt template from the file
        with open(prompt_file_path, 'r') as file:
            prompt_template = file.read()

        # Format the prompt with the transcription and timestamps reference
        prompt = prompt_template.format(transcription=transcription, timestamps_reference=timestamps_reference)

        # Log the constructed prompt for debugging purposes (truncate long prompts)
        print(f"Constructed prompt: {prompt[:500]}...")

        # Send the prompt to OpenAI's GPT API (gpt4o) for summarization
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an AI assistant that summarizes lecture transcripts."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=14000, # Maximum tokens for the response
            temperature=0.5,  # Controls creativity in response (lower temperature for more deterministic output)
        )

        # Extract and return the summary from the API response
        summary = response.choices[0].message.content.strip()
        print(f"Generated summary: {summary[:500]}...")

        return summary

    except Exception as e:
        # Log and return an error message if summarization fails
        print(f"Error in summarize_text: {e}")
        return "Error: Unable to generate summary."

if __name__ == "__main__":
    # Dummy transcription and segments for testing
    dummy_transcription = "This is an example transcription of a lecture. It covers various topics in detail."
    dummy_segments = [
        {"start": 0.0, "end": 10.0, "text": "This is an example segment of the transcription."},
        {"start": 10.0, "end": 20.0, "text": "This segment goes into more detail about the lecture topics."},
    ]

    # Generate summary
    summary = summarize_text(dummy_transcription, dummy_segments)
    print("Summary:", summary)
