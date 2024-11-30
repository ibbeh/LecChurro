# src/core/summarization/summarize_text.py

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
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
    
    # Format timestamps for major segments as a reference for summarization
    major_segments = []
    for segment in segments:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]

        if len(text) > 50:  # Assuming a major segment is defined by having more significant content
            major_segments.append(f"{start:.2f}s - {end:.2f}s: {text[:100]}...")  # Truncate to avoid long texts
    
    # Creating a reference for the AI to use timestamps
    timestamps_reference = "\n".join(major_segments)

    prompt = f"""
    You are an AI assistant that helps students learn by summarizing lecture content.
    
    Summarize the following lecture transcription in a clear, concise format that highlights key points, main topics, and important details.
    Focus on the following:
    1. Identify and list main topics covered.
    2. Highlight key points and essential details for each topic.
    3. Simplify complex information for easy understanding.
    4. Provide an overview of the lecture structure.
    5. Include the timestamps for the main topics.

    Use bullet points or short paragraphs to organize the summary. Avoid overly technical jargon and make it student-friendly.
    
    Lecture Transcription:
    "{transcription}"

    Timestamps for Reference:
    {timestamps_reference}
    """

    # Send the prompt to the GPT API for summarization
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI assistant that summarizes lecture transcripts."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.5,
    )

    # Extract the summary from the response
    summary = response.choices[0].message.content.strip()

    return summary
