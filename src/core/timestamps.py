# src/core/timestamps.py
import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from the .env file to access API keys or other configurations
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) # Initialize OpenAI client with the API key

def generate_conceptual_timestamps(transcription, segments):
    """
    Generates conceptual timestamps for a lecture transcript.
    Args:
        transcription (str): The full transcription of the lecture.
        segments (list of dict): A list of segments, where each segment contains:
            - 'start' (float): Start time of the segment in seconds.
            - 'end' (float): End time of the segment in seconds.
            - 'text' (str): Text content of the segment.
    Returns:
        str: HTML-formatted string of conceptual groups with timestamps and summaries.
    """
    # Return a message if no segments are provided
    if not segments:
        return "<p>No segments found.</p>"
    # Prepare a list of segments as JSON for the GPT prompt
    segments_data = [{"start": s["start"], "end": s["end"], "text": s["text"].strip()} for s in segments]

    # Load the conceptual grouping prompt from a file
    current_dir = os.path.dirname(os.path.abspath(__file__)) # Get the directory of this script
    prompt_file_path = os.path.join(current_dir, 'prompts/group_concepts_prompt.txt') # Combine directory with prompt file name

    with open(prompt_file_path, 'r') as f:
        prompt_template = f.read()

    # Inject the segments data into the prompt
    prompt = f"{prompt_template}\n\nInput segments (JSON):\n{json.dumps(segments_data)}"

    try:
        # Send the prompt to the OpenAI API (gpt4) to generate conceptual groups
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs only the requested data."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500, # Max tokens for response
            temperature=0.7, # Controls randomness (creativity) in response
        )

        # Parse the response into a JSON object
        concept_groups_raw = response.choices[0].message.content.strip()
        concept_groups = json.loads(concept_groups_raw)

    # Handle errors during the API call or response parsing    
    except Exception as e:
        print(f"Error generating conceptual timestamps: {e}")
        return "<p>Unable to generate conceptual timestamps. Please try again later.</p>"

    # Build HTML output for the conceptual groups
    timestamps_html = ""
    for group in concept_groups:
        # Extract details of each conceptual group
        start = group.get("start_time", 0)
        end = group.get("end_time", 0)
        title = group.get("title", "Untitled Concept")
        summary = group.get("summary", "")

        # Main concept title link
        main_link = f"<a href='#' class='timestamp-link' data-time='{start:.2f}'><b>{title}</b></a>"

        # Generate a list of sub-topics within the group
        segs_html = "<ul>"
        for sub in group.get("segments", []):
            sub_title = sub.get("mini_title", "Sub-topic")
            sub_start = sub.get("start_time", start)
            sub_text = sub.get("text", "")
            segs_html += f"<li><a href='#' class='timestamp-link' data-time='{sub_start:.2f}'><b>{sub_title}</b></a>: {sub_text}</li>"
        segs_html += "</ul>"

        # Add the conceptual group and its sub-topics to the HTML output
        timestamps_html += f"""
        <div style='margin-bottom:20px; border-bottom:1px solid #ccc; padding-bottom:10px;'>
            <h3>{main_link} <small>({start:.2f}s - {end:.2f}s)</small></h3>
            <p><em>{summary}</em></p>
            {segs_html}
        </div>
        """

    return timestamps_html

def format_timestamps(segments):
    """
    Formats lecture segments into clickable timestamps.
    Args:
        segments (list of dict): A list of segments, where each segment contains:
            - 'start' (float): Start time of the segment in seconds.
            - 'end' (float): End time of the segment in seconds.
            - 'text' (str): Text content of the segment.
    Returns:
        list of dict: Each dict includes "Start Time", "End Time", and "Text" with clickable HTML.
    """
    timestamps_data = []
    for segment in segments:
        # Extract start time, end time, and text for each segment
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]
        # Create clickable HTML for the segment text
        clickable_text = f"<a href='javascript:void(0);' onclick='seekVideo({start});'>{text}</a>"
        # Append formatted data to the list
        timestamps_data.append({"Start Time": start, "End Time": end, "Text": clickable_text})
        
    return timestamps_data

