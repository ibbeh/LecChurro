# src/core/timestamps/generate_conceptual_timestamps.py
import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_conceptual_timestamps(transcription, segments):
    if not segments:
        return "<p>No segments found.</p>"

    segments_data = [{"start": s["start"], "end": s["end"], "text": s["text"].strip()} for s in segments]

    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_file_path = os.path.join(current_dir, 'prompts/group_concepts_prompt.txt')
    with open(prompt_file_path, 'r') as f:
        prompt_template = f.read()

    prompt = f"{prompt_template}\n\nInput segments (JSON):\n{json.dumps(segments_data)}"

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs only the requested data."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.7,
        )
        concept_groups_raw = response.choices[0].message.content.strip()
        concept_groups = json.loads(concept_groups_raw)
    except Exception as e:
        print(f"Error generating conceptual timestamps: {e}")
        return "<p>Unable to generate conceptual timestamps. Please try again later.</p>"

    # Build HTML output
    timestamps_html = ""
    for group in concept_groups:
        start = group.get("start_time", 0)
        end = group.get("end_time", 0)
        title = group.get("title", "Untitled Concept")
        summary = group.get("summary", "")

        # Main concept title link
        main_link = f"<a href='#' class='timestamp-link' data-time='{start:.2f}'><b>{title}</b></a>"

        segs_html = "<ul>"
        for sub in group.get("segments", []):
            sub_title = sub.get("mini_title", "Sub-topic")
            sub_start = sub.get("start_time", start)
            sub_text = sub.get("text", "")
            segs_html += f"<li><a href='#' class='timestamp-link' data-time='{sub_start:.2f}'><b>{sub_title}</b></a>: {sub_text}</li>"
        segs_html += "</ul>"

        timestamps_html += f"""
        <div style='margin-bottom:20px; border-bottom:1px solid #ccc; padding-bottom:10px;'>
            <h3>{main_link} <small>({start:.2f}s - {end:.2f}s)</small></h3>
            <p><em>{summary}</em></p>
            {segs_html}
        </div>
        """

    return timestamps_html

def format_timestamps(segments):
    timestamps_data = []
    for segment in segments:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]
        clickable_text = f"<a href='javascript:void(0);' onclick='seekVideo({start});'>{text}</a>"
        timestamps_data.append({"Start Time": start, "End Time": end, "Text": clickable_text})
    return timestamps_data

