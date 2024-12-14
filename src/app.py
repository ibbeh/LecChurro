# src/app.py
import os
import shutil
import gradio as gr
import ffmpeg
import whisper
import openai
import warnings
import traceback
import json
import re

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
warnings.filterwarnings("ignore", message="You are using torch.load with weights_only=False")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

from core.summarization.summarize_text import summarize_text
from core.quiz_generation.quiz_generation_transcription import generate_quiz
from core.flashcards.flashcards_generation import generate_flashcards
from core.timestamps.generate_conceptual_timestamps import generate_conceptual_timestamps

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
AUDIO_DIR = os.path.join(DATA_DIR, 'audio')
TEXT_DIR = os.path.join(ROOT_DIR, 'data', 'text_timestamps')
VIDEO_DIR = os.path.join(DATA_DIR, 'video')

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

print("Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("Whisper model loaded.")

MAX_QUESTIONS = 50  # You can adjust this as needed

def extract_audio(video_file_path, audio_file_path):
    print("Extracting audio with ffmpeg...")
    try:
        (
            ffmpeg
            .input(video_file_path)
            .output(audio_file_path, format='wav', acodec='pcm_s16le', ac=1, ar='16k')
            .overwrite_output()
            .run(quiet=True)
        )
        print("Audio extraction successful.")
        return True
    except Exception as e:
        print(f"Error extracting audio: {e}")
        traceback.print_exc()
        return False

def transcribe_audio(audio_file_path):
    print("Transcribing audio with Whisper...")
    result = whisper_model.transcribe(audio_file_path)
    transcription = result["text"]
    segments = result["segments"]
    print("Transcription complete.")
    return transcription, segments

def process_video(video_file):
    print(f"Processing video file: {video_file}")
    if not video_file or not os.path.isfile(video_file):
        print("Invalid video file path.")
        return "Error extracting audio.", None, None, None, None

    video_filename = os.path.basename(video_file)
    video_path = os.path.join(VIDEO_DIR, video_filename)
    shutil.copy(video_file, video_path)
    print(f"Video saved to: {video_path}")

    audio_filename = os.path.splitext(video_filename)[0] + ".wav"
    audio_path = os.path.join(AUDIO_DIR, audio_filename)
    success = extract_audio(video_path, audio_path)
    if not success:
        return "Error extracting audio.", None, None, None, None

    try:
        transcription, segments = transcribe_audio(audio_path)
        print(f"Transcription snippet: {transcription[:100]}...")
        print(f"First 3 segments: {segments[:3]}...")
    except Exception as e:
        print(f"Error in transcription: {e}")
        traceback.print_exc()
        return "Error transcribing audio.", None, None, None, None

    try:
        summary = summarize_text(transcription, segments)
        print(f"Summary generated snippet: {summary[:100]}...")
    except Exception as e:
        print(f"Error in summarizing text: {e}")
        traceback.print_exc()
        summary = None

    try:
        quizzes_str = generate_quiz(transcription)
        print(f"Quiz generated snippet: {quizzes_str[:100]}...")
        match = re.search(r"quizzes\s*=\s*(\[.*\])", quizzes_str, flags=re.DOTALL)
        if match:
            quizzes_json_str = match.group(1)
            quizzes = json.loads(quizzes_json_str)
        else:
            quizzes = []
    except Exception as e:
        print(f"Error parsing quizzes: {e}")
        traceback.print_exc()
        quizzes = None

    try:
        flashcards = generate_flashcards(transcription)
    except Exception as e:
        print(f"Error in generating flashcards: {e}")
        traceback.print_exc()
        flashcards = None

    return video_path, summary, segments, quizzes, flashcards

def format_timestamps(segments):
    timestamps_data = []
    for segment in segments:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]
        clickable_text = f"<a href='javascript:void(0);' onclick='seekVideo({start});'>{text}</a>"
        timestamps_data.append({"Start Time": start, "End Time": end, "Text": clickable_text})
    return timestamps_data

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


def main():
    with gr.Blocks() as demo:
        gr.Markdown("# LecChurro: From Lecture to Learning")

        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="Upload Lecture Video", elem_id="main_video_player")
                transcribe_button = gr.Button("Transcribe Now")

        with gr.Tabs():
            with gr.Tab("Summary/Notes"):
                summary_output = gr.Markdown(label="Summary/Notes")
            with gr.Tab("Timestamps"):
                timestamps_output = gr.HTML(label="Timestamps")
            with gr.Tab("Quizzes"):
                quiz_output = gr.HTML(label="Quizzes")
                quiz_radios = []
                for i in range(MAX_QUESTIONS):
                    r = gr.Radio(choices=[], label=f"Q{i+1}", visible=False)
                    quiz_radios.append(r)
                submit_quiz_button = gr.Button("Submit Quiz", visible=False)
                quiz_feedback = gr.Markdown("", visible=False)
            with gr.Tab("Flashcards"):
                # Use Markdown to display interactive flashcards
                flashcards_output = gr.Markdown(label="Flashcards")

        quizzes_state = gr.State()

        def on_transcribe(video_file):
            if video_file is None:
                return (gr.update(), "Please upload a video file.", "", "", "",
                        *([gr.update(visible=False)]*MAX_QUESTIONS),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        [])

            if isinstance(video_file, dict) and "name" in video_file:
                video_file_path = video_file["name"]
            else:
                video_file_path = video_file if isinstance(video_file, str) else None

            if not video_file_path or not os.path.isfile(video_file_path):
                return (gr.update(), "Please upload a video file.", "", "", "",
                        *([gr.update(visible=False)]*MAX_QUESTIONS),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        [])

            try:
                video_path, summary, segments, quizzes, flashcards = process_video(video_file_path)
            except Exception as e:
                print("Error during process_video call:")
                traceback.print_exc()
                return (gr.update(), f"Error processing video: {e}", "", "", "",
                        *([gr.update(visible=False)]*MAX_QUESTIONS),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        [])

            timestamps_html = generate_conceptual_timestamps(summary, segments) if segments else ""
            if quizzes and isinstance(quizzes, list) and len(quizzes) > 0:
                quiz_html = "<p>Select your answers and click Submit Quiz.</p>"
                quiz_count = min(len(quizzes), MAX_QUESTIONS)
                radios_updates = []
                for i in range(MAX_QUESTIONS):
                    if i < quiz_count:
                        q = quizzes[i]
                        radios_updates.append(gr.update(label=f"Q{i+1}: {q['question']}", choices=q["options"], visible=True))
                    else:
                        radios_updates.append(gr.update(visible=False))
                submit_upd = gr.update(visible=True)
                feedback_upd = gr.update(visible=True, value="")
            else:
                quiz_html = "<p>No quizzes generated.</p>"
                radios_updates = [gr.update(visible=False) for _ in range(MAX_QUESTIONS)]
                submit_upd = gr.update(visible=False)
                feedback_upd = gr.update(visible=False)

            if flashcards and isinstance(flashcards, str) and flashcards.strip():
                flashcards_markdown = format_flashcards_markdown(flashcards)
            else:
                flashcards_markdown = "No flashcards generated."

            return (video_file, summary, timestamps_html, quiz_html, flashcards_markdown,
                    *radios_updates, submit_upd, feedback_upd, quizzes if quizzes else [])

        def grade_quizzes(*args):
            *user_answers, quizzes = args
            if not quizzes or not isinstance(quizzes, list) or len(quizzes) == 0:
                return "No quizzes to grade."

            result = []
            for idx, q in enumerate(quizzes):
                if idx >= MAX_QUESTIONS:
                    break
                user_ans = user_answers[idx]
                correct = q["answer"]
                if user_ans == correct:
                    result.append(f"**Question {idx+1}**: Correct! ✅")
                else:
                    result.append(f"**Question {idx+1}**: Incorrect ❌ (Correct answer: {correct})")

            return "\n".join(result)

        # Connect the Transcribe button to the on_transcribe function
        transcribe_button.click(
            on_transcribe,
            inputs=[video_input],
            outputs=[video_input, summary_output, timestamps_output, quiz_output, flashcards_output] +
                    quiz_radios + [submit_quiz_button, quiz_feedback, quizzes_state]
        )

        # Connect the Submit Quiz button to the grade_quizzes function
        submit_quiz_button.click(
            grade_quizzes,
            inputs=quiz_radios + [quizzes_state],
            outputs=quiz_feedback
        )

        demo.load()

    print("Launching Gradio interface...")
    demo.launch(allowed_paths=[VIDEO_DIR, AUDIO_DIR, TEXT_DIR], share=False)

if __name__ == "__main__":
    main()
