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

# Suppress specific warnings we don't care about to declutter logs
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
warnings.filterwarnings("ignore", message="You are using torch.load with weights_only=False")

# Avoid potential multiprocessing issues
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from dotenv import load_dotenv
load_dotenv()

# Set up OpenAI API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

# Import custom core functionalities for our featuers from the application 
from core.summaries import summarize_text
from core.quizzes import generate_quiz, grade_quizzes
from core.flashcards import generate_flashcards, format_flashcards_markdown
from core.timestamps import generate_conceptual_timestamps

# Directory paths for organizing data
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
AUDIO_DIR = os.path.join(DATA_DIR, 'audio')
TEXT_DIR = os.path.join(ROOT_DIR, 'data', 'text_timestamps')
VIDEO_DIR = os.path.join(DATA_DIR, 'video')

# Ensure required directories exist
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

# Load the Whisper model for audio transcription
print("Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("Whisper model loaded.")

# Maximum number of questions for quizzes
MAX_QUESTIONS = 50 # Adjustable


def extract_audio(video_file_path, audio_file_path):
    """
    Extracts audio from a video file using FFmpeg.
    Args:
        video_file_path (str): Path to the input video file.
        audio_file_path (str): Path to save the extracted audio file.
    Returns:
        bool: True if extraction succeeds, False otherwise.
    """
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
    """
    Transcribes audio using Whisper.
    Args:
        audio_file_path (str): Path to the audio file.
    Returns:
        tuple: (transcription text, list of segments with timestamps).
    """
    print("Transcribing audio with Whisper...")
    result = whisper_model.transcribe(audio_file_path)
    transcription = result["text"]
    segments = result["segments"]
    print("Transcription complete.")
    return transcription, segments


def process_video(video_file):
    """
    Processes the uploaded video to extract and analyze its content.
    Args:
        video_file (str): Path to the uploaded video file.
    Returns:
        tuple: Paths and generated data (video path, summary, segments, quizzes, flashcards).
    """
    print(f"Processing video file: {video_file}")

    # Validate video file
    if not video_file or not os.path.isfile(video_file):
        print("Invalid video file path.")
        return "Error extracting audio.", None, None, None, None

    # Save the video to the designated directory
    video_filename = os.path.basename(video_file)
    video_path = os.path.join(VIDEO_DIR, video_filename)
    shutil.copy(video_file, video_path)
    print(f"Video saved to: {video_path}")

    # Extract audio from the video
    audio_filename = os.path.splitext(video_filename)[0] + ".wav"
    audio_path = os.path.join(AUDIO_DIR, audio_filename)
    success = extract_audio(video_path, audio_path)

    if not success:
        return "Error extracting audio.", None, None, None, None

    # Transcribe the audio
    try:
        transcription, segments = transcribe_audio(audio_path)
        print(f"Transcription snippet: {transcription[:100]}...")
        print(f"First 3 segments: {segments[:3]}...")
    except Exception as e:
        print(f"Error in transcription: {e}")
        traceback.print_exc()
        return "Error transcribing audio.", None, None, None, None

    # Generate summary
    try:
        summary = summarize_text(transcription, segments)
        print(f"Summary generated snippet: {summary[:100]}...")
    except Exception as e:
        print(f"Error in summarizing text: {e}")
        traceback.print_exc()
        summary = None

    # Generate quizzes
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

    # Generate flashcards
    try:
        flashcards = generate_flashcards(transcription)
    except Exception as e:
        print(f"Error in generating flashcards: {e}")
        traceback.print_exc()
        flashcards = None

    return video_path, summary, segments, quizzes, flashcards


def on_transcribe(video_file):
    """
    Handles the transcription and analysis process when a video is uploaded.
    Args:
        video_file: Uploaded video file.
    Returns:
        Outputs for the Gradio interface (summary, quizzes, flashcards, etc.).
    """
    # Step 1: Validate the uploaded video file
    if video_file is None:
        # If no file is provided, return placeholder values and hide all outputs
        return (gr.update(), "Please upload a video file.", "", "", "",
                *([gr.update(visible=False)]*MAX_QUESTIONS),
                gr.update(visible=False),
                gr.update(visible=False),
                [])

    # Determine the file path of the uploaded video
    if isinstance(video_file, dict) and "name" in video_file:
        # If the video file is provided as a dictionary, extract the 'name' key
        video_file_path = video_file["name"]
    else:
        # Otherwise, use the file path directly if it's a string
        video_file_path = video_file if isinstance(video_file, str) else None

    # Ensure the file path is valid
    if not video_file_path or not os.path.isfile(video_file_path):
        # If the file path is invalid, return placeholder values and hide all outputs
        return (gr.update(), "Please upload a video file.", "", "", "",
                *([gr.update(visible=False)]*MAX_QUESTIONS),
                gr.update(visible=False),
                gr.update(visible=False),
                [])

    # Step 2: Process the video file to generate outputs
    try:
        video_path, summary, segments, quizzes, flashcards = process_video(video_file_path)
    except Exception as e:
        # Handle errors that occur during video processing
        print("Error during process_video call:")
        traceback.print_exc()
        return (gr.update(), f"Error processing video: {e}", "", "", "",
                *([gr.update(visible=False)]*MAX_QUESTIONS),
                gr.update(visible=False),
                gr.update(visible=False),
                [])

    # Step 3: Generate conceptual timestamps from the summary and segments
    timestamps_html = generate_conceptual_timestamps(summary, segments) if segments else ""

    # Step 4: Handle quizzes
    if quizzes and isinstance(quizzes, list) and len(quizzes) > 0:
        # If quizzes are available, format them for display
        quiz_html = "<p>Select your answers and click Submit Quiz.</p>"
        quiz_count = min(len(quizzes), MAX_QUESTIONS) # Limit quizzes to MAX_QUESTIONS
        radios_updates = []
        for i in range(MAX_QUESTIONS):
            if i < quiz_count:
                q = quizzes[i]
                 # Update each radio button with the quiz question and options
                radios_updates.append(gr.update(label=f"Q{i+1}: {q['question']}", choices=q["options"], visible=True))
            else:
                # Hide unused radio buttons
                radios_updates.append(gr.update(visible=False))
        submit_upd = gr.update(visible=True) # Show the submit quiz button
        feedback_upd = gr.update(visible=True, value="") # Show the quiz feedback area
    else:
        # If no quizzes are available, hide quiz-related components
        quiz_html = "<p>No quizzes generated.</p>"
        radios_updates = [gr.update(visible=False) for _ in range(MAX_QUESTIONS)]
        submit_upd = gr.update(visible=False) # Hide submit quiz button
        feedback_upd = gr.update(visible=False) # Hide the quiz feedback area

    # Step 5: Handle flashcards
    if flashcards and isinstance(flashcards, str) and flashcards.strip():
        # If flashcards are available, format them in Markdown
        flashcards_markdown = format_flashcards_markdown(flashcards)
    else:
        # If no flashcards are available, provide a default message
        flashcards_markdown = "No flashcards generated."

    # Step 6: Return all outputs to the Gradio interface
    return (video_file, summary, timestamps_html, quiz_html, flashcards_markdown,
            *radios_updates, submit_upd, feedback_upd, quizzes if quizzes else [])


def main():
    """
    Main function to define and launch the Gradio interface.
    This serves as the entry point for the LecChurro application, providing users 
    with a graphical interface to upload lecture videos and receive processed outputs 
    (summaries, timestamps, quizzes, and flashcards).
    """
    # Create a Gradio Blocks interface for layout and interactivity
    with gr.Blocks() as demo:
        # Title of the application
        gr.Markdown("# LecChurro: From Lecture to Learning")

        # Input section for video uploads
        with gr.Row():
            with gr.Column(scale=1):
                 # Video upload component
                video_input = gr.Video(label="Upload Lecture Video", elem_id="main_video_player")
                # Button to trigger transcription and processing
                transcribe_button = gr.Button("Transcribe Now")

        # Output section with tabs for different functionalities
        with gr.Tabs():
            # Tab for displaying lecture summary/notes
            with gr.Tab("Summary/Notes"):
                summary_output = gr.Markdown(label="Summary/Notes")
            # Tab for displaying timestamps
            with gr.Tab("Timestamps"):
                timestamps_output = gr.HTML(label="Timestamps")
            # Tab for displaying quizzes
            with gr.Tab("Quizzes"):
                quiz_output = gr.HTML(label="Quizzes")
                # Dynamic creation of radio buttons for quiz options
                quiz_radios = []
                for i in range(MAX_QUESTIONS):
                    # Initialize radio buttons, initially hidden
                    r = gr.Radio(choices=[], label=f"Q{i+1}", visible=False)
                    quiz_radios.append(r)
                    # Button to submit quiz answers, initially hidden
                submit_quiz_button = gr.Button("Submit Quiz", visible=False)
                # Feedback section for quiz results, initially hidden
                quiz_feedback = gr.Markdown("", visible=False)
            # Tab for displaying flashcards
            with gr.Tab("Flashcards"):
                # Use Markdown to display interactive flashcards
                flashcards_output = gr.Markdown(label="Flashcards")

        # State variable to hold quiz data across interactions
        quizzes_state = gr.State()

        # Define interaction: Connect the Transcribe button to the on_transcribe function
        transcribe_button.click(
            on_transcribe, # Function to call when button is clicked
            inputs=[video_input],
            outputs=[video_input, summary_output, timestamps_output, quiz_output, flashcards_output] +
                    quiz_radios + [submit_quiz_button, quiz_feedback, quizzes_state]
        )

        # Define interaction: Connect the Submit Quiz button to the grade_quizzes function
        submit_quiz_button.click(
            grade_quizzes, # Function to call when button is clicked
            inputs=quiz_radios + [quizzes_state],
            outputs=quiz_feedback
        )

        # Preload and set up the interface
        demo.load()

    print("Launching Gradio interface...")

    # Launch the Gradio application with restricted file access paths
    demo.launch(allowed_paths=[VIDEO_DIR, AUDIO_DIR, TEXT_DIR], share=False)


if __name__ == "__main__":
    # Call main to start app
    main()
