# src/app.py
import os
import shutil
import gradio as gr
import ffmpeg
import whisper
import openai

# Set the environment variable to allow duplicate OpenMP runtimes (temporary workaround)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Import existing functions from core modules
from core.summarization.summarize_text import summarize_text
from core.quiz_generation.quiz_generation_transcription import generate_quiz
from core.flashcards.flashcards_generation import generate_flashcards

# Define paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
AUDIO_DIR = os.path.join(DATA_DIR, 'audio')
TEXT_DIR = os.path.join(DATA_DIR, 'text_timestamps')
VIDEO_DIR = os.path.join(DATA_DIR, 'video')

# Ensure directories exist
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

# Load Whisper model
whisper_model = whisper.load_model("base")  # Use "small", "medium", or "large" as needed

def extract_audio(video_file_path, audio_file_path):
    """
    Extract audio from video file and save as WAV.
    """
    try:
        (
            ffmpeg
            .input(video_file_path)
            .output(audio_file_path, format='wav', acodec='pcm_s16le', ac=1, ar='16k')
            .overwrite_output()
            .run(quiet=True)
        )
        return True
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return False

def transcribe_audio(audio_file_path):
    """
    Transcribe audio file using Whisper with timestamps.
    """
    result = whisper_model.transcribe(audio_file_path)
    transcription = result["text"]
    segments = result["segments"]
    return transcription, segments

def process_video(video_file):
    """
    Process the uploaded video: extract audio, transcribe, summarize, generate quizzes.
    """
    # Save video file to VIDEO_DIR
    video_filename = os.path.basename(video_file)
    video_path = os.path.join(VIDEO_DIR, video_filename)
    shutil.copy(video_file, video_path)

    # Extract audio
    audio_filename = os.path.splitext(video_filename)[0] + ".wav"
    audio_path = os.path.join(AUDIO_DIR, audio_filename)
    success = extract_audio(video_path, audio_path)
    if not success:
        return "Error extracting audio.", None, None, None, None

    # Transcribe audio
    transcription, segments = transcribe_audio(audio_path)

    # Save transcription
    transcription_filename = os.path.splitext(video_filename)[0] + ".txt"
    transcription_path = os.path.join(TEXT_DIR, transcription_filename)
    with open(transcription_path, "w", encoding="utf-8") as f:
        f.write(transcription)

    # Generate a comprehensive summary using the entire transcription
    summary = summarize_text(transcription, segments)

    # Generate quizzes
    quizzes = generate_quiz(transcription)

    # Generate flashcards
    flashcards = generate_flashcards(transcription)

    return video_path, summary, segments, quizzes, flashcards

def format_timestamps(segments):
    """
    Format timestamps data for display and make them clickable.
    """
    timestamps_data = []
    for segment in segments:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]
        # Create clickable text
        clickable_text = f"<a href='javascript:void(0);' onclick='seekVideo({start});'>{text}</a>"
        timestamps_data.append({"Start Time": start, "End Time": end, "Text": clickable_text})
    return timestamps_data

def main():
    with gr.Blocks() as demo:
        gr.Markdown("# LecChurro: From Lecture to Learning")

        # JavaScript code to handle seeking
        seek_js = """
        <script>
        function seekVideo(time) {
            var video = document.querySelector('video');
            if (video) {
                video.currentTime = time;
            }
        }
        </script>
        """

        # Include the JavaScript in an HTML component
        gr.HTML(seek_js)

        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="Upload Lecture Video", elem_id="lecture_video")
                transcribe_button = gr.Button("Transcribe Now")

        with gr.Tabs():
            with gr.Tab("Summary/Notes"):
                summary_output = gr.HTML(label="Summary/Notes")
            with gr.Tab("Timestamps"):
                timestamps_output = gr.HTML(label="Timestamps")
            with gr.Tab("Quizzes"):
                quiz_output = gr.HTML(label="Quizzes")
            with gr.Tab("Flashcards"):
                flashcards_output = gr.HTML(label="Flashcards")

        def on_transcribe(video_file):
            if video_file is None:
                # Return 5 outputs: video_player (update), summary, timestamps, quiz, flashcards
                return gr.update(), "Please upload a video file.", "", "", ""

            try:
                video_path, summary, segments, quizzes, flashcards = process_video(video_file)
            except Exception as e:
                return gr.update(), f"Error processing video: {e}", "", "", ""

            # Create clickable timestamps
            timestamps_html = ""
            for i, segment in enumerate(segments):
                if i % 5 == 0:  # Adjust to capture meaningful intervals
                    start = segment["start"]
                    end = segment["end"]
                    text = segment["text"]
                    timestamp = f"{start:.2f}s - {end:.2f}s"
                    clickable_text = f"<a href='javascript:void(0);' onclick='seekVideo({start});'>{text}</a>"
                    timestamps_html += f"<p><b>[{timestamp}]</b> {clickable_text}</p>"

            # Format quizzes and flashcards as HTML
            quiz_html = f"<pre>{quizzes}</pre>"
            flashcards_html = f"<pre>{flashcards}</pre>"

            # Update the video_player to point to the new video
            return video_path, summary, timestamps_html, quiz_html, flashcards_html

        transcribe_button.click(
            on_transcribe,
            inputs=[video_input],
            outputs=[video_input, summary_output, timestamps_output, quiz_output, flashcards_output]
        )

    demo.launch()

if __name__ == "__main__":
    main()
