# src/app.py
import os
import shutil
import gradio as gr
import ffmpeg
import whisper
import openai
import warnings
import json

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
warnings.filterwarnings("ignore", message="You are using ⁠ torch.load ⁠ with ⁠ weights_only=False ⁠")

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
TRANSCRIPTION = ""


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
        #print(f"Error extracting audio: {e}")
        return False

def transcribe_audio(audio_file_path):
    """
    Transcribe audio file using Whisper with timestamps.
    """
    result = whisper_model.transcribe(audio_file_path)
    transcription = result["text"]
    TRANSCRIPTION = transcription
    segments = result["segments"]
    return transcription, segments

def process_video(video_file):

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
    #print(f"Transcription: {transcription[:100]}...")  # Limit to first 100 characters
    #print(f"Segments: {segments[:3]}...")  # Limit to first 3 segments

    # Summarize transcription
    try:
        summary = summarize_text(transcription, segments)
        #print(f"Summary generated: {summary[:100]}...")
    except Exception as e:
        #print(f"Error in summarizing text: {e}")
        summary = None

    # Generate quizzes
    try:
        quizzes = generate_quiz(transcription)
        
        #print(f"Quizzes generated: {quizzes[:100]}...")
    except Exception as e:
        #print(f"Error in generating quizzes: {e}")
        quizzes = None

    # Generate flashcards
    try:
        flashcards = generate_flashcards(transcription)
        #print(f"Flashcards generated: {flashcards[:100]}...")
    except Exception as e:
        #print(f"Error in generating flashcards: {e}")
        flashcards = None

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

def generate_gradio_quiz(quizzes):
    # Create a list of quiz question components
    quiz_elements = []

    for idx, question in enumerate(quizzes):
        #question = json.loads(question)
        # Create a column for each question with a heading and dropdown
        quiz_elements.append(gr.Markdown(f"### {question['question']}"))  # Add question as heading
        quiz_elements.append(gr.Dropdown(choices=question['options'], label=f"Choose an option for question {idx + 1}", interactive=True))

    return quiz_elements



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
        
        # Function to capture the answers after the user selects an option
        def capture_answers(*answers):
            
            quizzes = [
                    {
                        "question": "What type of cell division results in four daughter cells with half the number of chromosomes as the parent cell?",
                        "options": ["Mitosis", "Meiosis", "Binary fission", "Budding"],
                        "answer": "Meiosis"
                    },
                    {
                        "question": "Which of the following is NOT a phase of mitosis?",
                        "options": ["Prophase", "Metaphase", "Anaphase", "Telophase"],
                        "answer": "Meiosis"
                    },
                    {
                        "question": "During which phase of meiosis do homologous chromosomes pair up and exchange genetic material?",
                        "options": ["Prophase I", "Metaphase I", "Anaphase I", "Telophase I"],
                        "answer": "Prophase I"
                    },
                    {
                        "question": "What is the end result of meiosis in humans?",
                        "options": ["2 diploid cells", "4 haploid cells", "4 diploid cells", "2 haploid cells"],
                        "answer": "4 haploid cells"
                    },
                    {
                        "question": "Which of the following is responsible for genetic diversity in sexually reproducing organisms?",
                        "options": ["Mitosis", "Binary fission", "Meiosis", "Budding"],
                        "answer": "Meiosis"
                    },
                    {
                        "question": "What is the purpose of meiosis in multicellular organisms?",
                        "options": ["Growth and repair", "Asexual reproduction", "Increase genetic diversity", "Production of gametes"],
                        "answer": "Production of gametes"
                    },
                    {
                        "question": "Which stage of meiosis ensures that each daughter cell receives a complete set of chromosomes?",
                        "options": ["Prophase I", "Metaphase I", "Anaphase I", "Telophase I"],
                        "answer": "Anaphase I"
                    },
                    {
                        "question": "In meiosis, sister chromatids separate during which phase?",
                        "options": ["Prophase I", "Anaphase I", "Prophase II", "Anaphase II"],
                        "answer": "Anaphase II"
                    },
                    {
                        "question": "What is the significance of crossing over during meiosis?",
                        "options": ["Formation of gametes", "Increase genetic variation", "Creation of identical daughter cells", "Asexual reproduction"],
                        "answer": "Increase genetic variation"
                    },
                    {
                        "question": "Which of the following events occurs during meiosis but not during mitosis?",
                        "options": ["Synapsis", "Cytokinesis", "Chromatid separation", "Formation of spindle fibers"],
                        "answer": "Synapsis"
                    }
                ]
            result = []
            for idx, (answer, quiz) in enumerate(zip(answers, quizzes)):
                correct = "Correct!" if answer == quiz["answer"] else "Incorrect"
                result.append(f"Question {idx + 1}: {correct}")
            
            return "\n".join(result)  # Return results for display

        with gr.Tabs():
            with gr.Tab("Summary/Notes"):
                summary_output = gr.Markdown(label="Summary/Notes")
            with gr.Tab("Timestamps"):
                timestamps_output = gr.HTML(label="Timestamps")
            with gr.Tab("Quizzes"):
                # Generate list of dropdowns from JSON
                quizzes =generate_quiz(TRANSCRIPTION)

                quizzes = [
                    {
                        "question": "What type of cell division results in four daughter cells with half the number of chromosomes as the parent cell?",
                        "options": ["Mitosis", "Meiosis", "Binary fission", "Budding"],
                        "answer": "Meiosis"
                    },
                    {
                        "question": "Which of the following is NOT a phase of mitosis?",
                        "options": ["Prophase", "Metaphase", "Anaphase", "Telophase"],
                        "answer": "Meiosis"
                    },
                    {
                        "question": "During which phase of meiosis do homologous chromosomes pair up and exchange genetic material?",
                        "options": ["Prophase I", "Metaphase I", "Anaphase I", "Telophase I"],
                        "answer": "Prophase I"
                    },
                    {
                        "question": "What is the end result of meiosis in humans?",
                        "options": ["2 diploid cells", "4 haploid cells", "4 diploid cells", "2 haploid cells"],
                        "answer": "4 haploid cells"
                    },
                    {
                        "question": "Which of the following is responsible for genetic diversity in sexually reproducing organisms?",
                        "options": ["Mitosis", "Binary fission", "Meiosis", "Budding"],
                        "answer": "Meiosis"
                    },
                    {
                        "question": "What is the purpose of meiosis in multicellular organisms?",
                        "options": ["Growth and repair", "Asexual reproduction", "Increase genetic diversity", "Production of gametes"],
                        "answer": "Production of gametes"
                    },
                    {
                        "question": "Which stage of meiosis ensures that each daughter cell receives a complete set of chromosomes?",
                        "options": ["Prophase I", "Metaphase I", "Anaphase I", "Telophase I"],
                        "answer": "Anaphase I"
                    },
                    {
                        "question": "In meiosis, sister chromatids separate during which phase?",
                        "options": ["Prophase I", "Anaphase I", "Prophase II", "Anaphase II"],
                        "answer": "Anaphase II"
                    },
                    {
                        "question": "What is the significance of crossing over during meiosis?",
                        "options": ["Formation of gametes", "Increase genetic variation", "Creation of identical daughter cells", "Asexual reproduction"],
                        "answer": "Increase genetic variation"
                    },
                    {
                        "question": "Which of the following events occurs during meiosis but not during mitosis?",
                        "options": ["Synapsis", "Cytokinesis", "Chromatid separation", "Formation of spindle fibers"],
                        "answer": "Synapsis"
                    }
                ]
                dropdowns = generate_gradio_quiz(quizzes)
                
                # Render all dropdowns in a single column
                quiz_output = gr.Column(*dropdowns)
                
                # Collect answers and show selected answers in the textbox
                answer_output = gr.Textbox(label="Selected Answers")
                
                # Listen for changes in the dropdowns and call capture_answers
                gr.Button("Submit").click(
                    capture_answers,  # Function to handle selection
                    inputs=dropdowns,  # Pass list of dropdowns as inputs
                    outputs=answer_output
                )

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

    demo.launch(allowed_paths=[VIDEO_DIR, AUDIO_DIR, TEXT_DIR])

if __name__ == "__main__":
    main()
