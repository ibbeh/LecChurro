# src/app.py
import os
import shutil
import gradio as gr
import ffmpeg
import whisper
import openai
import warnings
import json
import traceback

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
warnings.filterwarnings("ignore", message="You are using torch.load with weights_only=False")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Import existing functions
from core.summarization.summarize_text import summarize_text
from core.quiz_generation.quiz_generation_transcription import generate_quiz
from core.flashcards.flashcards_generation import generate_flashcards
from core.timestamps.generate_conceptual_timestamps import generate_conceptual_timestamps

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

print("Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("Whisper model loaded.")


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
        #print(f"Error extracting audio: {e}")
        return False


def transcribe_audio(audio_file_path):
    print("Transcribing audio with Whisper...")
    result = whisper_model.transcribe(audio_file_path)
    transcription = result["text"]
    TRANSCRIPTION = transcription
    segments = result["segments"]
    print("Transcription complete.")
    return transcription, segments


def process_video(video_file):
    print(f"Processing video file: {video_file}")
    # The user's working version assumed video_file is a file path directly.
    # According to the working code, video_file is already a file path (string),
    # not a dict. We'll assume that remains true.
    if not video_file or not os.path.isfile(video_file):
        print("Invalid video file path.")
        return "Error extracting audio.", None, None, None, None

    video_filename = os.path.basename(video_file)
    video_path = os.path.join(VIDEO_DIR, video_filename)
    shutil.copy(video_file, video_path)

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
        quizzes = generate_quiz(transcription)
        print(f"Quiz generated snippet: {quizzes[:100]}...")
    except Exception as e:
        print(f"Error in generating quizzes: {e}")
        traceback.print_exc()
        quizzes = None

    try:
        flashcards = generate_flashcards(transcription)
        #print(f"Flashcards generated: {flashcards[:100]}...")
    except Exception as e:
        #print(f"Error in generating flashcards: {e}")
        print(f"Flashcards generated snippet: {flashcards[:100]}...")
    except Exception as e:
        print(f"Error in generating flashcards: {e}")
        traceback.print_exc()
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
def format_flashcards_html(flashcards_text):
    # Parse flashcards in the format:
    # Front: ...
    # Back: ...
    cards = flashcards_text.strip().split('\n')
    flashcards_list = []
    current_front = None
    current_back = None

    for line in cards:
        line = line.strip()
        if line.lower().startswith("front:"):
            current_front = line[len("Front:"):].strip()
        elif line.lower().startswith("back:"):
            current_back = line[len("Back:"):].strip()
            if current_front and current_back:
                flashcards_list.append((current_front, current_back))
                current_front = None
                current_back = None

    html = ""
    # Build flip-card-like behavior using Show/Hide
    for front, back in flashcards_list:
        html += f"""
        <div class="flashcard" style="margin-bottom:20px; border:1px solid #ccc; padding:10px; width:300px;">
          <div class="front"><b>Front:</b> {front}</div>
          <div class="back" style="display:none; margin-top:10px;"><b>Back:</b> {back}</div>
          <button class="toggle-answer" style="margin-top:10px;">Show Answer</button>
        </div>
        """
    return html


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

        with gr.Row():
            with gr.Column(scale=1):
                # Keep it as in the working version: just a Video component
                video_input = gr.Video(label="Upload Lecture Video", elem_id="main_video_player")
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
            # According to the working code, video_file is a file path directly
            # If gr.Video returns a dict, we handle that:
            # If previously working code was that video_file = None means no file,
            # we keep same logic:
            if video_file is None:
                return gr.update(), "Please upload a video file.", "", "", ""

            # If video_file is a dict from gr.Video, extract 'name'
            if isinstance(video_file, dict) and "name" in video_file:
                video_file_path = video_file["name"]
            else:
                # If it's already a string path (the user said it worked before),
                # just use it directly
                video_file_path = video_file if isinstance(video_file, str) else None

            if not video_file_path or not os.path.isfile(video_file_path):
                return gr.update(), "Please upload a video file.", "", "", ""

            try:
                video_path, summary, segments, quizzes, flashcards = process_video(video_file_path)
            except Exception as e:
                print("Error during process_video call:")
                traceback.print_exc()
                return gr.update(), f"Error processing video: {e}", "", "", ""

            print("Generating conceptual timestamps...")
            timestamps_html = generate_conceptual_timestamps(summary, segments)
            print("Conceptual timestamps generated successfully.")

            quiz_html = f"<pre>{quizzes}</pre>" if quizzes else "<p>No quizzes generated.</p>"

            if flashcards:
                flashcards_html = format_flashcards_html(flashcards)
            else:
                flashcards_html = "<p>No flashcards generated.</p>"

            # Return updates
            # Return the video_file as is (since it was working)
            return video_file, summary, timestamps_html, quiz_html, flashcards_html




        transcribe_button.click(
            on_transcribe,
            inputs=[video_input],
            outputs=[video_input, summary_output, timestamps_output, quiz_output, flashcards_output]
        )

        # Attach the JS after the interface loads to handle timestamps and flashcards toggle
        demo.load(js="""
function my_func() {
  // Handle timestamp links
  const links = document.querySelectorAll(".timestamp-link");
  links.forEach(link => {
    link.addEventListener("click", e => {
      e.preventDefault();
      const time = parseFloat(e.currentTarget.getAttribute("data-time"));
      const video = document.querySelector('#main_video_player video');
      if (video && !isNaN(time)) {
        video.currentTime = time;
        video.play();
      }
    });
  });

  // Handle flashcard show/hide answer
  const flashcardButtons = document.querySelectorAll(".toggle-answer");
  flashcardButtons.forEach(btn => {
    btn.addEventListener("click", e => {
      e.preventDefault();
      const card = e.currentTarget.closest(".flashcard");
      const back = card.querySelector(".back");
      if (back.style.display === "none") {
        back.style.display = "block";
        e.currentTarget.textContent = "Hide Answer";
      } else {
        back.style.display = "none";
        e.currentTarget.textContent = "Show Answer";
      }
    });
  });
}
my_func();
""")

    print("Launching Gradio interface...")
    demo.launch(allowed_paths=[VIDEO_DIR, AUDIO_DIR, TEXT_DIR], share=False)


if __name__ == "__main__":
    main()
